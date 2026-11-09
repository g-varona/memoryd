use std::sync::{Arc, Mutex, RwLock};

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::embed::Embedder;
use crate::index::{Filter, VectorIndex, rrf};
use crate::store::{Memory, Store};

/// Owns the database, the vector index and the embedder. All methods block, so async
/// callers should run them on `tokio::task::spawn_blocking`.
///
/// Writes go through one `Mutex<Store>`, so there is a single SQLite writer and
/// "database is locked" cannot happen inside the server.
pub struct Engine {
    store: Mutex<Store>,
    index: RwLock<VectorIndex>,
    embedder: Arc<dyn Embedder>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, schemars::JsonSchema, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Meaning and keywords combined (default).
    #[default]
    Hybrid,
    /// Meaning only.
    Semantic,
    /// Keywords only (substring match, works for Japanese).
    Keyword,
}

#[derive(Debug, Serialize)]
pub struct Hit {
    #[serde(flatten)]
    pub memory: Memory,
    /// Rank score within this result list; comparable between hits, not between searches.
    pub score: f32,
    /// Cosine similarity between the query and this memory (semantic and hybrid modes only).
    /// Absolute, so it says how related the hit really is: ~0.3-0.8 is related, ~0.0-0.15 is not.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub similarity: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct Added {
    pub id: i64,
    pub duplicate: bool,
}

/// How many candidates each ranker contributes before fusion.
const CANDIDATES: usize = 50;

/// Semantic candidates below this cosine are dropped, so a search on a topic with no
/// related memories returns nothing instead of the least-unrelated `limit` memories.
/// With all-MiniLM-L6-v2, paraphrases score ~0.3-0.8 and unrelated sentences ~0.0-0.15.
const MIN_SIMILARITY: f32 = 0.2;

impl Engine {
    pub fn new(store: Store, embedder: Arc<dyn Embedder>) -> Result<Self> {
        let mut index = VectorIndex::new(embedder.dim());
        for v in store.all_vectors()? {
            if v.embedding.len() != embedder.dim() {
                bail!("memory {} has a {}-d vector but the model is {}-d", v.id, v.embedding.len(), embedder.dim());
            }
            index.upsert(v.id, v.project, v.tags, v.embedding);
        }
        Ok(Self { store: Mutex::new(store), index: RwLock::new(index), embedder })
    }

    pub fn len(&self) -> usize {
        self.index.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn add(&self, content: &str, tags: Vec<String>, project: Option<String>) -> Result<Added> {
        let content = content.trim();
        if content.is_empty() {
            bail!("content is empty");
        }
        // Embed before taking the lock so a slow embed never blocks other writers.
        let vector = self.embedder.embed_one(content)?;
        let (id, duplicate) = self.store.lock().unwrap().insert(content, &tags, project.as_deref(), &vector)?;
        if !duplicate {
            self.index.write().unwrap().upsert(id, project, tags, vector);
        }
        Ok(Added { id, duplicate })
    }

    pub fn search(&self, query: &str, limit: usize, mode: SearchMode, filter: &Filter) -> Result<Vec<Hit>> {
        let limit = limit.clamp(1, 50);
        // Keyword mode never embeds the query (it exists for text the model cannot read).
        let query_vector = match mode {
            SearchMode::Keyword => None,
            _ => Some(self.embedder.embed_one(query)?),
        };
        let semantic = |q: &[f32]| -> Vec<(i64, f32)> {
            let mut hits = self.index.read().unwrap().top_k(q, CANDIDATES.max(limit), filter);
            hits.retain(|(_, cos)| *cos >= MIN_SIMILARITY);
            hits
        };
        let keyword = || -> Result<Vec<i64>> {
            self.store.lock().unwrap().keyword(query, filter.project.as_deref(), &filter.tags, CANDIDATES.max(limit))
        };

        let ranked: Vec<(i64, f32)> = match (mode, &query_vector) {
            (SearchMode::Semantic, Some(q)) => semantic(q).into_iter().take(limit).collect(),
            (SearchMode::Hybrid, Some(q)) => {
                let sem: Vec<i64> = semantic(q).into_iter().map(|(id, _)| id).collect();
                rrf(&[sem, keyword()?], limit)
            }
            _ => keyword()?.into_iter().take(limit).enumerate().map(|(rank, id)| (id, 1.0 / (rank as f32 + 1.0))).collect(),
        };

        let store = self.store.lock().unwrap();
        let index = self.index.read().unwrap();
        let mut hits = Vec::with_capacity(ranked.len());
        for (id, score) in ranked {
            if let Some(memory) = store.get(id)? {
                let similarity = query_vector.as_deref().and_then(|q| index.similarity(id, q));
                hits.push(Hit { memory, score, similarity });
            }
        }
        Ok(hits)
    }

    pub fn get(&self, id: i64) -> Result<Option<Memory>> {
        self.store.lock().unwrap().get(id)
    }

    pub fn list(&self, project: Option<&str>, tag: Option<&str>, limit: usize) -> Result<Vec<Memory>> {
        self.store.lock().unwrap().list(project, tag, limit.clamp(1, 200))
    }

    pub fn update(&self, id: i64, content: Option<&str>, tags: Option<Vec<String>>) -> Result<Option<Memory>> {
        let content = content.map(str::trim);
        if content == Some("") {
            bail!("content is empty");
        }
        let vector = content.map(|c| self.embedder.embed_one(c)).transpose()?;
        let updated = self.store.lock().unwrap().update(id, content, tags.as_deref(), vector.as_deref())?;
        if let Some(m) = &updated {
            let mut index = self.index.write().unwrap();
            match vector {
                Some(v) => index.upsert(m.id, m.project.clone(), m.tags.clone(), v),
                None => index.set_tags(m.id, m.tags.clone()),
            }
        }
        Ok(updated)
    }

    pub fn delete(&self, id: i64) -> Result<bool> {
        let deleted = self.store.lock().unwrap().delete(id)?;
        if deleted {
            self.index.write().unwrap().remove(id);
        }
        Ok(deleted)
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::HashEmbedder;

    fn engine() -> Engine {
        Engine::new(Store::open_in_memory().unwrap(), Arc::new(HashEmbedder { dim: 64 })).unwrap()
    }

    #[test]
    fn add_search_update_delete() {
        let e = engine();
        let a = e.add("Database failover runbook lives in the wiki", vec!["dr".into()], Some("billing".into())).unwrap();
        e.add("Lunch on Friday is ramen", vec![], None).unwrap();
        assert!(e.add("Database failover runbook lives in the wiki", vec![], None).unwrap().duplicate);
        assert_eq!(e.len(), 2);

        for mode in [SearchMode::Hybrid, SearchMode::Semantic, SearchMode::Keyword] {
            let hits = e.search("failover runbook", 5, mode, &Filter::default()).unwrap();
            assert_eq!(hits[0].memory.id, a.id, "{mode:?}");
        }
        let filtered = e
            .search("failover", 5, SearchMode::Hybrid, &Filter { project: None, tags: vec!["nope".into()] })
            .unwrap();
        assert!(filtered.is_empty());

        e.update(a.id, Some("Database DR runbook moved to the handbook"), None).unwrap();
        let hits = e.search("handbook", 5, SearchMode::Keyword, &Filter::default()).unwrap();
        assert_eq!(hits[0].memory.id, a.id);

        assert!(e.delete(a.id).unwrap());
        assert_eq!(e.len(), 1);
        assert!(e.search("handbook", 5, SearchMode::Hybrid, &Filter::default()).unwrap().iter().all(|h| h.memory.id != a.id));
    }

    #[test]
    fn unrelated_queries_return_nothing_and_hits_carry_similarity() {
        // A wide hash space so unrelated words do not collide into a spurious similarity.
        let e = Engine::new(Store::open_in_memory().unwrap(), Arc::new(HashEmbedder { dim: 4096 })).unwrap();
        e.add("Database failover runbook lives in the wiki", vec![], None).unwrap();
        e.add("Lunch on Friday is ramen", vec![], None).unwrap();
        // No shared words: cosine is 0 under the bag-of-words test embedder, below the floor.
        let unrelated = Filter::default();
        assert!(e.search("quarterly budget review", 5, SearchMode::Semantic, &unrelated).unwrap().is_empty());
        assert!(e.search("quarterly budget review", 5, SearchMode::Hybrid, &unrelated).unwrap().is_empty());

        let hits = e.search("failover runbook", 5, SearchMode::Hybrid, &unrelated).unwrap();
        assert_eq!(hits.len(), 1, "only the related memory passes the floor: {hits:?}");
        assert!(hits[0].similarity.unwrap() >= MIN_SIMILARITY);
        let hits = e.search("failover", 5, SearchMode::Keyword, &unrelated).unwrap();
        assert_eq!(hits[0].similarity, None);
    }

    #[test]
    fn index_is_rebuilt_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.db");
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder { dim: 32 });
        let id = Engine::new(Store::open(&path).unwrap(), embedder.clone()).unwrap().add("persisted note", vec![], None).unwrap().id;
        let reopened = Engine::new(Store::open(&path).unwrap(), embedder).unwrap();
        assert_eq!(reopened.search("persisted", 1, SearchMode::Semantic, &Filter::default()).unwrap()[0].memory.id, id);
    }
}
