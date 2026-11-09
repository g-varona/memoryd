use std::collections::HashMap;

/// Every vector kept in RAM and scanned on each search. At 10k memories x 384 floats
/// that is 15MB and well under a millisecond per query, so no ANN structure is needed.
pub struct VectorIndex {
    dim: usize,
    entries: Vec<Entry>,
    position: HashMap<i64, usize>,
}

struct Entry {
    id: i64,
    project: Option<String>,
    tags: Vec<String>,
    vector: Vec<f32>,
}

/// Restricts which memories a search may return.
#[derive(Debug, Default, Clone)]
pub struct Filter {
    pub project: Option<String>,
    /// A memory matches if it has every one of these tags.
    pub tags: Vec<String>,
}

impl Filter {
    pub fn matches(&self, project: Option<&str>, tags: &[String]) -> bool {
        self.project.as_deref().is_none_or(|p| project == Some(p))
            && self.tags.iter().all(|t| tags.contains(t))
    }
}

impl VectorIndex {
    pub fn new(dim: usize) -> Self {
        Self { dim, entries: Vec::new(), position: HashMap::new() }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn upsert(&mut self, id: i64, project: Option<String>, tags: Vec<String>, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim, "vector has wrong dimension");
        let entry = Entry { id, project, tags, vector };
        match self.position.get(&id) {
            Some(&i) => self.entries[i] = entry,
            None => {
                self.position.insert(id, self.entries.len());
                self.entries.push(entry);
            }
        }
    }

    /// Updates tags without touching the vector.
    pub fn set_tags(&mut self, id: i64, tags: Vec<String>) {
        if let Some(&i) = self.position.get(&id) {
            self.entries[i].tags = tags;
        }
    }

    pub fn remove(&mut self, id: i64) {
        if let Some(i) = self.position.remove(&id) {
            self.entries.swap_remove(i);
            if let Some(moved) = self.entries.get(i) {
                self.position.insert(moved.id, i);
            }
        }
    }

    /// The `k` most similar memories, best first, as (id, cosine similarity).
    pub fn top_k(&self, query: &[f32], k: usize, filter: &Filter) -> Vec<(i64, f32)> {
        let mut scored: Vec<(i64, f32)> = self
            .entries
            .iter()
            .filter(|e| filter.matches(e.project.as_deref(), &e.tags))
            .map(|e| (e.id, dot(query, &e.vector)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        scored.truncate(k);
        scored
    }

    /// Cosine similarity between `query` and one memory's vector, if the memory is indexed.
    pub fn similarity(&self, id: i64, query: &[f32]) -> Option<f32> {
        self.position.get(&id).map(|&i| dot(query, &self.entries[i].vector))
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Reciprocal rank fusion: merges several ranked id lists into one.
/// Each list contributes 1 / (k + rank); k = 60 is the value from the original paper.
pub fn rrf(lists: &[Vec<i64>], limit: usize) -> Vec<(i64, f32)> {
    const K: f32 = 60.0;
    let mut scores: HashMap<i64, f32> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *scores.entry(*id).or_default() += 1.0 / (K + rank as f32 + 1.0);
        }
    }
    let mut fused: Vec<(i64, f32)> = scores.into_iter().collect();
    fused.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_orders_by_similarity_and_filters() {
        let mut idx = VectorIndex::new(2);
        idx.upsert(1, Some("a".into()), vec![], vec![1.0, 0.0]);
        idx.upsert(2, Some("b".into()), vec!["t".into()], vec![0.6, 0.8]);
        idx.upsert(3, None, vec![], vec![0.0, 1.0]);
        let all = idx.top_k(&[1.0, 0.0], 10, &Filter::default());
        assert_eq!(all.iter().map(|x| x.0).collect::<Vec<_>>(), vec![1, 2, 3]);
        let only_b = idx.top_k(&[1.0, 0.0], 10, &Filter { project: Some("b".into()), tags: vec![] });
        assert_eq!(only_b.iter().map(|x| x.0).collect::<Vec<_>>(), vec![2]);
        let tagged = idx.top_k(&[1.0, 0.0], 10, &Filter { project: None, tags: vec!["t".into()] });
        assert_eq!(tagged.len(), 1);
    }

    #[test]
    fn remove_keeps_positions_consistent() {
        let mut idx = VectorIndex::new(1);
        for id in 1..=3 {
            idx.upsert(id, None, vec![], vec![id as f32]);
        }
        idx.remove(1);
        idx.upsert(3, None, vec![], vec![9.0]);
        assert_eq!(idx.len(), 2);
        let ids: Vec<i64> = idx.top_k(&[1.0], 10, &Filter::default()).iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![3, 2]);
    }

    #[test]
    fn rrf_rewards_agreement() {
        let fused = rrf(&[vec![1, 2, 3], vec![3, 1]], 10);
        let ids: Vec<i64> = fused.iter().map(|x| x.0).collect();
        assert_eq!(ids, vec![1, 3, 2]);
        assert_eq!(rrf(&[vec![1, 2, 3]], 2).len(), 2);
    }
}
