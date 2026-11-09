use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use rusqlite::{Connection, OpenFlags, OptionalExtension, Row, params};
use serde::Serialize;
use sha2::{Digest, Sha256};

const SCHEMA_VERSION: i64 = 1;

/// Created first, on its own, so the schema version can be checked before any other DDL runs.
const META_SCHEMA: &str = "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);";

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY,
    content      TEXT    NOT NULL,
    tags         TEXT    NOT NULL DEFAULT '[]',  -- JSON array of strings
    project      TEXT,
    content_hash TEXT    NOT NULL UNIQUE,
    embedding    BLOB    NOT NULL,               -- little-endian f32 x dim
    created_at   INTEGER NOT NULL,               -- unix seconds
    updated_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_project ON memories(project, created_at);

-- Trigram tokenizer: substring matching that also works for Japanese text.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, content='memories', content_rowid='id', tokenize='trigram'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags) VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
"#;

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// A row plus its vector, used to build the in-RAM index at startup.
pub struct StoredVector {
    pub id: i64,
    pub project: Option<String>,
    pub tags: Vec<String>,
    pub embedding: Vec<f32>,
}

#[derive(Debug)]
pub struct Store {
    conn: Connection,
}

impl Store {
    /// Opens the database, creating it (and its directory) if needed.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        Self::init(Connection::open(path)?)
    }

    /// Opens an existing database and fails if there is none, so read-only commands
    /// (`search`, `backup`) cannot silently create an empty store at a mistyped path.
    pub fn open_existing(path: &Path) -> Result<Self> {
        if !path.is_file() {
            bail!("no database at {}", path.display());
        }
        let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX;
        Self::init(Connection::open_with_flags(path, flags)?)
    }

    pub fn open_in_memory() -> Result<Self> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self> {
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.pragma_update(None, "synchronous", "NORMAL")?;
        conn.pragma_update(None, "foreign_keys", "ON")?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        // Check the version before touching anything else, so an older binary never
        // adds its tables or triggers to a database written by a newer schema.
        conn.execute_batch(META_SCHEMA)?;
        let version: Option<String> =
            conn.query_row("SELECT value FROM meta WHERE key = 'schema_version'", [], |r| r.get(0)).optional()?;
        match version {
            None => {
                conn.execute("INSERT INTO meta VALUES ('schema_version', ?1)", [SCHEMA_VERSION.to_string()])?;
            }
            Some(v) if v == SCHEMA_VERSION.to_string() => {}
            Some(v) => bail!("database schema version {v} is not supported (expected {SCHEMA_VERSION})"),
        }
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Stores a memory. Returns the existing id and `true` if the same content is already stored.
    pub fn insert(&self, content: &str, tags: &[String], project: Option<&str>, embedding: &[f32]) -> Result<(i64, bool)> {
        let hash = content_hash(content);
        if let Some(id) = self
            .conn
            .query_row("SELECT id FROM memories WHERE content_hash = ?1", [&hash], |r| r.get(0))
            .optional()?
        {
            return Ok((id, true));
        }
        let now = now();
        self.conn.execute(
            "INSERT INTO memories (content, tags, project, content_hash, embedding, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6)",
            params![content.trim(), serde_json::to_string(tags)?, project, hash, to_blob(embedding), now],
        )?;
        Ok((self.conn.last_insert_rowid(), false))
    }

    pub fn get(&self, id: i64) -> Result<Option<Memory>> {
        Ok(self
            .conn
            .query_row(&format!("SELECT {COLUMNS} FROM memories WHERE id = ?1"), [id], row_to_memory)
            .optional()?)
    }

    /// Replaces content and/or tags. Pass a new embedding whenever content changes.
    pub fn update(&self, id: i64, content: Option<&str>, tags: Option<&[String]>, embedding: Option<&[f32]>) -> Result<Option<Memory>> {
        let Some(current) = self.get(id)? else { return Ok(None) };
        let content = content.map(str::trim).unwrap_or(&current.content);
        let tags = tags.unwrap_or(&current.tags);
        let hash = content_hash(content);
        let clash: Option<i64> = self
            .conn
            .query_row("SELECT id FROM memories WHERE content_hash = ?1 AND id != ?2", params![hash, id], |r| r.get(0))
            .optional()?;
        if let Some(other) = clash {
            bail!("memory {other} already has this content");
        }
        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "UPDATE memories SET content = ?1, tags = ?2, content_hash = ?3, updated_at = ?4 WHERE id = ?5",
            params![content, serde_json::to_string(tags)?, hash, now(), id],
        )?;
        if let Some(embedding) = embedding {
            tx.execute("UPDATE memories SET embedding = ?1 WHERE id = ?2", params![to_blob(embedding), id])?;
        }
        tx.commit()?;
        self.get(id)
    }

    pub fn delete(&self, id: i64) -> Result<bool> {
        Ok(self.conn.execute("DELETE FROM memories WHERE id = ?1", [id])? > 0)
    }

    /// Most recent first.
    pub fn list(&self, project: Option<&str>, tag: Option<&str>, limit: usize) -> Result<Vec<Memory>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {COLUMNS} FROM memories WHERE (?1 IS NULL OR project = ?1)
               AND (?2 IS NULL OR EXISTS (SELECT 1 FROM json_each(tags) WHERE value = ?2))
             ORDER BY created_at DESC, id DESC LIMIT ?3"
        ))?;
        let rows = stmt.query_map(params![project, tag, limit as i64], row_to_memory)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    /// Full-text search. Returns ids, best match first. Only memories in `project` (if given)
    /// that carry every tag in `tags` are considered; filtering happens in SQL so the
    /// `limit` applies after the filter, not before.
    ///
    /// Terms of 3+ characters go through FTS5 (ranked by BM25). Shorter terms, common in
    /// Japanese (e.g. 東京), cannot use the trigram index, so they fall back to a LIKE scan
    /// and their matches are appended after the FTS5 results. Short pure-ASCII terms
    /// ("I", "an", "to") are stopwords that would match nearly every row, so they are ignored.
    pub fn keyword(&self, query: &str, project: Option<&str>, tags: &[String], limit: usize) -> Result<Vec<i64>> {
        let tags = serde_json::to_string(tags)?;
        let mut ids: Vec<i64> = Vec::new();
        if let Some(fts_query) = fts_query(query) {
            let mut stmt = self.conn.prepare(&format!(
                "SELECT m.id FROM memories_fts f JOIN memories m ON m.id = f.rowid
                 WHERE memories_fts MATCH ?1 AND (?2 IS NULL OR m.project = ?2) AND {HAS_ALL_TAGS}
                 ORDER BY bm25(memories_fts) LIMIT ?4"
            ))?;
            let rows = stmt.query_map(params![fts_query, project, tags, limit as i64], |r| r.get(0))?;
            ids = rows.collect::<rusqlite::Result<_>>()?;
        }
        let short: Vec<&str> = query.split_whitespace().filter(|t| t.chars().count() < 3 && !t.is_ascii()).collect();
        if !short.is_empty() && ids.len() < limit {
            let mut stmt = self.conn.prepare(&format!(
                "SELECT m.id FROM memories m
                 WHERE m.content LIKE ?1 ESCAPE '\\' AND (?2 IS NULL OR m.project = ?2) AND {HAS_ALL_TAGS}
                 ORDER BY m.created_at DESC, m.id DESC LIMIT ?4"
            ))?;
            for term in short {
                let pattern = format!("%{}%", term.replace('\\', "\\\\").replace('%', "\\%").replace('_', "\\_"));
                for id in stmt.query_map(params![pattern, project, tags, limit as i64], |r| r.get::<_, i64>(0))? {
                    let id = id?;
                    if ids.len() < limit && !ids.contains(&id) {
                        ids.push(id);
                    }
                }
            }
        }
        Ok(ids)
    }

    pub fn all_vectors(&self) -> Result<Vec<StoredVector>> {
        let mut stmt = self.conn.prepare("SELECT id, project, tags, embedding FROM memories")?;
        let rows = stmt.query_map([], |r| {
            Ok(StoredVector {
                id: r.get(0)?,
                project: r.get(1)?,
                tags: parse_tags(&r.get::<_, String>(2)?),
                embedding: from_blob(&r.get::<_, Vec<u8>>(3)?),
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self.conn.query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))?)
    }

    /// Writes a consistent copy of the database to `dest`, safe while the server is running.
    pub fn backup(&self, dest: &Path) -> Result<()> {
        if dest.exists() {
            bail!("{} already exists", dest.display());
        }
        self.conn.execute("VACUUM INTO ?1", [dest.to_string_lossy()])?;
        Ok(())
    }
}

const COLUMNS: &str = "id, content, tags, project, created_at, updated_at";

/// SQL predicate: memory `m` has every tag in the JSON array bound to `?3`
/// ("no wanted tag is missing"). An empty array matches every memory.
const HAS_ALL_TAGS: &str = "NOT EXISTS (
    SELECT 1 FROM json_each(?3) want
    WHERE NOT EXISTS (SELECT 1 FROM json_each(m.tags) have WHERE have.value = want.value))";

fn row_to_memory(r: &Row) -> rusqlite::Result<Memory> {
    Ok(Memory {
        id: r.get(0)?,
        content: r.get(1)?,
        tags: parse_tags(&r.get::<_, String>(2)?),
        project: r.get(3)?,
        created_at: r.get(4)?,
        updated_at: r.get(5)?,
    })
}

fn parse_tags(json: &str) -> Vec<String> {
    serde_json::from_str(json).unwrap_or_default()
}

/// Identical content (ignoring surrounding and repeated whitespace) hashes the same.
pub fn content_hash(content: &str) -> String {
    let normalized = content.split_whitespace().collect::<Vec<_>>().join(" ");
    Sha256::digest(normalized.as_bytes()).iter().map(|b| format!("{b:02x}")).collect()
}

/// Builds an FTS5 query that ORs every term, each quoted so user input cannot inject
/// FTS5 syntax. The trigram tokenizer cannot match terms shorter than 3 characters.
pub fn fts_query(query: &str) -> Option<String> {
    let terms: Vec<String> = query
        .split_whitespace()
        .filter(|t| t.chars().count() >= 3)
        .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tags(t: &[&str]) -> Vec<String> {
        t.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn insert_get_roundtrip_and_dedup() {
        let s = Store::open_in_memory().unwrap();
        let (id, dup) = s.insert("  The deploy uses   GitHub Actions ", &tags(&["deploy"]), Some("billing"), &[0.5, 0.25]).unwrap();
        assert!(!dup);
        let (again, dup) = s.insert("The deploy uses GitHub Actions", &[], None, &[0.0, 0.0]).unwrap();
        assert!(dup);
        assert_eq!(again, id);
        let m = s.get(id).unwrap().unwrap();
        assert_eq!(m.content, "The deploy uses   GitHub Actions");
        assert_eq!(m.tags, tags(&["deploy"]));
        assert_eq!(m.project.as_deref(), Some("billing"));
        assert_eq!(s.all_vectors().unwrap()[0].embedding, vec![0.5, 0.25]);
    }

    #[test]
    fn keyword_search_matches_substrings_and_japanese() {
        let s = Store::open_in_memory().unwrap();
        let a = s.insert("Database failover runbook for billing", &[], Some("billing"), &[0.0]).unwrap().0;
        let b = s.insert("東京リージョンのフェイルオーバー手順", &[], None, &[0.0]).unwrap().0;
        s.insert("Unrelated note about lunch", &[], None, &[0.0]).unwrap();
        assert_eq!(s.keyword("failover", None, &[], 10).unwrap(), vec![a]);
        assert_eq!(s.keyword("フェイルオーバー", None, &[], 10).unwrap(), vec![b]);
        assert_eq!(s.keyword("failover", Some("other"), &[], 10).unwrap(), Vec::<i64>::new());
        // Two-character terms are too short for the trigram index and use the LIKE fallback.
        assert_eq!(s.keyword("東京", None, &[], 10).unwrap(), vec![b]);
        assert_eq!(s.keyword("failover 東京", None, &[], 10).unwrap(), vec![a, b]);
        assert!(s.keyword("_%", None, &[], 10).unwrap().is_empty());
        // Short ASCII words are stopwords: "I" must not LIKE-match every row containing an "i".
        assert!(s.keyword("where can I", None, &[], 10).unwrap().is_empty());
    }

    #[test]
    fn keyword_tag_filter_applies_before_the_limit() {
        let s = Store::open_in_memory().unwrap();
        // 60 matches, far more than the limit; only three carry the tag, and they are
        // inserted last so a post-filter on the first 50 would never see them.
        for i in 0..57 {
            s.insert(&format!("deploy note {i}"), &[], None, &[0.0]).unwrap();
        }
        let tagged: Vec<i64> =
            (0..3).map(|i| s.insert(&format!("deploy runbook {i}"), &tags(&["runbook"]), None, &[0.0]).unwrap().0).collect();
        let mut found = s.keyword("deploy", None, &tags(&["runbook"]), 50).unwrap();
        found.sort();
        assert_eq!(found, tagged);
        // Every tag must be present, and the filter also applies to the LIKE fallback.
        assert!(s.keyword("deploy", None, &tags(&["runbook", "missing"]), 50).unwrap().is_empty());
        assert_eq!(s.keyword("運用", None, &tags(&["runbook"]), 50).unwrap(), Vec::<i64>::new());
        s.insert("運用手順", &tags(&["runbook"]), None, &[0.0]).unwrap();
        assert_eq!(s.keyword("運用", None, &tags(&["runbook"]), 50).unwrap().len(), 1);
        assert!(s.keyword("運用", None, &tags(&["other"]), 50).unwrap().is_empty());
    }

    #[test]
    fn open_existing_refuses_to_create() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("missing/memories.db");
        let err = Store::open_existing(&path).unwrap_err();
        assert!(err.to_string().starts_with("no database at"), "{err}");
        assert!(!path.parent().unwrap().exists());
        Store::open(&path).unwrap().insert("kept", &[], None, &[0.0]).unwrap();
        assert_eq!(Store::open_existing(&path).unwrap().count().unwrap(), 1);
    }

    #[test]
    fn newer_schema_is_rejected_before_any_ddl() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("future.db");
        {
            let conn = Connection::open(&path).unwrap();
            conn.execute_batch(META_SCHEMA).unwrap();
            conn.execute("INSERT INTO meta VALUES ('schema_version', '2')", []).unwrap();
        }
        let err = Store::open(&path).unwrap_err();
        assert!(err.to_string().contains("schema version 2 is not supported"), "{err}");
        let conn = Connection::open(&path).unwrap();
        let tables: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'memories%'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(tables, 0, "v1 DDL must not run against a v2 database");
    }

    #[test]
    fn update_to_existing_content_is_a_clear_error() {
        let s = Store::open_in_memory().unwrap();
        let a = s.insert("first", &[], None, &[0.0]).unwrap().0;
        let b = s.insert("second", &[], None, &[0.0]).unwrap().0;
        let err = s.update(b, Some("first"), None, Some(&[0.0])).unwrap_err();
        assert_eq!(err.to_string(), format!("memory {a} already has this content"));
        // Re-saving a memory's own content is fine.
        assert!(s.update(a, Some("first"), None, Some(&[0.0])).is_ok());
    }

    #[test]
    fn keyword_query_is_escaped() {
        let s = Store::open_in_memory().unwrap();
        s.insert("quote \"inside\" text", &[], None, &[0.0]).unwrap();
        // FTS5 operators and stray quotes must not cause syntax errors.
        assert!(s.keyword("\"inside\" AND NOT( NEAR", None, &[], 10).is_ok());
        assert_eq!(fts_query("a b"), None);
    }

    #[test]
    fn update_keeps_fts_in_sync_and_delete_removes() {
        let s = Store::open_in_memory().unwrap();
        let id = s.insert("old wording", &[], None, &[0.0]).unwrap().0;
        s.update(id, Some("new phrasing"), Some(&tags(&["x"])), Some(&[1.0])).unwrap();
        assert!(s.keyword("wording", None, &[], 10).unwrap().is_empty());
        assert_eq!(s.keyword("phrasing", None, &[], 10).unwrap(), vec![id]);
        assert_eq!(s.list(None, Some("x"), 10).unwrap().len(), 1);
        assert!(s.delete(id).unwrap());
        assert!(s.get(id).unwrap().is_none());
        assert!(s.keyword("phrasing", None, &[], 10).unwrap().is_empty());
    }

    #[test]
    fn list_filters_by_project_and_tag() {
        let s = Store::open_in_memory().unwrap();
        s.insert("one", &tags(&["a"]), Some("p1"), &[0.0]).unwrap();
        s.insert("two", &tags(&["b"]), Some("p1"), &[0.0]).unwrap();
        s.insert("three", &tags(&["a"]), Some("p2"), &[0.0]).unwrap();
        assert_eq!(s.list(Some("p1"), None, 10).unwrap().len(), 2);
        assert_eq!(s.list(None, Some("a"), 10).unwrap().len(), 2);
        assert_eq!(s.list(Some("p1"), Some("a"), 10).unwrap().len(), 1);
    }
}
