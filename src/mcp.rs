use std::sync::Arc;

use rmcp::{
    ErrorData, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, ContentBlock, Implementation, ServerCapabilities, ServerConfig},
    schemars::{self, JsonSchema},
    tool, tool_handler, tool_router,
};
use serde::{Deserialize, Serialize};

use crate::engine::{Engine, SearchMode};
use crate::index::Filter;

const INSTRUCTIONS: &str = "\
memoryd is the user's long-term memory, shared by all their Claude Code sessions.
- Before answering questions about past decisions, people, projects, or preferences, call memory_search.
- When you learn something durable (a decision and its reason, a non-obvious fact, a preference, \
a pointer to where something lives), call memory_add with one self-contained fact per memory.
- Always write memories in English, translating if the source is in another language: \
search works on meaning, and the embedding model only understands English. Keep the meaning \
and context, not the verbatim wording. Search in English too.
- Set `project` to the repository or topic name when the memory only matters there.
- Do not store secrets, credentials, or customer data.
- If a memory turns out to be wrong, fix it with memory_update or remove it with memory_delete.";

#[derive(Clone)]
pub struct MemoryServer {
    engine: Arc<Engine>,
    tool_router: ToolRouter<Self>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddArgs {
    /// The memory: one self-contained fact, written so it makes sense without the conversation.
    pub content: String,
    /// Optional labels, e.g. ["decision", "billing"].
    #[serde(default)]
    pub tags: Vec<String>,
    /// Optional repository or topic the memory belongs to.
    pub project: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SearchArgs {
    /// What you are looking for, in natural language or keywords.
    pub query: String,
    /// Maximum results (default 5, max 50).
    pub limit: Option<usize>,
    /// Only return memories from this project.
    pub project: Option<String>,
    /// Only return memories that have all of these tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// hybrid (default), semantic, or keyword.
    #[serde(default)]
    pub mode: SearchMode,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct IdArgs {
    pub id: i64,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListArgs {
    pub project: Option<String>,
    pub tag: Option<String>,
    /// Maximum results (default 20, max 200).
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct UpdateArgs {
    pub id: i64,
    /// New content. The memory is re-embedded.
    pub content: Option<String>,
    /// New tags, replacing the old ones.
    pub tags: Option<Vec<String>>,
}

impl MemoryServer {
    pub fn new(engine: Arc<Engine>) -> Self {
        Self { engine, tool_router: Self::tool_router() }
    }

    /// Runs blocking engine work (embedding, SQLite) off the async runtime.
    async fn run<T, F>(&self, f: F) -> Result<CallToolResult, ErrorData>
    where
        T: Serialize + Send + 'static,
        F: FnOnce(&Engine) -> anyhow::Result<T> + Send + 'static,
    {
        let engine = self.engine.clone();
        let result = tokio::task::spawn_blocking(move || f(&engine))
            .await
            .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
        match result {
            Ok(value) => {
                let json = serde_json::to_string(&value).map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
                Ok(CallToolResult::success(vec![ContentBlock::text(json)]))
            }
            Err(e) => Ok(CallToolResult::error(vec![ContentBlock::text(format!("{e:#}"))])),
        }
    }
}

fn not_found(id: i64) -> anyhow::Error {
    anyhow::anyhow!("no memory with id {id}")
}

#[tool_router]
impl MemoryServer {
    #[tool(description = "Save a memory. Returns its id; saving identical content again returns the existing id.")]
    async fn memory_add(&self, Parameters(a): Parameters<AddArgs>) -> Result<CallToolResult, ErrorData> {
        self.run(move |e| e.add(&a.content, a.tags, a.project)).await
    }

    #[tool(description = "Search memories by meaning and keywords. Returns the best matches first; memories unrelated to \
                          the query are omitted, so an empty result means nothing relevant is stored. Each hit's \
                          `similarity` is the cosine between query and memory (~0.3+ is related).")]
    async fn memory_search(&self, Parameters(a): Parameters<SearchArgs>) -> Result<CallToolResult, ErrorData> {
        self.run(move |e| {
            let filter = Filter { project: a.project, tags: a.tags };
            e.search(&a.query, a.limit.unwrap_or(5), a.mode, &filter)
        })
        .await
    }

    #[tool(description = "Get one memory by id.")]
    async fn memory_get(&self, Parameters(a): Parameters<IdArgs>) -> Result<CallToolResult, ErrorData> {
        self.run(move |e| e.get(a.id)?.ok_or_else(|| not_found(a.id))).await
    }

    #[tool(description = "List the most recent memories, optionally filtered by project or tag.")]
    async fn memory_list(&self, Parameters(a): Parameters<ListArgs>) -> Result<CallToolResult, ErrorData> {
        self.run(move |e| e.list(a.project.as_deref(), a.tag.as_deref(), a.limit.unwrap_or(20))).await
    }

    #[tool(description = "Change a memory's content and/or tags.")]
    async fn memory_update(&self, Parameters(a): Parameters<UpdateArgs>) -> Result<CallToolResult, ErrorData> {
        self.run(move |e| e.update(a.id, a.content.as_deref(), a.tags)?.ok_or_else(|| not_found(a.id))).await
    }

    #[tool(description = "Delete a memory by id.")]
    async fn memory_delete(&self, Parameters(a): Parameters<IdArgs>) -> Result<CallToolResult, ErrorData> {
        self.run(move |e| if e.delete(a.id)? { Ok("deleted") } else { Err(not_found(a.id)) }).await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for MemoryServer {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("memoryd", env!("CARGO_PKG_VERSION")))
            .with_instructions(INSTRUCTIONS)
    }
}
