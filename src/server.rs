use std::sync::Arc;

use axum::{Router, middleware, routing::get};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::never::NeverSessionManager,
};

use crate::{auth, engine::Engine, mcp::MemoryServer};

/// `/mcp` (MCP over streamable HTTP, bearer token required) and `/health` (no auth, no data).
///
/// The server is stateless: no MCP sessions are kept, so restarting memoryd never strands a
/// Claude session holding a session id the server has forgotten, and idle sessions cannot expire.
/// memoryd only answers tool calls and never pushes messages, so it needs no session state.
pub fn router(engine: Arc<Engine>, token: String) -> Router {
    let config = StreamableHttpServerConfig::default()
        .with_legacy_session_mode(false)
        .with_json_response(true)
        // Only loopback Host headers (rmcp's default) and no browser Origins at all.
        .with_allowed_hosts(["localhost", "127.0.0.1", "::1"])
        .enforce_origin_validation()
        .with_max_request_body_bytes(1024 * 1024);
    let mcp = StreamableHttpService::new(
        move || Ok(MemoryServer::new(engine.clone())),
        Arc::new(NeverSessionManager::default()),
        config,
    );
    let token = Arc::new(token);
    Router::new()
        .nest_service("/mcp", mcp)
        .layer(middleware::from_fn_with_state(token, auth::require_bearer))
        .route("/health", get(|| async { "ok" }))
}
