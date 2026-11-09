use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    body::Body,
    http::{Request, StatusCode, header},
};
use http_body_util::BodyExt;
use memoryd::{embed::HashEmbedder, engine::Engine, server, store::Store};
use serde_json::{Value, json};
use tower::ServiceExt;

const TOKEN: &str = "test-token-0123456789abcdef0123456789abcdef";

fn app() -> Router {
    let engine = Engine::new(Store::open_in_memory().unwrap(), Arc::new(HashEmbedder { dim: 64 })).unwrap();
    server::router(Arc::new(engine), TOKEN.to_owned())
}

fn post(body: Value, session: Option<&str>) -> Request<Body> {
    let mut req = Request::post("/mcp")
        .header(header::HOST, "127.0.0.1:7788")
        .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "application/json, text/event-stream");
    if let Some(s) = session {
        req = req.header("mcp-session-id", s);
    }
    req.body(Body::from(body.to_string())).unwrap()
}

/// Reads the response body until a JSON-RPC message with the given id arrives.
/// Handles both plain JSON and SSE (`data: {...}`) responses.
async fn rpc_result(resp: axum::response::Response, id: i64) -> Value {
    let mut body = resp.into_body();
    let mut buf = String::new();
    loop {
        let frame = tokio::time::timeout(Duration::from_secs(5), body.frame())
            .await
            .expect("timed out waiting for response")
            .expect("body ended without a response")
            .unwrap();
        if let Ok(data) = frame.into_data() {
            buf.push_str(std::str::from_utf8(&data).unwrap());
        }
        for line in buf.lines() {
            let line = line.strip_prefix("data:").unwrap_or(line).trim();
            if let Ok(msg) = serde_json::from_str::<Value>(line)
                && msg["id"] == id
            {
                return msg;
            }
        }
    }
}

async fn initialize(app: &Router) -> Option<String> {
    let init = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
        "protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"1"}}});
    let resp = app.clone().oneshot(post(init, None)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let session = resp.headers().get("mcp-session-id").map(|v| v.to_str().unwrap().to_owned());
    let msg = rpc_result(resp, 1).await;
    assert_eq!(msg["result"]["serverInfo"]["name"], "memoryd");
    assert!(msg["result"]["instructions"].as_str().unwrap().contains("memory_search"));
    let done = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
    let resp = app.clone().oneshot(post(done, session.as_deref())).await.unwrap();
    assert!(resp.status().is_success());
    session
}

async fn call(app: &Router, session: Option<&str>, id: i64, name: &str, args: Value) -> Value {
    let body = json!({"jsonrpc":"2.0","id":id,"method":"tools/call","params":{"name":name,"arguments":args}});
    let resp = app.clone().oneshot(post(body, session)).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let msg = rpc_result(resp, id).await;
    let text = msg["result"]["content"][0]["text"].as_str().unwrap_or_else(|| panic!("{msg}"));
    assert_ne!(msg["result"]["isError"], true, "{text}");
    serde_json::from_str(text).unwrap_or(Value::String(text.to_owned()))
}

#[tokio::test]
async fn rejects_missing_or_wrong_token() {
    let app = app();
    let req = Request::post("/mcp").header(header::HOST, "127.0.0.1").body(Body::empty()).unwrap();
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), StatusCode::UNAUTHORIZED);
    let req = Request::post("/mcp")
        .header(header::HOST, "127.0.0.1")
        .header(header::AUTHORIZATION, "Bearer wrong")
        .body(Body::empty())
        .unwrap();
    assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn rejects_foreign_host_and_any_origin() {
    let app = app();
    let init = json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}});
    let mut req = post(init.clone(), None);
    req.headers_mut().insert(header::HOST, "evil.example.com".parse().unwrap());
    assert!(app.clone().oneshot(req).await.unwrap().status().is_client_error());
    let mut req = post(init, None);
    req.headers_mut().insert(header::ORIGIN, "http://localhost:3000".parse().unwrap());
    assert!(app.oneshot(req).await.unwrap().status().is_client_error());
}

#[tokio::test]
async fn stale_session_id_after_restart_still_works() {
    // A client that connected before a restart keeps sending its old session id.
    let app = app();
    initialize(&app).await;
    let hits = call(&app, Some("session-from-before-restart"), 2, "memory_list", json!({})).await;
    assert_eq!(hits, json!([]));
}

#[tokio::test]
async fn health_needs_no_token() {
    let req = Request::get("/health").header(header::HOST, "127.0.0.1").body(Body::empty()).unwrap();
    assert_eq!(app().oneshot(req).await.unwrap().status(), StatusCode::OK);
}

#[tokio::test]
async fn add_then_search_over_mcp() {
    let app = app();
    let session = initialize(&app).await;
    let s = session.as_deref();

    let added = call(&app, s, 2, "memory_add", json!({"content":"Database failover runbook is in the wiki","tags":["dr"]})).await;
    let id = added["id"].as_i64().unwrap();
    assert_eq!(added["duplicate"], false);

    let hits = call(&app, s, 3, "memory_search", json!({"query":"failover runbook"})).await;
    assert_eq!(hits[0]["id"], id);
    assert_eq!(hits[0]["tags"], json!(["dr"]));

    let updated = call(&app, s, 4, "memory_update", json!({"id":id,"tags":["dr","db"]})).await;
    assert_eq!(updated["tags"], json!(["dr","db"]));
    let listed = call(&app, s, 5, "memory_list", json!({"tag":"db"})).await;
    assert_eq!(listed.as_array().unwrap().len(), 1);
    assert_eq!(call(&app, s, 6, "memory_delete", json!({"id":id})).await, "deleted");
}
