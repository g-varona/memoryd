//! memoryd: a small local memory server for Claude Code.
//!
//! One process owns the SQLite database and the in-RAM vector index, and every
//! Claude Code session talks to it over MCP on localhost HTTP.

pub mod auth;
pub mod config;
pub mod embed;
pub mod engine;
pub mod index;
pub mod mcp;
pub mod server;
pub mod store;
