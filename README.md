# memoryd

A small, local memory server for Claude Code. One process serves every Claude Code
session over MCP on `127.0.0.1`, so all sessions share the same memories.

- **Tools:** `memory_add`, `memory_search`, `memory_get`, `memory_list`, `memory_update`, `memory_delete`.
- **Search:** hybrid. Meaning-based (all-MiniLM-L6-v2, 384-d) and keyword-based (SQLite FTS5
  trigram, which also matches Japanese) results are merged with reciprocal rank fusion.
- **Storage:** one SQLite file at `~/.local/share/memoryd/memories.db`. Vectors are also kept
  in RAM and scanned on every search, which is fast enough for tens of thousands of memories.
- **Local only:** the binary contains no HTTP client and makes no outbound connections.
  The model is loaded from local files. There is no telemetry and there are no hooks.
- **Access:** requests must carry a bearer token (`~/.config/memoryd/token`, mode 0600).
  Non-loopback `Host` headers and any browser `Origin` are rejected.

## Layout

| File | What it does |
|---|---|
| `src/embed.rs` | Loads the BERT model with candle; mean pooling and L2 normalisation |
| `src/store.rs` | SQLite schema, CRUD, FTS5 and backups |
| `src/index.rs` | In-RAM vectors, top-k search, reciprocal rank fusion |
| `src/engine.rs` | Ties store, index and embedder together; single writer |
| `src/mcp.rs` | MCP tool definitions and server instructions |
| `src/server.rs`, `src/auth.rs` | HTTP routing, bearer-token check |
| `src/main.rs` | CLI: `serve`, `token`, `search`, `embed`, `backup`, `bench` |

## Install

```sh
# 1. Model (one-time download, about 90MB; pins the commit and SHA-256 in scripts/model.lock)
scripts/fetch-model.sh

# 2. Build and install the binary
cargo build --release
mkdir -p ~/.local/bin ~/Library/Logs/memoryd
cp target/release/memoryd ~/.local/bin/

# 3. Run at login
sed "s|HOME_DIR|$HOME|g" packaging/com.g-varona.memoryd.plist > ~/Library/LaunchAgents/com.g-varona.memoryd.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.g-varona.memoryd.plist

# 4. Register with Claude Code (all projects)
claude mcp add --scope user --transport http memoryd http://127.0.0.1:7788/mcp \
  --header "Authorization: Bearer $(~/.local/bin/memoryd token)"
```

Restart running Claude sessions to pick it up.

## Operate

```sh
launchctl kickstart -k gui/$(id -u)/com.g-varona.memoryd   # restart (e.g. after installing a new build)
tail -f ~/Library/Logs/memoryd/memoryd.log
memoryd search "database failover"                          # query from the terminal
memoryd backup ~/memoryd-$(date +%F).db                     # consistent copy, safe while running
memoryd bench --n 10000                                     # latency and RSS on a throwaway DB
```

## Uninstall

```sh
claude mcp remove memoryd -s user
launchctl bootout gui/$(id -u)/com.g-varona.memoryd
rm ~/Library/LaunchAgents/com.g-varona.memoryd.plist ~/.local/bin/memoryd
# Data: ~/.local/share/memoryd (memories and model) and ~/.config/memoryd (token)
```

## Development

`cargo test` runs without the model; tests use a deterministic hash embedder.
