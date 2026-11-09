use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use memoryd::config::{self, DEFAULT_PORT, Paths};
use memoryd::embed::{Embedder, MiniLm};
use memoryd::engine::{Engine, SearchMode};
use memoryd::index::Filter;
use memoryd::store::Store;

#[derive(Parser)]
#[command(version, about = "Local shared memory server for Claude Code")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the MCP server on 127.0.0.1.
    Serve {
        #[arg(long, default_value_t = DEFAULT_PORT)]
        port: u16,
    },
    /// Print the bearer token, creating it if needed.
    Token,
    /// Search from the terminal (opens the database directly; safe while the server runs).
    Search {
        query: String,
        #[arg(long, default_value_t = 5)]
        limit: usize,
        #[arg(long, value_enum, default_value = "hybrid")]
        mode: ModeArg,
    },
    /// Print the embedding of a text (for checking the model).
    Embed { text: String },
    /// Write a consistent copy of the database.
    Backup { dest: PathBuf },
    /// Measure add/search latency and memory use on a throwaway database.
    Bench {
        #[arg(long, default_value_t = 10_000)]
        n: usize,
    },
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum ModeArg {
    Hybrid,
    Semantic,
    Keyword,
}

impl From<ModeArg> for SearchMode {
    fn from(m: ModeArg) -> Self {
        match m {
            ModeArg::Hybrid => SearchMode::Hybrid,
            ModeArg::Semantic => SearchMode::Semantic,
            ModeArg::Keyword => SearchMode::Keyword,
        }
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .init();
    let cli = Cli::parse();
    let paths = Paths::from_env()?;
    match cli.command {
        Command::Serve { port } => serve(&paths, port),
        Command::Token => {
            println!("{}", config::load_or_create_token(&paths.token())?);
            Ok(())
        }
        Command::Search { query, limit, mode } => {
            // Read-only: never create a database at a mistyped path.
            let engine = open_engine(&paths, Store::open_existing(&paths.db())?)?;
            let hits = engine.search(&query, limit, mode.into(), &Filter::default())?;
            println!("{}", serde_json::to_string_pretty(&hits)?);
            Ok(())
        }
        Command::Embed { text } => {
            let v = MiniLm::load(&paths.model_dir())?.embed_one(&text)?;
            println!("{}", serde_json::to_string(&v)?);
            Ok(())
        }
        Command::Backup { dest } => {
            Store::open_existing(&paths.db())?.backup(&dest)?;
            println!("wrote {}", dest.display());
            Ok(())
        }
        Command::Bench { n } => bench(&paths, n),
    }
}

fn open_engine(paths: &Paths, store: Store) -> Result<Engine> {
    let embedder: Arc<dyn Embedder> = Arc::new(MiniLm::load(&paths.model_dir())?);
    Engine::new(store, embedder)
}

fn open_store(db: &std::path::Path) -> Result<Store> {
    Store::open(db).with_context(|| format!("opening {}", db.display()))
}

#[tokio::main]
async fn serve(paths: &Paths, port: u16) -> Result<()> {
    let token = config::load_or_create_token(&paths.token())?;
    let started = Instant::now();
    let engine = Arc::new(open_engine(paths, open_store(&paths.db())?)?);
    tracing::info!(memories = engine.len(), load_ms = started.elapsed().as_millis() as u64, "loaded");

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port)).await?;
    tracing::info!("listening on http://{}/mcp", listener.local_addr()?);
    axum::serve(listener, memoryd::server::router(engine, token)).with_graceful_shutdown(shutdown_signal()).await?;
    tracing::info!("stopped");
    Ok(())
}

/// Resolves on SIGINT (ctrl-c in a terminal) or SIGTERM (what launchd sends on
/// `bootout` / `kickstart -k`), so in-flight tool calls finish before the process exits.
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("cannot listen for SIGTERM: {e}");
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => tracing::info!("received SIGINT, shutting down"),
        _ = term.recv() => tracing::info!("received SIGTERM, shutting down"),
    }
}

fn bench(paths: &Paths, n: usize) -> Result<()> {
    let dir = std::env::temp_dir().join(format!("memoryd-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let engine = open_engine(paths, open_store(&dir.join("bench.db"))?)?;

    const SUBJECTS: [&str; 8] = ["The deploy pipeline", "Our on-call rotation", "The database migration", "Email fan-out",
        "The image service", "Sprint planning", "The regional failover", "Release tagging"];
    const VERBS: [&str; 6] = ["depends on", "was moved to", "is blocked by", "should use", "replaced", "is documented in"];
    const OBJECTS: [&str; 8] = ["infrastructure modules", "a new chat channel", "the legacy schema", "paging schedules",
        "a wiki runbook", "the platform team", "the new database", "the CD system"];

    let mut add_ms = Vec::with_capacity(n);
    for i in 0..n {
        let text = format!("{} {} {} (note {i})", SUBJECTS[i % 8], VERBS[(i / 8) % 6], OBJECTS[(i / 48) % 8]);
        let t = Instant::now();
        engine.add(&text, vec![], None)?;
        add_ms.push(t.elapsed().as_secs_f64() * 1e3);
        if (i + 1) % 1000 == 0 {
            eprintln!("added {}", i + 1);
        }
    }
    let queries = ["where is the failover runbook", "what blocks the database migration", "who owns paging", "CD system"];
    let mut search_ms = Vec::new();
    for _ in 0..25 {
        for q in queries {
            let t = Instant::now();
            engine.search(q, 5, SearchMode::Hybrid, &Filter::default())?;
            search_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
    }
    let rss_kb = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0);
    println!("memories: {n}");
    println!("add    p50 {:.1}ms  p95 {:.1}ms", pct(&mut add_ms, 0.50), pct(&mut add_ms, 0.95));
    println!("search p50 {:.1}ms  p95 {:.1}ms", pct(&mut search_ms, 0.50), pct(&mut search_ms, 0.95));
    println!("rss    {} MB", rss_kb / 1024);
    std::fs::remove_dir_all(&dir)?;
    Ok(())
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}
