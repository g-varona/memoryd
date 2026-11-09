//! Checks against the real model files. Skipped by default because they need
//! `scripts/fetch-model.sh`; run with `cargo test --release -- --ignored`.

use memoryd::config::Paths;
use memoryd::embed::{Embedder, MiniLm};

fn model() -> MiniLm {
    MiniLm::load(&Paths::from_env().unwrap().model_dir()).expect("run scripts/fetch-model.sh first")
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[test]
#[ignore]
fn matches_sentence_transformers_reference() {
    // First values of SentenceTransformer("all-MiniLM-L6-v2").encode("hello world").
    let v = model().embed_one("hello world").unwrap();
    assert_eq!(v.len(), 384);
    let expected = [-0.034_477_27, 0.031_023_18, 0.006_734_97, 0.026_108_94];
    for (got, want) in v.iter().zip(expected) {
        assert!((got - want).abs() < 1e-4, "got {got}, want {want}");
    }
}

#[test]
#[ignore]
fn paraphrases_rank_above_unrelated_text() {
    let m = model();
    let v = m
        .embed(&[
            "The service fails over to the backup region when the primary is down",
            "If the main datacenter goes offline, traffic moves to the standby region",
            "We are having pizza for lunch on Friday",
        ])
        .unwrap();
    let (related, unrelated) = (cos(&v[0], &v[1]), cos(&v[0], &v[2]));
    assert!(related > unrelated + 0.25, "related {related}, unrelated {unrelated}");
}

#[test]
#[ignore]
fn batching_and_padding_do_not_change_vectors() {
    let m = model();
    let alone = m.embed_one("short").unwrap();
    let batched = m.embed(&["short", "a much longer sentence that forces padding of the short one"]).unwrap();
    assert!(cos(&alone, &batched[0]) > 0.9999);
}
