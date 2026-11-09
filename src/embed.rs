use std::path::Path;

use anyhow::{Context, Result, anyhow};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

/// Turns text into unit-length vectors, so a dot product is the cosine similarity.
pub trait Embedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(&[text])?.pop().ok_or_else(|| anyhow!("embedder returned nothing"))
    }
}

/// sentence-transformers/all-MiniLM-L6-v2 running on the CPU through candle.
pub struct MiniLm {
    model: BertModel,
    tokenizer: Tokenizer,
    dim: usize,
}

/// all-MiniLM-L6-v2 was trained with sequences of at most 256 tokens.
const MAX_TOKENS: usize = 256;

impl MiniLm {
    /// Loads `config.json`, `tokenizer.json` and `model.safetensors` from `dir`.
    /// Never touches the network.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| {
            std::fs::read_to_string(dir.join(name))
                .with_context(|| format!("missing {}; run scripts/fetch-model.sh", dir.join(name).display()))
        };
        let config: Config = serde_json::from_str(&read("config.json")?)?;
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!(e))?;
        tokenizer
            .with_truncation(Some(TruncationParams { max_length: MAX_TOKENS, ..Default::default() }))
            .map_err(|e| anyhow!(e))?;
        tokenizer.with_padding(Some(PaddingParams::default()));

        let weights = dir.join("model.safetensors");
        // SAFETY: the file is memory-mapped read-only and memoryd never modifies it.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DTYPE, &Device::Cpu)? };
        let model = BertModel::load(vb, &config)?;
        Ok(Self { model, tokenizer, dim: config.hidden_size })
    }
}

impl Embedder for MiniLm {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), true).map_err(|e| anyhow!(e))?;
        let device = &self.model.device;
        let stack = |field: fn(&tokenizers::Encoding) -> &[u32]| -> Result<Tensor> {
            let rows = encodings
                .iter()
                .map(|e| Tensor::new(field(e), device))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Ok(Tensor::stack(&rows, 0)?)
        };
        let ids = stack(|e| e.get_ids())?;
        let type_ids = stack(|e| e.get_type_ids())?;
        let mask = stack(|e| e.get_attention_mask())?;

        let hidden = self.model.forward(&ids, &type_ids, Some(&mask))?; // (batch, tokens, dim)

        // Mean pooling over real tokens only, then L2 normalisation, as sentence-transformers does.
        let mask = mask.to_dtype(DTYPE)?.unsqueeze(2)?; // (batch, tokens, 1)
        let summed = hidden.broadcast_mul(&mask)?.sum(1)?;
        let counts = mask.sum(1)?.clamp(1e-9, f64::MAX)?;
        let mean = summed.broadcast_div(&counts)?;
        let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?.clamp(1e-12, f64::MAX)?;
        Ok(mean.broadcast_div(&norm)?.to_vec2::<f32>()?)
    }
}

/// A deterministic bag-of-words embedder for tests, so tests do not need the model files.
pub struct HashEmbedder {
    pub dim: usize,
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        use sha2::{Digest, Sha256};
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0f32; self.dim];
                for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
                    let h = Sha256::digest(word.as_bytes());
                    let slot = u16::from_le_bytes([h[0], h[1]]) as usize % self.dim;
                    v[slot] += 1.0;
                }
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect())
    }
}
