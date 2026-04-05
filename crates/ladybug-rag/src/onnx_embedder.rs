//! ONNX Runtime-based embedder using multilingual sentence-transformer models.
//!
//! Requires the `onnx` feature flag:
//! ```toml
//! ladybug-rag = { path = "...", features = ["onnx"] }
//! ```
//!
//! Supports any sentence-transformers ONNX model. Recommended:
//! - `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, 50+ languages)
//! - `intfloat/multilingual-e5-small` (384-dim, 100+ languages)

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use ort::{GraphOptimizationLevel, Session};
use tokenizers::Tokenizer;

use crate::embeddings::Embedder;

/// An embedder backed by an ONNX model and HuggingFace tokenizer.
pub struct OnnxEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    dim: usize,
    max_length: usize,
}

/// Configuration for loading an ONNX embedder.
pub struct OnnxEmbedderConfig {
    /// Path to the ONNX model file.
    pub model_path: PathBuf,
    /// Path to the tokenizer.json file.
    pub tokenizer_path: PathBuf,
    /// Embedding dimensionality (e.g. 384 for MiniLM).
    pub dimension: usize,
    /// Maximum token sequence length.
    pub max_length: usize,
    /// Number of intra-op threads for ONNX Runtime.
    pub num_threads: usize,
}

impl Default for OnnxEmbedderConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("model.onnx"),
            tokenizer_path: PathBuf::from("tokenizer.json"),
            dimension: 384,
            max_length: 512,
            num_threads: 4,
        }
    }
}

impl OnnxEmbedder {
    /// Create an embedder from explicit file paths.
    pub fn from_files(config: OnnxEmbedderConfig) -> Result<Self, OnnxEmbedderError> {
        let session = Session::builder()
            .map_err(|e| OnnxEmbedderError::SessionInit(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxEmbedderError::SessionInit(e.to_string()))?
            .with_intra_threads(config.num_threads)
            .map_err(|e| OnnxEmbedderError::SessionInit(e.to_string()))?
            .commit_from_file(&config.model_path)
            .map_err(|e| OnnxEmbedderError::ModelLoad(e.to_string()))?;

        let tokenizer = Tokenizer::from_file(&config.tokenizer_path)
            .map_err(|e| OnnxEmbedderError::TokenizerLoad(e.to_string()))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            dim: config.dimension,
            max_length: config.max_length,
        })
    }

    /// Download a model from HuggingFace Hub and create an embedder.
    ///
    /// Example:
    /// ```no_run
    /// let embedder = OnnxEmbedder::from_hf_hub(
    ///     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ///     384,
    /// ).unwrap();
    /// ```
    pub fn from_hf_hub(repo_id: &str, dimension: usize) -> Result<Self, OnnxEmbedderError> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| OnnxEmbedderError::HubDownload(e.to_string()))?;
        let repo = api.model(repo_id.to_string());

        let model_path = repo
            .get("model.onnx")
            .or_else(|_| repo.get("onnx/model.onnx"))
            .map_err(|e| OnnxEmbedderError::HubDownload(format!("model.onnx: {}", e)))?;

        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| OnnxEmbedderError::HubDownload(format!("tokenizer.json: {}", e)))?;

        Self::from_files(OnnxEmbedderConfig {
            model_path,
            tokenizer_path,
            dimension,
            max_length: 512,
            num_threads: 4,
        })
    }

    /// Load from a local directory containing `model.onnx` and `tokenizer.json`.
    pub fn from_dir(dir: &Path, dimension: usize) -> Result<Self, OnnxEmbedderError> {
        Self::from_files(OnnxEmbedderConfig {
            model_path: dir.join("model.onnx"),
            tokenizer_path: dir.join("tokenizer.json"),
            dimension,
            ..OnnxEmbedderConfig::default()
        })
    }

    fn embed_internal(&self, text: &str) -> Result<Vec<f32>, OnnxEmbedderError> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| OnnxEmbedderError::Tokenize(e.to_string()))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let mut attention_mask: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let mut token_type_ids: Vec<i64> =
            encoding.get_type_ids().iter().map(|&t| t as i64).collect();

        // Truncate to max_length
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
            token_type_ids.truncate(self.max_length);
        }

        let seq_len = input_ids.len();

        // Create tensors
        let input_ids_array =
            Array2::from_shape_vec((1, seq_len), input_ids)
                .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;
        let attention_mask_array =
            Array2::from_shape_vec((1, seq_len), attention_mask.clone())
                .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;
        let token_type_ids_array =
            Array2::from_shape_vec((1, seq_len), token_type_ids)
                .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;

        // Run inference
        let session = self
            .session
            .lock()
            .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;

        // Try with token_type_ids first, fall back without
        let outputs = session
            .run(ort::inputs![
                "input_ids" => input_ids_array.clone(),
                "attention_mask" => attention_mask_array.clone(),
                "token_type_ids" => token_type_ids_array
            ].map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?)
            .or_else(|_| {
                // Some models don't use token_type_ids — retry without
                let inputs = ort::inputs![
                    "input_ids" => input_ids_array,
                    "attention_mask" => attention_mask_array
                ].unwrap();
                session.run(inputs)
            })
            .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;

        // Extract last_hidden_state or sentence_embedding
        // Models may output different names — try common ones
        let output_key = if outputs.get("sentence_embedding").is_some() {
            "sentence_embedding"
        } else if outputs.get("last_hidden_state").is_some() {
            "last_hidden_state"
        } else {
            // Fall back to first output
            outputs
                .keys()
                .next()
                .ok_or_else(|| OnnxEmbedderError::Inference("no model outputs".into()))?
        };

        let output_tensor = outputs[output_key]
            .try_extract_tensor::<f32>()
            .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;

        let shape = output_tensor.shape();
        let embedding = if shape.len() == 3 {
            // [batch, seq_len, hidden_dim] — mean pool over seq dim
            let view = output_tensor.view();
            let reshaped = view
                .to_shape((shape[0], shape[1], shape[2]))
                .map_err(|e| OnnxEmbedderError::Inference(e.to_string()))?;

            // Masked mean pooling
            let mask_f32: Vec<f32> = attention_mask.iter().map(|&m| m as f32).collect();
            let mask_sum: f32 = mask_f32.iter().sum::<f32>().max(1.0);

            let mut pooled = vec![0.0f32; shape[2]];
            for seq_idx in 0..shape[1] {
                let mask_val = mask_f32.get(seq_idx).copied().unwrap_or(0.0);
                for dim_idx in 0..shape[2] {
                    pooled[dim_idx] += reshaped[[0, seq_idx, dim_idx]] * mask_val;
                }
            }
            for v in &mut pooled {
                *v /= mask_sum;
            }
            pooled
        } else if shape.len() == 2 {
            // [batch, hidden_dim] — already pooled
            let view = output_tensor.view();
            (0..shape[1]).map(|i| view[[0, i]]).collect()
        } else {
            return Err(OnnxEmbedderError::Inference(format!(
                "unexpected output shape: {:?}",
                shape
            )));
        };

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            embedding.iter().map(|x| x / norm).collect()
        } else {
            embedding
        };

        Ok(normalized)
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Vec<f32> {
        self.embed_internal(text).unwrap_or_else(|e| {
            eprintln!("OnnxEmbedder::embed failed: {}", e);
            vec![0.0; self.dim]
        })
    }

    fn embed_batch(&self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dim
    }
}

// Send+Sync: Session is behind Mutex, Tokenizer is Send+Sync
unsafe impl Send for OnnxEmbedder {}
unsafe impl Sync for OnnxEmbedder {}

#[derive(Debug, thiserror::Error)]
pub enum OnnxEmbedderError {
    #[error("session init: {0}")]
    SessionInit(String),
    #[error("model load: {0}")]
    ModelLoad(String),
    #[error("tokenizer load: {0}")]
    TokenizerLoad(String),
    #[error("hub download: {0}")]
    HubDownload(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("inference: {0}")]
    Inference(String),
}
