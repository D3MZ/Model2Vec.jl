// Native Rust reference benchmark for model2vec inference (no FFI — the fairest possible
// comparison; this is the same tokenize -> pool -> normalize algorithm as
// MonsieurPapin/deps/model2vec_rs_worker/src/model.rs, trimmed of jlrs/hf-hub so it runs as a
// plain binary). Reads a local model2vec snapshot dir + a corpus file (one record per line),
// encodes every record, and reports the minimum wall time of `encode_with_args` over many runs.
use half::f16;
use safetensors::{SafeTensors, tensor::Dtype};
use serde_json::Value;
use std::{fs, path::Path, time::Instant};
use tokenizers::Tokenizer;

struct StaticModel {
    tokenizer: Tokenizer,
    embeddings: Vec<Vec<f32>>, // row per token id
    normalize: bool,
    median: usize,
    unknown: Option<usize>,
}

impl StaticModel {
    fn from_local<P: AsRef<Path>>(dir: P) -> Self {
        let dir = dir.as_ref();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("load tokenizer");
        let mut lengths = tokenizer
            .get_vocab(false)
            .keys()
            .map(|token| token.len())
            .collect::<Vec<_>>();
        lengths.sort_unstable();
        let median = lengths.get(lengths.len() / 2).copied().unwrap_or(1);

        let config = serde_json::from_reader::<_, Value>(
            fs::File::open(dir.join("config.json")).expect("open config.json"),
        )
        .expect("parse config.json");
        let normalize = config
            .get("normalize")
            .and_then(Value::as_bool)
            .unwrap_or(true);

        let specification = serde_json::from_str::<Value>(
            &tokenizer.to_string(false).expect("tokenizer -> JSON"),
        )
        .expect("parse tokenizer JSON");
        let unknown = specification
            .get("model")
            .and_then(|model| model.get("unk_token"))
            .and_then(Value::as_str)
            .and_then(|token| tokenizer.token_to_id(token))
            .map(|id| id as usize);

        let bytes = fs::read(dir.join("model.safetensors")).expect("read model.safetensors");
        let tensors = SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let tensor = tensors
            .tensor("embeddings")
            .or_else(|_| tensors.tensor("0"))
            .expect("embeddings tensor");
        let [rows, cols]: [usize; 2] = tensor.shape().try_into().expect("2-D embeddings");
        let flat = floats(tensor);
        let embeddings = flat.chunks_exact(cols).map(|row| row.to_vec()).collect::<Vec<_>>();
        assert_eq!(embeddings.len(), rows);

        Self { tokenizer, embeddings, normalize, median, unknown }
    }

    fn encode_batch(&self, texts: &[String]) -> usize {
        let mut total = 0usize;
        for batch in texts.chunks(64) {
            let inputs = batch
                .iter()
                .map(|text| truncate(text, 512, self.median))
                .collect::<Vec<_>>();
            let encodings = self.tokenizer.encode_batch(inputs, false).expect("tokenize");
            for encoding in encodings {
                let mut ids = encoding.get_ids().to_vec();
                if let Some(unknown) = self.unknown {
                    ids.retain(|&id| id as usize != unknown);
                }
                ids.truncate(512);
                let pooled = self.pool(&ids);
                total += pooled.len(); // touch the result so nothing is elided
            }
        }
        total
    }

    fn pool(&self, ids: &[u32]) -> Vec<f32> {
        let width = self.embeddings[0].len();
        let mut sum = vec![0.0f32; width];
        for &id in ids {
            let row = &self.embeddings[id as usize];
            for (index, value) in row.iter().enumerate() {
                sum[index] += value;
            }
        }
        let denominator = (ids.len().max(1)) as f32;
        sum.iter_mut().for_each(|value| *value /= denominator);
        if self.normalize {
            let norm = sum.iter().map(|value| value * value).sum::<f32>().sqrt().max(1e-12);
            sum.iter_mut().for_each(|value| *value /= norm);
        }
        sum
    }
}

fn truncate(text: &str, max: usize, median: usize) -> &str {
    let chars = max.saturating_mul(median);
    match text.char_indices().nth(chars) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

fn floats(tensor: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    let bytes = tensor.data();
    match tensor.dtype() {
        Dtype::F32 => bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        Dtype::F16 => bytes.chunks_exact(2).map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
        Dtype::I8 => bytes.iter().map(|&b| f32::from(b as i8)).collect(),
        dtype => panic!("unsupported embedding dtype {dtype:?}"),
    }
}

fn main() {
    let modeldir = std::env::args().nth(1).expect("usage: m2v_ref <model_dir> <corpus.txt> [runs]");
    let corpuspath = std::env::args().nth(2).expect("usage: m2v_ref <model_dir> <corpus.txt> [runs]");
    let runs: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(5);

    let model = StaticModel::from_local(&modeldir);
    let texts: Vec<String> = fs::read_to_string(&corpuspath)
        .expect("read corpus")
        .lines()
        .map(String::from)
        .collect();

    model.encode_batch(&texts); // warmup

    let mut best = u128::MAX;
    for _ in 0..runs {
        let t = Instant::now();
        let touched = model.encode_batch(&texts);
        let ns = t.elapsed().as_nanos();
        std::hint::black_box(touched);
        if ns < best {
            best = ns;
        }
    }

    println!("records={} min_ns={}", texts.len(), best);
}
