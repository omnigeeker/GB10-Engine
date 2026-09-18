//! Memory-mapped safetensors reader.
//!
//! Only the header (a few hundred KB for this checkpoint) is parsed up front;
//! tensor payloads are sliced out of an `mmap`, so opening the 21.9 GB
//! checkpoint costs no I/O. Weight loading into device memory is driven by
//! [`ShardedSafeTensors::tensor_bytes`].

use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Element type as stored in the file.
///
/// Names match the safetensors spec verbatim (`F8_E4M3`), hence the lint allow.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    F8_E4M3,
    F8_E5M2,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    BOOL,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::F8_E4M3 | DType::F8_E5M2 | DType::I8 | DType::U8 | DType::BOOL => 1,
        }
    }
}

/// One tensor's location inside a shard.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset of the payload within the file.
    pub offset: usize,
    /// Payload length in bytes.
    pub nbytes: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: DType,
    shape: Vec<usize>,
    data_offsets: (usize, usize),
}

/// A single memory-mapped shard plus its parsed header.
pub struct Shard {
    pub path: PathBuf,
    mmap: Mmap,
    /// Byte offset where tensor payloads begin (8 + header length).
    data_start: usize,
}

impl Shard {
    pub fn open(path: impl AsRef<Path>) -> anyhow::Result<(Self, Vec<TensorInfo>)> {
        let path = path.as_ref().to_path_buf();
        let file = std::fs::File::open(&path)?;
        // SAFETY: the checkpoint is read-only for the life of the process.
        let mmap = unsafe { Mmap::map(&file)? };

        anyhow::ensure!(mmap.len() >= 8, "{}: file too small", path.display());
        let hlen = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        anyhow::ensure!(
            hlen > 0 && 8 + hlen <= mmap.len(),
            "{}: bad safetensors header length {hlen}",
            path.display()
        );

        let raw: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..8 + hlen])?;
        let data_start = 8 + hlen;

        let mut infos = Vec::new();
        for (name, value) in raw {
            if name == "__metadata__" {
                continue;
            }
            let t: RawTensor = serde_json::from_value(value)?;
            let (begin, end) = t.data_offsets;
            anyhow::ensure!(
                end >= begin && data_start + end <= mmap.len(),
                "{}: tensor {name} offsets out of range",
                path.display()
            );
            infos.push(TensorInfo {
                name,
                dtype: t.dtype,
                shape: t.shape,
                offset: data_start + begin,
                nbytes: end - begin,
            });
        }
        infos.sort_by(|a, b| a.name.cmp(&b.name));
        Ok((
            Shard {
                path,
                mmap,
                data_start,
            },
            infos,
        ))
    }

    pub fn data_start(&self) -> usize {
        self.data_start
    }

    /// Raw payload bytes for a tensor, validated against its declared size.
    pub fn tensor_bytes(&self, info: &TensorInfo) -> anyhow::Result<&[u8]> {
        let expect = info.numel() * info.dtype.size_in_bytes();
        anyhow::ensure!(
            expect == info.nbytes,
            "{}: tensor {} declared {} bytes but shape/dtype imply {}",
            self.path.display(),
            info.name,
            info.nbytes,
            expect
        );
        Ok(&self.mmap[info.offset..info.offset + info.nbytes])
    }

    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }
}

/// All shards of a checkpoint, with a name -> (shard index, info) map.
pub struct ShardedSafeTensors {
    pub shards: Vec<Shard>,
    index: HashMap<String, (usize, TensorInfo)>,
}

impl ShardedSafeTensors {
    /// Open a directory containing `model.safetensors.index.json`, or a single
    /// `.safetensors` file.
    pub fn open(dir: impl AsRef<Path>) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");

        let mut shards = Vec::new();
        let mut index = HashMap::new();

        if index_path.exists() {
            #[derive(Deserialize)]
            struct WeightMap {
                weight_map: HashMap<String, String>,
            }
            let wm: WeightMap = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let mut names: Vec<&String> = wm.weight_map.keys().collect();
            names.sort();
            let mut shard_of: HashMap<String, usize> = HashMap::new();
            for name in names {
                let file = &wm.weight_map[name];
                let si = match shard_of.get(file) {
                    Some(i) => *i,
                    None => {
                        let (shard, infos) = Shard::open(dir.join(file))?;
                        let si = shards.len();
                        for info in infos {
                            index.insert(info.name.clone(), (si, info));
                        }
                        shards.push(shard);
                        shard_of.insert(file.clone(), si);
                        si
                    }
                };
                anyhow::ensure!(
                    index.contains_key(name),
                    "index lists {name} but shard {file} does not contain it"
                );
                let _ = si;
            }
        } else {
            let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
                .collect();
            anyhow::ensure!(!files.is_empty(), "no .safetensors files in {}", dir.display());
            files.sort();
            for f in files {
                let (shard, infos) = Shard::open(&f)?;
                let si = shards.len();
                for info in infos {
                    index.insert(info.name.clone(), (si, info));
                }
                shards.push(shard);
            }
        }

        Ok(Self { shards, index })
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|(_, i)| i)
    }

    pub fn tensor_bytes(&self, name: &str) -> anyhow::Result<&[u8]> {
        let (si, info) = self
            .index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor {name} not found in checkpoint"))?;
        self.shards[*si].tensor_bytes(info)
    }

    /// Names sorted lexicographically — deterministic iteration order.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.index.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// Every tensor whose name starts with `prefix`.
    pub fn names_with_prefix<'a>(&'a self, prefix: &str) -> Vec<&'a str> {
        self.names()
            .into_iter()
            .filter(|n| n.starts_with(prefix))
            .collect()
    }

    /// Total bytes across all tensors.
    pub fn total_bytes(&self) -> usize {
        self.index.values().map(|(_, i)| i.nbytes).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_dir() -> Option<PathBuf> {
        let p = PathBuf::from("../../models/Qwen3.8-27B-NVFP4");
        p.exists().then_some(p)
    }

    #[test]
    fn opens_real_checkpoint_and_matches_traffic_table() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: model not present");
            return;
        };
        let st = ShardedSafeTensors::open(&dir).expect("open checkpoint");
        assert_eq!(st.len(), 2194, "tensor count");
        assert_eq!(st.shards.len(), 3, "shard count");
        // 21.92 GB total, from docs/PHYSICS.md
        let gb = st.total_bytes() as f64 / 1e9;
        assert!((gb - 21.921).abs() < 0.01, "total bytes {gb} GB");
    }

    #[test]
    fn known_tensors_have_expected_dtype_and_shape() {
        let Some(dir) = model_dir() else { return };
        let st = ShardedSafeTensors::open(&dir).unwrap();

        let emb = st.info("model.language_model.embed_tokens.weight").unwrap();
        assert_eq!(emb.dtype, DType::BF16);
        assert_eq!(emb.shape, vec![248320, 5120]);

        // NVFP4 is stored packed two-per-byte, so the U8 tensor is half the
        // logical [17408, 5120] element count.
        let gate = st.info("model.language_model.layers.0.mlp.gate_proj.weight").unwrap();
        assert_eq!(gate.dtype, DType::U8);
        assert_eq!(gate.numel() * 2, 17408 * 5120);

        // ...and carries a group-16 scale tensor alongside it.
        assert!(st.contains("model.language_model.layers.0.mlp.gate_proj.weight_scale"));

        // Attention projections are FP8.
        let q = st.info("model.language_model.layers.3.self_attn.q_proj.weight").unwrap();
        assert_eq!(q.dtype, DType::F8_E4M3);
        assert_eq!(q.shape, vec![24 * 256 * 2, 5120]);
    }

    #[test]
    fn tensor_bytes_slice_has_declared_length() {
        let Some(dir) = model_dir() else { return };
        let st = ShardedSafeTensors::open(&dir).unwrap();
        let name = "model.language_model.layers.0.input_layernorm.weight";
        let info = st.info(name).unwrap().clone();
        let bytes = st.tensor_bytes(name).unwrap();
        assert_eq!(bytes.len(), info.nbytes);
        assert_eq!(info.nbytes, 5120 * 2);
    }

    #[test]
    fn mtp_layer_is_unquantized_bf16() {
        let Some(dir) = model_dir() else { return };
        let st = ShardedSafeTensors::open(&dir).unwrap();
        let names = st.names_with_prefix("mtp");
        assert!(!names.is_empty(), "MTP head must exist");
        for n in names {
            let i = st.info(n).unwrap();
            assert!(
                matches!(i.dtype, DType::BF16 | DType::F32),
                "MTP tensor {n} should be unquantized, got {:?}",
                i.dtype
            );
        }
    }
}
