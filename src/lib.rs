//! rooler — fast out-of-core cooler engine.
//!
//! Ops: [`cload`] (pairs -> cool), [`merge`], [`zoomify`], [`balance`], [`expected`].
//! [`cooler`] holds the v3 schema reader/writer, [`scratch`]/[`scratch_tiled`] the two
//! compressed-CSR SpMV kernels balance runs over, [`view`] the genome/region defaults.
//! The `rooler` binary (src/main.rs) is a thin clap wrapper over these.
pub mod balance;
pub mod cload;
pub mod cooler;
pub mod expected;
pub mod merge;
pub mod parwrite;
pub mod repack;
pub mod scratch;
pub mod scratch_tiled;
pub mod view;
pub mod zoomify;
