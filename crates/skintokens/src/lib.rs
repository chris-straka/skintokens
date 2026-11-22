//! SkinTokens / TokenRig auto-rigging in Rust (candle): the library behind
//! the `skintokens` CLI.

pub use skintokens_geom::{geom, glb, mesh, naming};

pub mod generate;
pub mod model;
pub mod parity;
pub mod pipeline;
pub mod qwen;
pub mod tokenizer;
pub mod vae;
