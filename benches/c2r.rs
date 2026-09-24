//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! Each FFTW planning mode (Estimate / Measure / Conserve) lives in its own
//! `[[bench]]` binary so FFTW's global per-process wisdom cache cannot leak
//! between modes; each run starts with a fresh process and empty wisdom.
//! Group names are shared with `realfft.rs` / the other
//! `fftw_*_r2c_c2r.rs` binaries; criterion does NOT auto-aggregate across binaries
//! — use `benches/plot_criterion_overlay.py` for the cross-binary overlay.
//!
//! This series combines MEASURE with `FFTW_CONSERVE_MEMORY` so FFTW selects
//! memory-efficient plan variants during the same search it would run for
//! plain MEASURE. This is a more faithful comparison for PhastFT, which is
//! also designed for low memory overhead — plain PATIENT / MEASURE let
//! FFTW spend memory freely in pursuit of speed.

use criterion::{criterion_group, criterion_main};

use crate::backend::fftw_lib::fftw_c2r_all;
use crate::backend::phastft_lib::{phastft_c2r_f32, phastft_c2r_f64};
use crate::backend::realfft_lib::{realfft_c2r_f32, realfft_c2r_f64};

mod backend;
mod common;

criterion_group!(
    benches,
    phastft_c2r_f32,
    phastft_c2r_f64,
    realfft_c2r_f32,
    realfft_c2r_f64,
    fftw_c2r_all
);
criterion_main!(benches);
