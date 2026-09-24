//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! Each FFTW planning mode (Estimate / Measure / Conserve) lives in its own
//! `[[bench]]` binary so FFTW's global per-process wisdom cache cannot leak
//! between modes; each run starts with a fresh process and empty wisdom.
//! Group names are shared with `bench.rs` / `rustfft.rs` / the other
//! `fftw_*_c2c.rs` binaries; criterion does NOT auto-aggregate across binaries
//! — use `benches/plot_criterion_overlay.py` for the cross-binary overlay.
//!
//! This series combines MEASURE with `FFTW_CONSERVE_MEMORY` so FFTW selects
//! memory-efficient plan variants during the same search it would run for
//! plain MEASURE. This is a more faithful comparison for PhastFT, which is
//! also designed for low memory overhead — plain PATIENT / MEASURE let
//! FFTW spend memory freely in pursuit of speed.

use criterion::{criterion_group, criterion_main, Criterion};
use fftw::types::Flag;

use crate::backend::phastft_lib::{phastft_c2c_fwd_f32, phastft_c2c_fwd_f64};
use crate::backend::rustfft_lib::{rustfft_fwd_f32, rustfft_fwd_f64};

mod backend;
mod common;

fn fftw_fwd(c: &mut Criterion) {
    backend::fftw_lib::run_c2c_forward(
        c,
        common::ids::FFTW_CONSERVE_C2C,
        Flag::DESTROYINPUT | Flag::MEASURE | Flag::CONSERVEMEMORY,
    );
    backend::fftw_lib::run_c2c_forward(
        c,
        common::ids::FFTW_ESTIMATE_C2C,
        Flag::DESTROYINPUT | Flag::ESTIMATE,
    );
    backend::fftw_lib::run_c2c_forward(
        c,
        common::ids::FFTW_MEASURE_C2C,
        Flag::DESTROYINPUT | Flag::MEASURE,
    );
}

criterion_group!(
    benches,
    phastft_c2c_fwd_f32,
    phastft_c2c_fwd_f64,
    rustfft_fwd_f32,
    rustfft_fwd_f64,
    fftw_fwd
);
criterion_main!(benches);
