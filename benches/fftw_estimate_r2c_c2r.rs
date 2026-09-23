//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! Each FFTW planning mode (Estimate / Measure / Conserve) lives in its own
//! `[[bench]]` binary so FFTW's global per-process wisdom cache cannot leak
//! between modes; each run starts with a fresh process and empty wisdom.
//! Group names are shared with `bench.rs` / `rustfft.rs` / the other
//! `fftw_*.rs` binaries; criterion does NOT auto-aggregate across binaries
//! — use `benches/plot_criterion_overlay.py` for the cross-binary overlay.

use criterion::{criterion_group, criterion_main, Criterion};
use fftw::types::Flag;

mod common;
mod fftw_lib;

fn run(c: &mut Criterion) {
    fftw_lib::run_r2c_c2r(
        c,
        common::ids::FFTW_ESTIMATE_R2C,
        common::ids::FFTW_ESTIMATE_C2R,
        Flag::DESTROYINPUT | Flag::ESTIMATE,
    );
}

criterion_group!(benches, run);
criterion_main!(benches);
