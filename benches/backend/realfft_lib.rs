//! Important: this benchmark only measures small-to-mid sizes; criterion is
//! not a good fit for measuring long-running tasks — see
//! `examples/benchmark.rs` for the harness for large sizes.
//!
//! Unlike the C2C cross-library comparison (split across `bench.rs` vs.
//! `rustfft.rs` vs. `fftw_*.rs`), both PhastFT R2C/C2R and the realfft
//! baseline live in this single bench binary. The split-per-library
//! convention exists primarily to isolate FFTW's per-process wisdom cache
//! between planning modes; realfft has no such cache, so a single binary
//! suffices and gives a self-contained PhastFT-vs-realfft comparison.

use criterion::{BatchSize, BenchmarkId, Criterion};
use realfft::RealFftPlanner;

use crate::common::{
    bench_at_sizes, groups, ids, real_signal, spectrum_interleaved, spectrum_split,
    throughput_real, LENGTHS,
};
//
// Group names (snake_case): r2c_f32 / r2c_f64 / c2r_f32 / c2r_f64 — distinct
// from the C2C groups, so no overlay aggregation across binaries needed.

macro_rules! realfft_r2c {
    ($name:ident, $float:ty, $planner:ty, $fft_fn:ident, $group:expr) => {
        pub fn $name(c: &mut Criterion) {
            bench_at_sizes(c, $group, LENGTHS, throughput_real::<$float>, |g, len| {
                let mut rf_planner = RealFftPlanner::<$float>::new();
                let rf_r2c = rf_planner.plan_fft_forward(len);
                let mut rf_output = rf_r2c.make_output_vec();
                let mut rf_scratch = rf_r2c.make_scratch_vec();
                g.bench_function(BenchmarkId::new(ids::REALFFT, len), |b| {
                    b.iter_batched(
                        || real_signal::<$float>(len),
                        |mut input| {
                            rf_r2c
                                .process_with_scratch(&mut input, &mut rf_output, &mut rf_scratch)
                                .unwrap();
                            std::hint::black_box(&mut rf_output);
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

macro_rules! realfft_c2r {
    ($name:ident, $float:ty, $planner:ty, $fft_fn:ident, $group:expr) => {
        pub fn $name(c: &mut Criterion) {
            bench_at_sizes(c, $group, LENGTHS, throughput_real::<$float>, |g, len| {
                let mut rf_planner = RealFftPlanner::<$float>::new();
                let rf_c2r = rf_planner.plan_fft_inverse(len);
                let mut rf_output = rf_c2r.make_output_vec();
                let mut rf_scratch = rf_c2r.make_scratch_vec();
                g.bench_function(BenchmarkId::new(ids::REALFFT, len), |b| {
                    b.iter_batched(
                        || spectrum_interleaved::<$float>(len),
                        |mut input| {
                            rf_c2r
                                .process_with_scratch(&mut input, &mut rf_output, &mut rf_scratch)
                                .unwrap();
                            std::hint::black_box(&mut rf_output);
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

realfft_r2c!(
    realfft_r2c_f32,
    f32,
    PlannerR2c32,
    r2c_fft_f32_with_planner_and_opts,
    groups::R2C_F32
);
realfft_r2c!(
    realfft_r2c_f64,
    f64,
    PlannerR2c64,
    r2c_fft_f64_with_planner_and_opts,
    groups::R2C_F64
);
realfft_c2r!(
    realfft_c2r_f32,
    f32,
    PlannerR2c32,
    c2r_fft_f32_with_planner_and_opts,
    groups::C2R_F32
);
realfft_c2r!(
    realfft_c2r_f64,
    f64,
    PlannerR2c64,
    c2r_fft_f64_with_planner_and_opts,
    groups::C2R_F64
);
