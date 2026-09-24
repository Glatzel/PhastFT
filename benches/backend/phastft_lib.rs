//! Important: this benchmark only measures small-to-mid sizes, which are
//! not the focus of PhastFT. Criterion is not a good fit for measuring
//! long-running tasks — see `examples/benchmark.rs` for the harness for
//! large sizes.
//!
//! The PhastFT, RustFFT, and FFTW bench binaries all write into the same
//! `target/criterion/<group>/<id>/<size>/` tree; criterion does NOT
//! auto-aggregate across binaries, so use
//! `benches/plot_criterion_overlay.py` to produce a single overlay plot per
//! group after running them all.

use criterion::{BatchSize, BenchmarkId, Criterion};
use phastft::options::Options;
use phastft::planner::{Direction, PlannerDit32, PlannerDit64, PlannerR2c32, PlannerR2c64};
use phastft::{
    c2r_fft_f32_with_planner_and_opts, c2r_fft_f64_with_planner_and_opts,
    fft_f32_dit_with_planner_and_opts, fft_f64_dit_with_planner_and_opts,
    r2c_fft_f32_with_planner_and_opts, r2c_fft_f64_with_planner_and_opts,
};

use crate::common::{
    bench_at_sizes, groups, ids, real_signal, spectrum_split, split_complex, throughput_complex,
    throughput_real, LENGTHS,
};

macro_rules! phastft_c2c {
    ($name:ident, $float:ty, $planner:ty, $fft:ident, $dir:expr, $group:expr) => {
        #[allow(
            dead_code,
            reason = "Functions are shared across targets, but not every target uses all of them."
        )]
        pub fn $name(c: &mut Criterion) {
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                |g, len| {
                    let opts = Options::guess_options(len);
                    let planner = <$planner>::new(len);
                    g.bench_function(BenchmarkId::new(ids::PHASTFT_DIT, len), |b| {
                        b.iter_batched(
                            || split_complex::<$float>(len),
                            |(mut reals, mut imags)| {
                                $fft(&mut reals, &mut imags, $dir, &planner, &opts);
                                std::hint::black_box((&mut reals, &mut imags));
                            },
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}
macro_rules! phastft_r2c {
    ($name:ident, $float:ty, $planner:ty, $fft_fn:ident, $group:expr) => {
        #[allow(
            dead_code,
            reason = "Functions are shared across targets, but not every target uses all of them."
        )]
        pub fn $name(c: &mut Criterion) {
            bench_at_sizes(c, $group, LENGTHS, throughput_real::<$float>, |g, len| {
                // Plan + output buffers allocated outside iter_batched —
                // planning and allocation cost is excluded from per-sample timings.
                let phast_planner = <$planner>::new(len);
                let phast_opts = Options::guess_options(len / 2);
                let mut phast_re = vec![0 as $float; len / 2 + 1];
                let mut phast_im = vec![0 as $float; len / 2 + 1];
                g.bench_function(BenchmarkId::new(ids::PHASTFT_R2C, len), |b| {
                    b.iter_batched(
                        || real_signal::<$float>(len),
                        |input| {
                            $fft_fn(
                                &input,
                                &mut phast_re,
                                &mut phast_im,
                                &phast_planner,
                                &phast_opts,
                            );
                            std::hint::black_box((&mut phast_re, &mut phast_im));
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}

macro_rules! phastft_c2r {
    ($name:ident, $float:ty, $planner:ty, $fft_fn:ident, $group:expr) => {
        #[allow(
            dead_code,
            reason = "Functions are shared across targets, but not every target uses all of them."
        )]
        pub fn $name(c: &mut Criterion) {
            bench_at_sizes(c, $group, LENGTHS, throughput_real::<$float>, |g, len| {
                let phast_planner = <$planner>::new(len);
                let phast_opts = Options::guess_options(len / 2);
                let mut phast_output = vec![0 as $float; len];
                let mut phast_scratch_re = vec![0 as $float; len / 2];
                let mut phast_scratch_im = vec![0 as $float; len / 2];
                g.bench_function(BenchmarkId::new(ids::PHASTFT_C2R, len), |b| {
                    b.iter_batched(
                        || spectrum_split::<$float>(len),
                        |(input_re, input_im)| {
                            $fft_fn(
                                &input_re,
                                &input_im,
                                &mut phast_output,
                                &phast_planner,
                                &phast_opts,
                                &mut phast_scratch_re,
                                &mut phast_scratch_im,
                            );
                            std::hint::black_box(&mut phast_output);
                        },
                        BatchSize::SmallInput,
                    );
                });
            });
        }
    };
}
phastft_c2c!(
    phastft_c2c_fwd_f32,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts,
    Direction::Forward,
    groups::C2C_FORWARD_F32
);
phastft_c2c!(
    phastft_c2c_inv_f32,
    f32,
    PlannerDit32,
    fft_f32_dit_with_planner_and_opts,
    Direction::Inverse,
    groups::C2C_INVERSE_F32
);
phastft_c2c!(
    phastft_c2c_fwd_f64,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts,
    Direction::Forward,
    groups::C2C_FORWARD_F64
);
phastft_c2c!(
    phastft_c2c_inv_f64,
    f64,
    PlannerDit64,
    fft_f64_dit_with_planner_and_opts,
    Direction::Inverse,
    groups::C2C_INVERSE_F64
);
phastft_r2c!(
    phastft_r2c_f32,
    f32,
    PlannerR2c32,
    r2c_fft_f32_with_planner_and_opts,
    groups::R2C_F32
);
phastft_r2c!(
    phastft_r2c_f64,
    f64,
    PlannerR2c64,
    r2c_fft_f64_with_planner_and_opts,
    groups::R2C_F64
);
phastft_c2r!(
    phastft_c2r_f32,
    f32,
    PlannerR2c32,
    c2r_fft_f32_with_planner_and_opts,
    groups::C2R_F32
);
phastft_c2r!(
    phastft_c2r_f64,
    f64,
    PlannerR2c64,
    c2r_fft_f64_with_planner_and_opts,
    groups::C2R_F64
);
