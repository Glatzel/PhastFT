//! Shared FFTW C2C bench logic. Three `[[bench]]` binaries
//! (`fftw_estimate`, `fftw_measure`, `fftw_conserve`) each call `run_all`
//! with their own series ID and `Flag` set; the per-binary split exists
//! only so FFTW's per-process wisdom cache cannot leak between planning
//! modes (every run starts with a fresh process and empty wisdom).
//!
//! Like `common/mod.rs`, this is a plain shared module under `benches/`,
//! not a `[[bench]]` target — each consumer pulls it in with `mod
//! fftw_lib;` and compiles its own copy.

#![allow(dead_code)]

use std::ptr::slice_from_raw_parts_mut;

use criterion::{BatchSize, BenchmarkId, Criterion};
use fftw::array::AlignedVec;
use fftw::plan::{
    C2CPlan, C2CPlan32, C2CPlan64, C2RPlan, C2RPlan32, C2RPlan64, R2CPlan, R2CPlan32, R2CPlan64,
    R2RPlan, R2RPlan32, R2RPlan64,
};
use fftw::types::{Flag, R2RKind, Sign};
use utilities::rustfft::num_complex::Complex;

use crate::common::{
    bench_at_sizes, groups, real_signal, split_complex, throughput_complex, throughput_real,
    LENGTHS,
};

macro_rules! fftw_sweep_c2c {
    ($name:ident, $float:ty, $plan:ty, $sign:expr, $group:expr) => {
        fn $name(c: &mut Criterion, id: &str, flags: Flag) {
            // `Flag` (bitflags 2 in the fftw crate) does not derive `Copy`,
            // so we can't capture it by value across iterations of an FnMut
            // closure. Round-trip through the underlying `u32` bits, which
            // are Copy, and reconstruct the flag set per `n`.
            let flag_bits = flags.bits();
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_complex::<$float>,
                move |g, len| {
                    let mut plan =
                        <$plan>::aligned(&[len], $sign, Flag::from_bits_retain(flag_bits)).unwrap();
                    g.bench_function(BenchmarkId::new(id, len), |b| {
                        b.iter_batched(
                            || {
                                let (reals, imags) = split_complex::<$float>(len);
                                let mut nums: AlignedVec<Complex<$float>> = AlignedVec::new(len);
                                for (z, (&re, &im)) in
                                    nums.iter_mut().zip(reals.iter().zip(imags.iter()))
                                {
                                    *z = Complex::new(re, im);
                                }
                                nums
                            },
                            |mut nums| {
                                plan.c2c(
                                    // SAFETY: identical shape to examples/fftwrb.rs:42-48.
                                    // DESTROYINPUT permits in-place c2c; the raw slice
                                    // aliases `nums` for the duration of the call and
                                    // `len` matches the AlignedVec allocation.
                                    unsafe {
                                        &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), len)
                                    },
                                    &mut nums,
                                )
                                .unwrap();
                                std::hint::black_box(&mut nums);
                            },
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

macro_rules! fftw_sweep_r2c {
    ($name:ident, $float:ty, $plan:ty, $group:expr) => {
        fn $name(c: &mut Criterion, id: &str, flags: Flag) {
            // `Flag` (bitflags 2 in the fftw crate) does not derive `Copy`,
            // so we can't capture it by value across iterations of an FnMut
            // closure. Round-trip through the underlying `u32` bits, which
            // are Copy, and reconstruct the flag set per `n`.
            let flag_bits = flags.bits();
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_real::<$float>,
                move |g, len| {
                    let mut plan =
                        <$plan>::aligned(&[len], Flag::from_bits_retain(flag_bits)).unwrap();
                    g.bench_function(BenchmarkId::new(id, len), |b| {
                        b.iter_batched(
                            || {
                                // Real input of length `len`; the r2c output
                                // is the conjugate-symmetric half-spectrum,
                                // length `len / 2 + 1`.
                                let signal = real_signal::<$float>(len);
                                let mut input: AlignedVec<$float> = AlignedVec::new(len);
                                input.copy_from_slice(&signal);
                                let output: AlignedVec<Complex<$float>> =
                                    AlignedVec::new(len / 2 + 1);
                                (input, output)
                            },
                            |(mut input, mut output)| {
                                plan.r2c(&mut input, &mut output).unwrap();
                                std::hint::black_box(&mut output);
                            },
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

macro_rules! fftw_sweep_c2r {
    ($name:ident, $float:ty, $plan:ty, $group:expr) => {
        fn $name(c: &mut Criterion, id: &str, flags: Flag) {
            // `Flag` (bitflags 2 in the fftw crate) does not derive `Copy`,
            // so we can't capture it by value across iterations of an FnMut
            // closure. Round-trip through the underlying `u32` bits, which
            // are Copy, and reconstruct the flag set per `n`.
            let flag_bits = flags.bits();
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_real::<$float>,
                move |g, len| {
                    let mut plan =
                        <$plan>::aligned(&[len], Flag::from_bits_retain(flag_bits)).unwrap();
                    g.bench_function(BenchmarkId::new(id, len), |b| {
                        b.iter_batched(
                            || {
                                // Complex half-spectrum input, length
                                // `len / 2 + 1`; real output of length `len`.
                                let (reals, imags) = split_complex::<$float>(len / 2 + 1);
                                let mut input: AlignedVec<Complex<$float>> =
                                    AlignedVec::new(len / 2 + 1);
                                for (z, (&re, &im)) in
                                    input.iter_mut().zip(reals.iter().zip(imags.iter()))
                                {
                                    *z = Complex::new(re, im);
                                }
                                let output: AlignedVec<$float> = AlignedVec::new(len);
                                (input, output)
                            },
                            |(mut input, mut output)| {
                                plan.c2r(&mut input, &mut output).unwrap();
                                std::hint::black_box(&mut output);
                            },
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

macro_rules! fftw_sweep_r2r {
    ($name:ident, $float:ty, $plan:ty, $group:expr, $kind:expr) => {
        fn $name(c: &mut Criterion, id: &str, flags: Flag) {
            // `Flag` (bitflags 2 in the fftw crate) does not derive `Copy`,
            // so we can't capture it by value across iterations of an FnMut
            // closure. Round-trip through the underlying `u32` bits, which
            // are Copy, and reconstruct the flag set per `n`.
            let flag_bits = flags.bits();
            bench_at_sizes(
                c,
                $group,
                LENGTHS,
                throughput_real::<$float>,
                move |g, len| {
                    // R2R has no Sign — direction/kind (DCT-II, DST-I, ...)
                    // is baked into `$kind` (an R2RKind), passed per-axis.
                    let mut plan =
                        <$plan>::aligned(&[len], $kind, Flag::from_bits_retain(flag_bits)).unwrap();
                    g.bench_function(BenchmarkId::new(id, len), |b| {
                        b.iter_batched(
                            || {
                                // Real in, real out — no Complex involved.
                                let (reals, _imags) = split_complex::<$float>(len);
                                let mut nums: AlignedVec<$float> = AlignedVec::new(len);
                                nums.copy_from_slice(&reals);
                                nums
                            },
                            |mut nums| {
                                plan.r2r(
                                    // SAFETY: in-place r2r transform; the raw
                                    // slice aliases `nums` for the duration
                                    // of the call and `len` matches the
                                    // AlignedVec allocation.
                                    unsafe {
                                        &mut *slice_from_raw_parts_mut(nums.as_mut_ptr(), len)
                                    },
                                    &mut nums,
                                )
                                .unwrap();
                                std::hint::black_box(&mut nums);
                            },
                            BatchSize::SmallInput,
                        );
                    });
                },
            );
        }
    };
}

fftw_sweep_c2c!(
    c2c_fwd_f32,
    f32,
    C2CPlan32,
    Sign::Forward,
    groups::C2C_FORWARD_F32
);
fftw_sweep_c2c!(
    c2c_inv_f32,
    f32,
    C2CPlan32,
    Sign::Backward,
    groups::C2C_INVERSE_F32
);
fftw_sweep_c2c!(
    c2c_fwd_f64,
    f64,
    C2CPlan64,
    Sign::Forward,
    groups::C2C_FORWARD_F64
);
fftw_sweep_c2c!(
    c2c_inv_f64,
    f64,
    C2CPlan64,
    Sign::Backward,
    groups::C2C_INVERSE_F64
);

fftw_sweep_r2c!(r2c_f32, f32, R2CPlan32, groups::R2C_F32);
fftw_sweep_r2c!(r2c_f64, f64, R2CPlan64, groups::R2C_F64);
fftw_sweep_c2r!(c2r_f32, f32, C2RPlan32, groups::C2R_F32);
fftw_sweep_c2r!(c2r_f64, f64, C2RPlan64, groups::C2R_F64);

fftw_sweep_r2r!(
    r2hc_f32,
    f32,
    R2RPlan32,
    groups::R2C_F32,
    R2RKind::FFTW_R2HC
);
fftw_sweep_r2r!(
    r2hc_f64,
    f64,
    R2RPlan64,
    groups::R2C_F64,
    R2RKind::FFTW_R2HC
);
fftw_sweep_r2r!(
    hc2r_f32,
    f32,
    R2RPlan32,
    groups::C2R_F32,
    R2RKind::FFTW_HC2R
);
fftw_sweep_r2r!(
    hc2r_f64,
    f64,
    R2RPlan64,
    groups::C2R_F64,
    R2RKind::FFTW_HC2R
);

/// Run all four c2c_* groups (forward/inverse × f32/f64) with the given
/// FFTW `flags` and series `id`. The three per-mode bench binaries each
/// call this once with their own `Flag` set.
pub fn run_c2c(c: &mut Criterion, id: &str, flags: Flag) {
    // `Flag` isn't `Copy`, so reconstruct from its u32 bits for each call.
    let bits = flags.bits();
    let mk = || Flag::from_bits_retain(bits);

    c2c_fwd_f32(c, id, mk());
    c2c_inv_f32(c, id, mk());
    c2c_fwd_f64(c, id, mk());
    c2c_inv_f64(c, id, mk());
}

/// Run all four r2c/c2r groups (r2c/c2r × f32/f64) with the given
/// FFTW `flags` and series `id`. The three per-mode bench binaries each
/// call this once with their own `Flag` set.
pub fn run_r2c_c2r(c: &mut Criterion, id_r2c: &str, id_c2r: &str, flags: Flag) {
    // `Flag` isn't `Copy`, so reconstruct from its u32 bits for each call.
    let bits = flags.bits();
    let mk = || Flag::from_bits_retain(bits);
    r2c_f32(c, id_r2c, mk());
    r2c_f64(c, id_r2c, mk());
    c2r_f32(c, id_c2r, mk());
    c2r_f64(c, id_c2r, mk());
}

/// Run all four r2c/c2r groups (r2c/c2r × f32/f64) with the given
/// FFTW `flags` and series `id`. The three per-mode bench binaries each
/// call this once with their own `Flag` set.
pub fn run_r2r(c: &mut Criterion, id: &str, flags: Flag) {
    // `Flag` isn't `Copy`, so reconstruct from its u32 bits for each call.
    let bits = flags.bits();
    let mk = || Flag::from_bits_retain(bits);
    r2hc_f32(c, id, mk());
    r2hc_f64(c, id, mk());
    hc2r_f32(c, id, mk());
    hc2r_f64(c, id, mk());
}
