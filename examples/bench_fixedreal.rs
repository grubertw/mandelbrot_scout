//! A/B benchmark: the fixed-point Mandelbrot reference-orbit step
//! (z = z^2 + c, with a shift-down after each multiply) implemented on the old
//! `num-bigint` backend vs. the new `rug`/GMP backend used by `FixedReal`.
//!
//! Both paths run the IDENTICAL algorithm at the same fixed-point shift, on the
//! IDENTICAL starting operands; only the big-integer type differs. GMP's edge
//! comes from (a) sub-quadratic multiply at large widths and (b) a dedicated
//! squaring fast-path (`square_ref`) that num-bigint lacks (it routes a square
//! through a general multiply).
//!
//! Run with:  cargo run --release --example bench_fixedreal
//!
//! `c` is built as a genuine full-width (shift-bit) integer so every multiply is
//! a real width-by-width multiply — not a sparse/power-of-two shortcut — which is
//! what governs deep-zoom reference-orbit cost. The width guard keeps the orbit
//! bounded (and thus the operands ~shift bits) without affecting the arithmetic.

use std::time::Instant;

use num_bigint::{BigInt, Sign};
use rug::integer::Order;
use rug::{Assign, Integer};

const SHIFTS: &[u32] = &[64, 256, 1024, 4096];

/// Iterations per shift, tuned so each run lands in the ~0.1–2s range.
fn iters_for(shift: u32) -> u64 {
    match shift {
        0..=64 => 2_000_000,
        65..=256 => 800_000,
        257..=1024 => 150_000,
        _ => 25_000,
    }
}

/// A deterministic full-width little-endian byte string of exactly `bits` bits
/// (top bit set so the operand truly occupies the full width). Same bytes feed
/// both backends, so they iterate identical values.
fn full_width_bytes(bits: u32, seed: u64) -> Vec<u8> {
    let nbytes = ((bits + 7) / 8) as usize;
    let mut out = vec![0u8; nbytes];
    let mut x = seed | 1;
    for b in out.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (x >> 33) as u8;
    }
    let top = nbytes - 1;
    let used = bits - top as u32 * 8; // 1..=8
    let mask: u8 = if used == 8 { 0xFF } else { (1u8 << used) - 1 };
    out[top] &= mask;
    out[top] |= 1u8 << (used - 1); // force the highest used bit on
    out
}

fn bigint_c(bits: u32, seed: u64) -> BigInt {
    BigInt::from_bytes_le(Sign::Plus, &full_width_bytes(bits, seed))
}

fn rug_c(bits: u32, seed: u64) -> Integer {
    Integer::from_digits(&full_width_bytes(bits, seed), Order::Lsf)
}

/// num-bigint: square via general multiply (no dedicated square path).
fn bench_bigint(shift: u32, iters: u64) -> (f64, u64) {
    let cre = bigint_c(shift, 0xC0FFEE);
    let cim = bigint_c(shift, 0xBADF00D);
    let cap = (shift + 4) as u64;
    let mut re = BigInt::ZERO;
    let mut im = BigInt::ZERO;

    let start = Instant::now();
    for _ in 0..iters {
        let re2 = (&re * &re) >> shift;
        let im2 = (&im * &im) >> shift;
        let reim = ((&re * &im) >> shift) << 1; // 2*re*im
        im = reim + &cim;
        re = (re2 - im2) + &cre;
        if re.bits() > cap || im.bits() > cap {
            re = cre.clone();
            im = cim.clone();
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (elapsed, re.bits().max(im.bits()))
}

/// rug/GMP: square via `square_ref` (GMP's dedicated squaring), multiply via mpz.
fn bench_rug(shift: u32, iters: u64) -> (f64, u64) {
    let cre = rug_c(shift, 0xC0FFEE);
    let cim = rug_c(shift, 0xBADF00D);
    let cap = shift + 4;
    let mut re = Integer::new();
    let mut im = Integer::new();

    let start = Instant::now();
    for _ in 0..iters {
        let re2 = Integer::from(re.square_ref()) >> shift;
        let im2 = Integer::from(im.square_ref()) >> shift;
        let reim = (Integer::from(&re * &im) >> shift) << 1u32; // 2*re*im
        im = reim + &cim;
        re = (re2 - im2) + &cre;
        if re.significant_bits() > cap || im.significant_bits() > cap {
            re.assign(&cre);
            im.assign(&cim);
        }
    }
    let elapsed = start.elapsed().as_secs_f64();
    (
        elapsed,
        re.significant_bits().max(im.significant_bits()) as u64,
    )
}

fn main() {
    println!(
        "{:>6} {:>10} {:>8} {:>12} {:>12} {:>12} {:>8}",
        "shift", "iters", "backend", "total(ms)", "ns/iter", "Miter/s", "bits"
    );
    println!("{}", "-".repeat(74));

    for &shift in SHIFTS {
        let iters = iters_for(shift);

        // Warm up each backend once before timing.
        let _ = bench_rug(shift, iters / 20 + 1);
        let _ = bench_bigint(shift, iters / 20 + 1);

        let (rug_s, rug_bits) = bench_rug(shift, iters);
        let (big_s, big_bits) = bench_bigint(shift, iters);

        let row = |name: &str, secs: f64, bits: u64| {
            println!(
                "{:>6} {:>10} {:>8} {:>12.1} {:>12.1} {:>12.3} {:>8}",
                shift,
                iters,
                name,
                secs * 1e3,
                secs / iters as f64 * 1e9,
                iters as f64 / secs / 1e6,
                bits,
            );
        };
        row("bigint", big_s, big_bits);
        row("rug", rug_s, rug_bits);
        println!(
            "{:>6} {:>10} {:>8} {:>26} {:>11.2}x",
            shift, "", "speedup", "", big_s / rug_s
        );
        println!();
    }
}
