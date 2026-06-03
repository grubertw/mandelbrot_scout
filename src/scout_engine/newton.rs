use crate::numerics::{FixedComplex, FixedReal};

pub const MAX_P: u32 = 32;
const EPS1: f64 = 1e-30;
const EPS2: f64 = 1e-40;
const OMEGA: f64 = 1e-6;
const BAILOUT: f64 = 16.0;

/// Attempt to refine a complex 'c' using Newton's method
/// to find a nearby hyperbolic component center.
///
/// Returns:
///   Some(refined_c) if converged
///   None if rejected / diverged / unsafe
pub fn try_newton_refine(
    seed: &FixedComplex,
    period: usize
) -> Option<FixedComplex> {
    let shift = seed.re.shift;
    let mut c = seed.clone();

    // Tunables
    let max_newton_iters = 10;
    let epsilon1 = FixedReal::from_f64(EPS1, shift);
    let epsilon2 = FixedReal::from_f64(EPS2, shift);
    let omega = FixedReal::from_f64(OMEGA, shift);
    let bailout = FixedReal::from_f64(BAILOUT, shift);

    // ----------------------------
    // 🔁 Newton iteration
    // ----------------------------
    for _ in 0..max_newton_iters {
        // z = f_c^p(0)
        // dz = d/dc f_c^p(0)
        let mut z = FixedComplex::zero(shift);
        let mut dz = FixedComplex::zero(shift);

        for _ in 0..period {
            // Save previous z
            let z_prev = z.clone();

            // z = z^2 + c
            z = z.square();
            z = z.clone() + c.clone();

            // dz = 2*z_prev*dz + 1
            let mut tmp = z_prev;
            tmp = tmp.clone() * FixedComplex::with_val_f64((2.0, 2.0) , shift);
            tmp = tmp.clone() * dz;
            tmp = tmp.clone() + FixedComplex::with_val_f64((1.0, 1.0) , shift);
            dz = tmp;
        }

        // Derivative too small → unstable
        if dz.clone().norm() < epsilon2 {
            return None;
        }

        // Compute Newton step: delta = z / dz
        let mut delta = z.clone();
        delta = delta.clone() / dz.clone();

        // If step is exploding, abort
        if delta.clone().norm() > omega {
            return None;
        }

        let mut c_next = c.clone();
        c_next = c_next.clone() - delta;

        // Convergence check
        let mut diff = c_next.clone();
        diff = diff.clone() - c;

        if diff.clone().norm() < epsilon1 {
            return Some(c_next);
        }

        // Divergence check (outside Mandelbrot bounds)
        if c_next.clone().norm() > bailout {
            return None;
        }

        c = c_next;
    }

    // Final fallback: reject if not converged
    None
}