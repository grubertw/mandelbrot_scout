use rug::{Complex, Float};

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
    seed: &Complex,
    period: usize
) -> Option<Complex> {
    let prec = seed.prec().0;
    let mut c = seed.clone();

    // Tunables
    let max_newton_iters = 10;
    let epsilon1 = Float::with_val(prec, EPS1);
    let epsilon2 = Float::with_val(prec, EPS2);
    let omega = Float::with_val(prec, OMEGA);
    let bailout = Float::with_val(prec, BAILOUT);

    // ----------------------------
    // 🔁 Newton iteration
    // ----------------------------
    for _ in 0..max_newton_iters {
        // z = f_c^p(0)
        // dz = d/dc f_c^p(0)
        let mut z = Complex::with_val(prec, 0);
        let mut dz = Complex::with_val(prec, 0);

        for _ in 0..period {
            // Save previous z
            let z_prev = z.clone();

            // z = z^2 + c
            z.square_mut();
            z += &c;

            // dz = 2*z_prev*dz + 1
            let mut tmp = z_prev;
            tmp *= 2;
            tmp *= &dz;
            tmp += 1;
            dz = tmp;
        }

        // Derivative too small → unstable
        if dz.clone().norm().real() < &epsilon2 {
            return None;
        }

        // Compute Newton step: delta = z / dz
        let mut delta = z.clone();
        delta /= &dz;

        // If step is exploding, abort
        if delta.clone().norm().real() > &omega {
            return None;
        }

        let mut c_next = c.clone();
        c_next -= &delta;

        // Convergence check
        let mut diff = c_next.clone();
        diff -= &c;

        if diff.clone().norm().real() < &epsilon1 {
            return Some(c_next);
        }

        // Divergence check (outside Mandelbrot bounds)
        if c_next.clone().norm().real() > &bailout {
            return None;
        }

        c = c_next;
    }

    // Final fallback: reject if not converged
    None
}