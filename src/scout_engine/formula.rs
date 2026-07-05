//! Fractal formula + parameterization: the two orthogonal axes that let
//! ScoutEngine render more than "core mandelbrot".
//!
//! - `Formula` is the iteration map f(z, c). It is the SINGLE source of truth
//!   for the CPU reference-orbit recurrence; the GPU shaders (naive +
//!   perturbation, f32 + fexp) mirror it.
//! - `Parameterization` decides how each pixel's anchor point maps to the
//!   concrete (z0, c) the recurrence starts from. It is ORTHOGONAL to `Formula`
//!   ("Julia of the burning ship" is a valid combination).
//!
//! Milestone 1 only wires `Formula::Power { power: 2 }` + `Mandelbrot`, which
//! is behavior-identical to the old hard-coded z^2 + c path.

use crate::numerics::{FixedComplex, FixedReal};

/// Which dynamical-plane the image represents. ORTHOGONAL to `Formula`: any
/// formula can be shown as a Mandelbrot parameter-plane or a Julia
/// dynamical-plane.
#[derive(Clone, Debug)]
pub enum Parameterization {
    /// Escape-time over the parameter `c`; every pixel seeds z0 = 0.
    /// The anchor point in the image IS the c value.
    Mandelbrot,
    /// Escape-time over the initial `z0`; `c` is fixed image-wide.
    /// The anchor point in the image IS the z0 value.
    Julia { c: FixedComplex },
}

impl Parameterization {
    /// Turn an anchor (an orbit's location in the image plane) into the
    /// concrete (z0, c) the reference recurrence starts from.
    ///
    /// The Julia `c` must already live at the anchor's shift — the owner of the
    /// `Parameterization` is responsible for rescaling it per view (see
    /// `rescale_to`), exactly as `c_ref` is rescaled for scoring.
    pub fn seed(&self, anchor: &FixedComplex) -> (FixedComplex, FixedComplex) {
        match self {
            Parameterization::Mandelbrot =>
                (FixedComplex::zero(anchor.re().shift), anchor.clone()),
            Parameterization::Julia { c } => {
                debug_assert_eq!(
                    c.re().shift, anchor.re().shift,
                    "Julia c must be rescaled to the anchor's shift before seeding"
                );
                (anchor.clone(), c.clone())
            }
        }
    }

    /// True for the Mandelbrot parameterization. Julia reference orbits are
    /// viewport-centered (Z[0] != 0), which the current BLA leaf/radius math
    /// does not support, so callers gate BLA off unless this holds.
    pub fn is_mandelbrot(&self) -> bool {
        matches!(self, Parameterization::Mandelbrot)
    }

    /// Rescale any embedded fixed-point constant to `shift` so it matches the
    /// precision of the orbits spawned this cycle. No-op for Mandelbrot.
    pub fn rescale_to(&mut self, shift: u32) {
        if let Parameterization::Julia { c } = self {
            let delta = shift as i32 - c.re().shift as i32;
            if delta != 0 {
                c.rescale(delta);
            }
        }
    }
}

/// The iteration map f(z, c). This enum is the single source of truth for the
/// CPU formula; the GPU shaders mirror it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Formula {
    /// z_{n+1} = z^power + c.  power == 2 is the classic set.  Holomorphic.
    Power { power: u32 },
    /// z_{n+1} = (|Re z| + i|Im z|)^2 + c.  Non-holomorphic (no scalar deriv).
    BurningShip,
}

impl Default for Formula {
    fn default() -> Self {
        Formula::Power { power: 2 }
    }
}

impl Formula {
    /// One full-precision reference step: z <- f(z, c). The ONLY place the CPU
    /// formula lives. Called from `ReferenceOrbit::compute_to`.
    #[inline]
    pub fn ref_step(&self, z: &FixedComplex, c: &FixedComplex) -> FixedComplex {
        let mut acc = match *self {
            Formula::Power { power } => ipow(z, power),
            Formula::BurningShip => {
                // fold into the first quadrant, then square
                FixedComplex::new(z.re().abs(), z.im().abs()).square()
            }
        };
        acc += c;
        acc
    }

    /// Whether a complex-scalar derivative exists (gates the concept of a scalar
    /// BLA + analytic DE). `BurningShip` returns false — it needs the mat2
    /// Jacobian path.
    pub fn is_holomorphic(&self) -> bool {
        matches!(self, Formula::Power { .. })
    }

    /// Whether the GPU BLA path is implemented for this formula. BLA needs the
    /// leaf `A = f'(Z) = p*Z^(p-1)` AND a validity-radius formula; the whole
    /// Power family now has both (leaf radius = eps*|Z|/(p-1), see bla.rs).
    /// Burning-ship's mat2 BLA is still future work.
    pub fn bla_supported(&self) -> bool {
        matches!(self, Formula::Power { .. })
    }

    /// The power for the Power family (used to build the BLA leaf coefficient).
    /// Defaults to 2 for any non-power formula (BLA is gated off for those).
    pub fn power(&self) -> u32 {
        match self {
            Formula::Power { power } => *power,
            Formula::BurningShip => 2,
        }
    }
}

/// z^power in full precision. power == 2 hits the tuned GMP `square_ref`
/// fast-path (do not lose it — ~2x on the hot square). Larger powers fall back
/// to naive repeated complex `Mul`, which is acceptable since deep zooms are
/// power-2 only.
#[inline]
fn ipow(z: &FixedComplex, power: u32) -> FixedComplex {
    match power {
        0 => FixedComplex::new(
            FixedReal::from_f64(1.0, z.re().shift),
            FixedReal::zero(z.re().shift),
        ),
        1 => z.clone(),
        2 => z.square(), // fast-path — keep this arm
        _ => {
            let mut acc = z.square();
            for _ in 2..power {
                acc = acc * z.clone(); // Mul for FixedComplex is owned
            }
            acc
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIFT: u32 = 64;

    fn fc(re: f64, im: f64) -> FixedComplex {
        FixedComplex::with_val_f64((re, im), SHIFT)
    }

    fn approx(a: &FixedComplex, re: f64, im: f64) {
        assert!((a.re().to_f64_lossy() - re).abs() < 1e-9, "re: {} != {}", a.re().to_f64_lossy(), re);
        assert!((a.im().to_f64_lossy() - im).abs() < 1e-9, "im: {} != {}", a.im().to_f64_lossy(), im);
    }

    #[test]
    fn power2_matches_hardcoded_square_plus_c() {
        // Old path: z.square() + c. New path: Formula::Power{2}.ref_step.
        let z = fc(0.3, -0.4);
        let c = fc(-0.5, 0.2);
        let expected = {
            let mut s = z.square();
            s += &c;
            s
        };
        let got = Formula::Power { power: 2 }.ref_step(&z, &c);
        approx(&got, expected.re().to_f64_lossy(), expected.im().to_f64_lossy());
    }

    #[test]
    fn power3_matches_complex_cube() {
        // (0.3 - 0.4i)^3 = 0.3^3 - 3*0.3*0.4^2  + i(3*0.3^2*(-0.4) - (-0.4)^3-ish)
        // just compute via num-style by hand: (a+bi)^3, a=0.3,b=-0.4
        let a = 0.3_f64;
        let b = -0.4_f64;
        let re = a * a * a - 3.0 * a * b * b;
        let im = 3.0 * a * a * b - b * b * b;
        let got = Formula::Power { power: 3 }.ref_step(&fc(a, b), &fc(0.0, 0.0));
        approx(&got, re, im);
    }

    #[test]
    fn burning_ship_folds_before_squaring() {
        // (|Re|+i|Im|)^2 + c with z = (-0.3, -0.4) -> folds to (0.3, 0.4)
        let a = 0.3_f64;
        let b = 0.4_f64;
        let re = a * a - b * b;
        let im = 2.0 * a * b;
        let got = Formula::BurningShip.ref_step(&fc(-0.3, -0.4), &fc(0.0, 0.0));
        approx(&got, re, im);
    }

    #[test]
    fn mandelbrot_seed_is_zero_z0_anchor_c() {
        let anchor = fc(-0.75, 0.1);
        let (z0, c) = Parameterization::Mandelbrot.seed(&anchor);
        approx(&z0, 0.0, 0.0);
        approx(&c, -0.75, 0.1);
    }

    // Mirror of the WGSL f_perturb_step binomial accumulation (same coefficient
    // recurrence, same term order). Pins the math the shaders rely on for
    // Power-of-X perturbation; if this matches (Z+dz)^p - Z^p, the shaders do too.
    fn perturb_binomial(z: num_complex::Complex64, dz: num_complex::Complex64, p: u32)
        -> num_complex::Complex64 {
        use num_complex::Complex64;
        let mut zpows = vec![Complex64::new(1.0, 0.0); (p + 1) as usize];
        for i in 1..=p as usize { zpows[i] = zpows[i - 1] * z; }
        let mut result = Complex64::new(0.0, 0.0);
        let mut dzk = Complex64::new(1.0, 0.0);
        let mut binom = 1.0f64;
        for k in 1..=p {
            dzk *= dz;
            binom = binom * (p - k + 1) as f64 / k as f64;
            result += zpows[(p - k) as usize] * dzk * binom;
        }
        result
    }

    #[test]
    fn perturb_binomial_matches_direct_difference() {
        use num_complex::Complex64;
        let z = Complex64::new(-0.37, 0.62);
        let dz = Complex64::new(1e-5, -2e-5);   // small delta, as in perturbation
        for p in 2u32..=7 {
            let got = perturb_binomial(z, dz, p);
            let direct = (z + dz).powu(p) - z.powu(p); // = f(Z+dz) - f(Z)
            // Tolerance sits above the direct form's own cancellation floor
            // (~1e-11 rel here — the binomial is the more accurate one) yet far
            // below the ~O(1) error a wrong binomial coefficient would produce.
            assert!((got - direct).norm() < 1e-7 * direct.norm(),
                "p={}: binomial={:?} direct={:?}", p, got, direct);
        }
    }

    #[test]
    fn perturb_binomial_power2_is_classic_form() {
        use num_complex::Complex64;
        // p=2 must equal 2*Z*dz + dz^2 (the proven power-2 recurrence, sans dc).
        let z = Complex64::new(0.3, -0.8);
        let dz = Complex64::new(0.01, 0.02);
        let got = perturb_binomial(z, dz, 2);
        let classic = (z + z) * dz + dz * dz;
        assert!((got - classic).norm() < 1e-14);
    }

    #[test]
    fn julia_seed_is_anchor_z0_const_c() {
        let anchor = fc(0.2, 0.3);
        let param = Parameterization::Julia { c: fc(-0.8, 0.156) };
        let (z0, c) = param.seed(&anchor);
        approx(&z0, 0.2, 0.3);
        approx(&c, -0.8, 0.156);
    }
}
