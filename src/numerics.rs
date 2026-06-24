use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::num::ParseIntError;
use bytemuck;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::str::FromStr;
use rug::Integer;
use rug::integer::ParseIntegerError;

pub const DEFAULT_BAILOUT: f64 = 4.0;
pub const MAX_SAFE_DF_MAG: f64 = 1e30;

/// Wraps a rug::Integer (GMP mpz) and a u32 to represent a fixed-point value.
/// The final number is represented as mantissa * 2^-shift, where
/// mantissa is a rug::Integer - GMP's arbitrary-precision integer, giving us
/// sub-quadratic multiply and a dedicated squaring fast-path - and
/// shift is a u32. Math operations on this type are kept minimal,
/// and the shift never changes. Instead, the shift is only updated
/// on zoom operations and on consideration of the viewport scale.
/// From the perspective of this class, shift should never be
/// modified once the object is allocated.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FixedReal {
    pub mantissa: Integer,
    pub shift: u32,
}

/// A real plus an imaginary component Fixed values are put together to
/// form a complex number. When creating a new FixedComplex, the shift
/// of both the real and imaginary components must be the same.
#[derive(Clone, Debug)]
pub struct FixedComplex {
    pub re: FixedReal,
    pub im: FixedReal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseFixedError {
    InvalidFormat,
    InvalidBits(ParseIntegerError),
    InvalidShift(ParseIntError),
}

impl FixedReal {
    pub fn new(bits: Integer, shift: u32) -> Self {
        Self { mantissa: bits, shift }
    }

    pub fn zero(shift: u32) -> Self {
        Self::new(Integer::new(), shift)
    }

    /// Number of significant bits in the mantissa (magnitude). Lets callers
    /// reason about precision headroom without reaching into the backing type.
    pub fn bits(&self) -> u64 {
        self.mantissa.significant_bits() as u64
    }

    /// True when the value is exactly zero, regardless of shift.
    pub fn is_zero(&self) -> bool {
        self.mantissa.cmp0() == std::cmp::Ordering::Equal
    }

    pub fn from_f32(float: f32, shift: u32) -> Self {
        Self::from_f64(float as f64, shift)
    }

    pub fn from_f64(float: f64, shift: u32) -> Self {
        let scaled = float * 2.0_f64.powi(shift as i32);

        Self::new(
            Integer::from_f64(scaled.round()).unwrap(),
            shift,
        )
    }

    pub fn to_f64_lossy(&self) -> f64 {
        const MAX_SHIFT: u32 = 960;

        let original_bits = self.mantissa.clone();

        let mut bits = self.mantissa.clone();
        let mut shift = self.shift;

        if shift > MAX_SHIFT {
            let reduction = shift - MAX_SHIFT;

            bits >>= reduction;

            if bits.cmp0() != std::cmp::Ordering::Equal {
                shift = MAX_SHIFT;
            } else {
                bits = original_bits;
            }
        }

        // rug's to_f64 already rounds to nearest and saturates to +/-inf when
        // the magnitude exceeds f64 range, so no fallback is needed.
        let value = bits.to_f64();

        value / 2.0_f64.powi(shift as i32)
    }

    pub fn to_f32_lossy(&self) -> f32 {
        const TARGET_BITS: u64 = 30;

        let mut bits = self.mantissa.clone();
        let mut shift = self.shift;

        let bit_count = bits.significant_bits() as u64;

        if bit_count > TARGET_BITS {
            let reduction = bit_count - TARGET_BITS;

            bits >>= reduction as u32;
            shift -= reduction as u32;
        }

        let numerator = bits.to_f64();

        (numerator / 2.0_f64.powi(shift as i32)) as f32
    }
    
    pub fn abs(&self) -> Self {
        Self::new(Integer::from(self.mantissa.abs_ref()), self.shift)
    }

    pub fn square_ref(x: &Self) -> Self {
        // square_ref() routes through GMP's dedicated squaring path (~2x faster
        // than a general multiply), then shift back down to the fixed point.
        Self::new(
            Integer::from(x.mantissa.square_ref()) >> x.shift,
            x.shift
        )
    }

    pub fn scale_by_f64(&self, factor: f64, up: bool) -> Self {
        let rhs = FixedReal::from_f64(factor, self.shift);
        if up {self.clone() * rhs} else {self.clone() / rhs}
    }

    pub fn double(self) -> Self {
        Self::new(self.mantissa << 1u32, self.shift)
    }

    pub fn sqrt(self) -> Self {
        let shift = self.shift;
        Self::new(
            (self.mantissa << shift).sqrt(),
            shift
        )
    }

    pub fn to_ui_string(&self, digits: usize) -> String {
        format!("{:.*e}", digits, self.to_f64_lossy())
    }

    pub fn to_storage_string(&self) -> String {
        format!("{}@{}", self.mantissa, self.shift)
    }

    pub fn rescale(&mut self, delta_shift: i32) {
        if delta_shift > 0 {
            let s = delta_shift as u32;

            self.mantissa <<= s;
            self.shift += s;
        }
        else {
            let s = (-delta_shift) as u32;

            self.mantissa >>= s;
            self.shift -= s;
        }
    }
}

impl Display for FixedReal {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{} * 2^-{}", self.mantissa, self.shift)
    }
}

impl FromStr for FixedReal {
    type Err = ParseFixedError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (bits, shift) = s
            .split_once('@')
            .ok_or(ParseFixedError::InvalidFormat)?;

        Ok(Self {
            mantissa: Integer::from_str(bits)?,
            shift: shift.parse()?,
        })
    }
}

impl Add for FixedReal {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        debug_assert_eq!(self.shift, rhs.shift);

        Self::new(self.mantissa + rhs.mantissa, self.shift)
    }
}

impl<'a, 'b> Add<&'b FixedReal> for &'a FixedReal {
    type Output = FixedReal;
    fn add(self, rhs: &'b FixedReal) -> FixedReal {
        FixedReal::new(
            Integer::from(&self.mantissa + &rhs.mantissa),
            self.shift,
        )
    }
}

impl AddAssign for FixedReal {
    fn add_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shift, rhs.shift);
        self.mantissa += rhs.mantissa;
    }
}

impl AddAssign<&FixedReal> for FixedReal {
    fn add_assign(&mut self, rhs: &FixedReal) {
        debug_assert_eq!(self.shift, rhs.shift);
        self.mantissa += &rhs.mantissa;
    }
}

impl Sub for FixedReal {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        debug_assert_eq!(self.shift, rhs.shift);

        Self::new(self.mantissa - rhs.mantissa, self.shift)
    }
}

impl<'a, 'b> Sub<&'b FixedReal> for &'a FixedReal {
    type Output = FixedReal;
    fn sub(self, rhs: &'b FixedReal) -> FixedReal {
        FixedReal::new(
            Integer::from(&self.mantissa - &rhs.mantissa),
            self.shift,
        )
    }
}

impl SubAssign for FixedReal {
    fn sub_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shift, rhs.shift);
        self.mantissa -= rhs.mantissa;
    }
}

impl SubAssign<&FixedReal> for FixedReal {
    fn sub_assign(&mut self, rhs: &FixedReal) {
        debug_assert_eq!(self.shift, rhs.shift);
        self.mantissa -= &rhs.mantissa;
    }
}

impl Mul for FixedReal {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        debug_assert_eq!(self.shift, rhs.shift);

        let shift = self.shift;

        Self::new(
            (self.mantissa * rhs.mantissa) >> shift,
            shift,
        )
    }
}

impl<'a, 'b> Mul<&'b FixedReal> for &'a FixedReal {
    type Output = FixedReal;
    fn mul(self, rhs: &'b FixedReal) -> FixedReal {
        FixedReal::new(
            Integer::from(&self.mantissa * &rhs.mantissa) >> self.shift,
            self.shift,
        )
    }
}

impl MulAssign for FixedReal {
    fn mul_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shift, rhs.shift);
        self.mantissa *= rhs.mantissa;
        self.mantissa >>= self.shift
    }
}

impl MulAssign<&FixedReal> for FixedReal {
    fn mul_assign(&mut self, rhs: &FixedReal) {
        debug_assert_eq!(&self.shift, &rhs.shift);
        self.mantissa *= &rhs.mantissa;
        self.mantissa >>= self.shift
    }
}

impl Div for FixedReal {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        debug_assert_eq!(self.shift, rhs.shift);

        let shift = self.shift;

        Self::new(
            (self.mantissa << shift) / rhs.mantissa,
            shift,
        )
    }
}

impl<'a, 'b> Div<&'b FixedReal> for &'a FixedReal {
    type Output = FixedReal;
    fn div(self, rhs: &'b FixedReal) -> FixedReal {
        FixedReal::new(
            Integer::from(&self.mantissa << self.shift) / &rhs.mantissa,
            self.shift,
        )
    }
}

impl DivAssign for FixedReal {
    fn div_assign(&mut self, rhs: Self) {
        debug_assert_eq!(self.shift, rhs.shift);
        self.mantissa <<= self.shift;
        self.mantissa /= rhs.mantissa;
    }
}

impl DivAssign<&FixedReal> for FixedReal {
    fn div_assign(&mut self, rhs: &FixedReal) {
        debug_assert_eq!(&self.shift, &rhs.shift);
        self.mantissa <<= self.shift;
        self.mantissa /= &rhs.mantissa;
    }
}

impl FixedComplex {

    pub fn new(re: FixedReal, im: FixedReal) -> Self {
        debug_assert_eq!(re.shift, im.shift);
        Self { re, im }
    }

    pub fn zero(shift: u32) -> Self {
        Self::new(FixedReal::zero(shift), FixedReal::zero(shift))
    }

    pub fn with_val_f32(val: (f32, f32), shift: u32) -> Self {
        Self {
            re: FixedReal::from_f32(val.0, shift),
            im: FixedReal::from_f32(val.1, shift)
        }
    }

    pub fn with_val_f64(val: (f64, f64), shift: u32) -> Self {
        Self {
            re: FixedReal::from_f64(val.0, shift),
            im: FixedReal::from_f64(val.1, shift)
        }
    }

    pub fn double(self) -> Self {
        Self::new(self.re.double(), self.im.double())
    }

    pub fn square(&self) -> Self {
        let aa = FixedReal::square_ref(&self.re);
        let bb = FixedReal::square_ref(&self.im);
        let ab = &self.re * &self.im;

        Self::new(
            aa - bb,
            ab.double(),
        )
    }

    pub fn norm_sqr(&self) -> FixedReal {
        let re2 = FixedReal::square_ref(&self.re);
        let im2 = FixedReal::square_ref(&self.im);

        re2 + im2
    }

    pub fn norm(&self) -> FixedReal {
        let n2 = self.norm_sqr();
        n2.sqrt()
    }

    pub fn re(&self) -> &FixedReal {
        &self.re
    }

    pub fn im(&self) -> &FixedReal {
        &self.im
    }

    pub fn to_ui_string(&self, digits: usize) -> String {
        format!("({:.*e} + {:.*e}i)",
            digits, self.re.to_f64_lossy(), digits, self.im.to_f64_lossy())
    }

    pub fn rescale(&mut self, delta_shift: i32) {
        self.re.rescale(delta_shift);
        self.im.rescale(delta_shift);
    }
}

impl Display for FixedComplex {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "({} + {}i)", self.re, self.im)
    }
}

impl Add for FixedComplex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<'a, 'b> Add<&'b FixedComplex> for &'a FixedComplex {
    type Output = FixedComplex;
    fn add(self, rhs: &'b FixedComplex) -> FixedComplex {
        FixedComplex::new(
            &self.re + &rhs.re,
            &self.im + &rhs.im
        )
    }
}

impl AddAssign<&FixedComplex> for FixedComplex {
    fn add_assign(&mut self, rhs: &FixedComplex) {
        debug_assert_eq!(&self.re.shift, &rhs.re.shift);
        self.re.mantissa += &rhs.re.mantissa;
        self.im.mantissa += &rhs.im.mantissa;
    }
}

impl Sub for FixedComplex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<'a, 'b> Sub<&'b FixedComplex> for &'a FixedComplex {
    type Output = FixedComplex;
    fn sub(self, rhs: &'b FixedComplex) -> FixedComplex {
        FixedComplex::new(
            &self.re - &rhs.re,
            &self.im - &rhs.im
        )
    }
}

impl Mul for FixedComplex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let re = self.re.clone() * rhs.re.clone() - self.im.clone() * rhs.im.clone();
        let im = self.im.clone() * rhs.re.clone() + self.re.clone() * rhs.im.clone();
        Self::new(re, im)
    }
}

impl<'a, 'b> Mul<&'b FixedComplex> for &'a FixedComplex {
    type Output = FixedComplex;
    fn mul(self, rhs: &'b FixedComplex) -> FixedComplex {
        let re = &self.re * &rhs.re - &self.im * &rhs.im;
        let im = &self.im * &rhs.re + &self.re * &rhs.im;
        FixedComplex::new(re, im)
    }
}

impl Div for FixedComplex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let norm_sqr = rhs.norm_sqr();
        let re = self.re.clone() * self.re.clone() + rhs.im.clone() * rhs.im.clone();
        let im = self.im.clone() * rhs.re.clone() - self.re.clone() * rhs.im.clone();
        Self::new(re / norm_sqr.clone(), im / norm_sqr)
    }
}

impl<'a, 'b> Div<&'b FixedComplex> for &'a FixedComplex {
    type Output = FixedComplex;
    fn div(self, rhs: &'b FixedComplex) -> FixedComplex {
        let norm_sqr = rhs.norm_sqr();
        let re = &self.re * &self.re + &rhs.im * &rhs.im;
        let im = &self.im * &rhs.re - &self.re * &rhs.im;

        FixedComplex::new(re / norm_sqr.clone(), im / norm_sqr)
    }
}

impl Display for ParseFixedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFormat => {
                write!(f, "invalid fixed-point format, expected '<bits>@<shift>'")
            }
            Self::InvalidBits(err) => {
                write!(f, "invalid integer mantissa: {}", err)
            }
            Self::InvalidShift(err) => {
                write!(f, "invalid shift value: {}", err)
            }
        }
    }
}

impl Error for ParseFixedError {}

impl From<ParseIntegerError> for ParseFixedError {
    fn from(err: ParseIntegerError) -> Self {
        Self::InvalidBits(err)
    }
}

impl From<ParseIntError> for ParseFixedError {
    fn from(err: ParseIntError) -> Self {
        Self::InvalidShift(err)
    }
}

// Float-exp: a f32 mantissa paired with a binary exponent (i32).
// Value = mantissa * 2^exp. No normalization invariant is enforced —
// the mantissa is allowed to drift during GPU iteration and is only
// normalized on the CPU side (during conversion) and after multiply
// in the WGSL shader to prevent f32 overflow.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FExp {
    pub m: f32,
    pub e: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComplexFExp {
    pub re: FExp,
    pub im: FExp,
}

impl FExp {
    pub fn zero() -> Self {
        Self { m: 0.0, e: 0 }
    }

    pub fn from_f64(val: f64) -> Self {
        if val == 0.0 {
            return Self::zero();
        }
        // Extract binary exponent from the f64 bit pattern.
        // f64: 1 sign bit, 11 exponent bits (biased by 1023), 52 mantissa bits.
        // Raw biased exponent gives floor(log2(|val|)) for normal values.
        let bits = val.abs().to_bits();
        let raw_exp = ((bits >> 52) & 0x7FF) as i32 - 1023;
        // Set e = raw_exp + 1 so mantissa lands in (-1, -0.5] ∪ [0.5, 1).
        let e = raw_exp + 1;
        let m = (val / 2f64.powi(e)) as f32;
        Self { m, e }
    }

    pub fn from_fixed(x: &FixedReal) -> Self {
        Self::from_f64(x.to_f64_lossy())
    }

    pub fn to_f64(&self) -> f64 {
        (self.m as f64) * 2f64.powi(self.e)
    }
}

impl ComplexFExp {
    pub fn zero() -> Self {
        Self { re: FExp::zero(), im: FExp::zero() }
    }

    pub fn from_fixed(c: &FixedComplex) -> Self {
        Self {
            re: FExp::from_fixed(c.re()),
            im: FExp::from_fixed(c.im()),
        }
    }
}

// -----------------------------------------------------------------------------
// FExp / ComplexFExp arithmetic — the CPU mirror of mandelbrot_fexp.wgsl.
//
// Invariant (same as the shader): after every op the mantissa is renormalized
// into [0.5, 1) via frexp, with a true zero (m == 0) handled explicitly. The
// stored mantissa is f32 to match the GPU representation; intermediate math
// promotes to f64 and rounds back on normalize. This is what keeps a deeply
// merged BLA coefficient (the orbit derivative, which over/underflows plain f64)
// representable at any scale.
// -----------------------------------------------------------------------------

/// Split x into (m, e) with x == m * 2^e and m in [0.5, 1) (or x itself for 0/inf/nan).
fn frexp_f64(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    if exp_field == 0 {
        // Subnormal: scale into normal range (x * 2^64), then correct the exponent.
        let (m, e) = frexp_f64(x * 18446744073709551616.0);
        return (m, e - 64);
    }
    let e = exp_field - 1022; // value = m * 2^e with m in [0.5, 1)
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, e)
}

impl FExp {
    /// The value 1.0 in canonical form (0.5 * 2^1).
    pub fn one() -> Self {
        Self { m: 0.5, e: 1 }
    }

    /// Normalize a (value * 2^exp) pair into a canonical FExp.
    fn from_parts(value: f64, exp: i32) -> Self {
        let (m, e) = frexp_f64(value);
        Self { m: m as f32, e: exp + e }
    }

    pub fn abs(self) -> Self {
        Self { m: self.m.abs(), e: self.e }
    }

    pub fn is_zero(self) -> bool {
        self.m == 0.0
    }

    /// Square root for non-negative values.
    pub fn sqrt(self) -> Self {
        if self.m <= 0.0 {
            return Self::zero();
        }
        // Halve the exponent; if it's odd, fold one factor of 2 into the mantissa
        // first so the remaining exponent halves exactly.
        let (m, e) = if self.e & 1 == 0 {
            (self.m as f64, self.e >> 1)
        } else {
            (self.m as f64 * 2.0, (self.e - 1) >> 1)
        };
        Self::from_parts(m.sqrt(), e)
    }

    /// Strict less-than for non-negative operands (mantissas >= 0). Valid because
    /// both operands are renormalized, so the exponent is the primary key.
    pub fn lt_pos(self, other: Self) -> bool {
        if self.m == 0.0 {
            return other.m != 0.0;
        }
        if other.m == 0.0 {
            return false;
        }
        if self.e != other.e {
            return self.e < other.e;
        }
        self.m < other.m
    }

    pub fn min_pos(self, other: Self) -> Self {
        if self.lt_pos(other) { self } else { other }
    }
}

impl Mul for FExp {
    type Output = FExp;
    fn mul(self, rhs: FExp) -> FExp {
        FExp::from_parts(self.m as f64 * rhs.m as f64, self.e + rhs.e)
    }
}

impl Add for FExp {
    type Output = FExp;
    fn add(self, rhs: FExp) -> FExp {
        // A true zero carries no meaningful exponent, so let the other operand win.
        if self.m == 0.0 {
            return rhs;
        }
        if rhs.m == 0.0 {
            return self;
        }
        // Align to the larger exponent; the shift exponent is always <= 0, so
        // powi never overflows (it underflows to 0 when an operand is negligible).
        let (sum, e) = if self.e >= rhs.e {
            (self.m as f64 + rhs.m as f64 * 2f64.powi(rhs.e - self.e), self.e)
        } else {
            (self.m as f64 * 2f64.powi(self.e - rhs.e) + rhs.m as f64, rhs.e)
        };
        FExp::from_parts(sum, e)
    }
}

impl Sub for FExp {
    type Output = FExp;
    fn sub(self, rhs: FExp) -> FExp {
        self + FExp { m: -rhs.m, e: rhs.e }
    }
}

impl Div for FExp {
    type Output = FExp;
    fn div(self, rhs: FExp) -> FExp {
        FExp::from_parts(self.m as f64 / rhs.m as f64, self.e - rhs.e)
    }
}

impl ComplexFExp {
    /// The value 1 + 0i.
    pub fn one() -> Self {
        Self { re: FExp::one(), im: FExp::zero() }
    }

    pub fn from_f64_pair(re: f64, im: f64) -> Self {
        Self { re: FExp::from_f64(re), im: FExp::from_f64(im) }
    }

    /// |z|^2 as FExp — correct at any scale (no squaring through f64 range).
    pub fn mag2(self) -> FExp {
        self.re * self.re + self.im * self.im
    }

    /// |z| as FExp.
    pub fn mag(self) -> FExp {
        self.mag2().sqrt()
    }

    pub fn double(self) -> Self {
        Self { re: self.re + self.re, im: self.im + self.im }
    }
}

impl Add for ComplexFExp {
    type Output = ComplexFExp;
    fn add(self, rhs: ComplexFExp) -> ComplexFExp {
        ComplexFExp { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for ComplexFExp {
    type Output = ComplexFExp;
    fn sub(self, rhs: ComplexFExp) -> ComplexFExp {
        ComplexFExp { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl Mul for ComplexFExp {
    type Output = ComplexFExp;
    fn mul(self, rhs: ComplexFExp) -> ComplexFExp {
        ComplexFExp {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIFT: u32 = 64;

    fn fr(v: f64) -> FixedReal {
        FixedReal::from_f64(v, SHIFT)
    }

    // f64 -> FixedReal -> f64 must round-trip for values exactly representable
    // at this shift.
    #[test]
    fn f64_round_trip() {
        for v in [0.0, 1.0, -1.0, 1.5, -2.25, 0.125, 1234.5, -0.0009765625] {
            assert_eq!(fr(v).to_f64_lossy(), v, "round trip failed for {v}");
        }
    }

    // square_ref routes through GMP's squaring fast-path; result must match the
    // mathematical square and stay at the same shift.
    #[test]
    fn square_ref_matches() {
        let x = fr(1.5);
        let sq = FixedReal::square_ref(&x);
        assert_eq!(sq.shift, SHIFT);
        assert_eq!(sq.to_f64_lossy(), 2.25);
        // Sign is irrelevant to a square.
        assert_eq!(FixedReal::square_ref(&fr(-1.5)).to_f64_lossy(), 2.25);
    }

    // Multiply (both owned and by-ref paths) keeps the fixed point after the
    // shift-down, including the sign.
    #[test]
    fn mul_keeps_fixed_point() {
        assert_eq!((fr(1.5) * fr(2.0)).to_f64_lossy(), 3.0);
        assert_eq!((&fr(-1.5) * &fr(2.0)).to_f64_lossy(), -3.0);
        assert_eq!((&fr(0.5) * &fr(0.5)).to_f64_lossy(), 0.25);
    }

    #[test]
    fn add_sub_owned_and_ref() {
        assert_eq!((fr(1.5) + fr(2.0)).to_f64_lossy(), 3.5);
        assert_eq!((&fr(1.5) + &fr(2.0)).to_f64_lossy(), 3.5);
        assert_eq!((fr(1.5) - fr(2.0)).to_f64_lossy(), -0.5);
        assert_eq!((&fr(1.5) - &fr(2.0)).to_f64_lossy(), -0.5);
    }

    #[test]
    fn div_keeps_fixed_point() {
        assert_eq!((fr(3.0) / fr(2.0)).to_f64_lossy(), 1.5);
        assert_eq!((&fr(-3.0) / &fr(2.0)).to_f64_lossy(), -1.5);
    }

    #[test]
    fn sqrt_floor() {
        assert_eq!(fr(4.0).sqrt().to_f64_lossy(), 2.0);
        let two = fr(2.0).sqrt().to_f64_lossy();
        assert!((two - std::f64::consts::SQRT_2).abs() < 1e-15, "got {two}");
    }

    #[test]
    fn is_zero_and_bits() {
        assert!(FixedReal::zero(SHIFT).is_zero());
        assert!(!fr(1.0).is_zero());
        // 1.0 at shift 64 is 2^64, whose magnitude needs 65 significant bits.
        assert_eq!(fr(1.0).bits(), 65);
        assert_eq!(FixedReal::zero(SHIFT).bits(), 0);
    }

    // Storage string -> parse must round-trip the exact mantissa and shift.
    #[test]
    fn storage_string_round_trip() {
        let x = fr(-2.25);
        let parsed: FixedReal = x.to_storage_string().parse().unwrap();
        assert_eq!(parsed.mantissa, x.mantissa);
        assert_eq!(parsed.shift, x.shift);
    }

    // FixedComplex squaring uses square_ref + the cross term; verify against the
    // closed form (a+bi)^2 = (a^2 - b^2) + 2ab i.
    #[test]
    fn complex_square() {
        let z = FixedComplex::with_val_f64((1.5, -2.0), SHIFT);
        let sq = z.square();
        assert_eq!(sq.re().to_f64_lossy(), 1.5 * 1.5 - 2.0 * 2.0); // -1.75
        assert_eq!(sq.im().to_f64_lossy(), 2.0 * 1.5 * -2.0); // -6.0
    }

    // ---- FExp / ComplexFExp arithmetic ----

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn fexp_mul_add_sub_div() {
        let three = FExp::from_f64(3.0);
        let five = FExp::from_f64(5.0);
        assert!(approx((three * five).to_f64(), 15.0, 1e-6));
        assert!(approx((three + five).to_f64(), 8.0, 1e-6));
        assert!(approx((three - five).to_f64(), -2.0, 1e-6));
        assert!(approx((FExp::from_f64(7.0) / FExp::from_f64(2.0)).to_f64(), 3.5, 1e-6));
        // Exact cancellation yields a canonical zero.
        assert!((three - three).is_zero());
    }

    #[test]
    fn fexp_add_disparate_exponents() {
        // 1 + 2^-100: the tiny term vanishes into f32 mantissa but must not corrupt
        // the result or the exponent.
        let big = FExp::from_f64(1.0);
        let tiny = FExp { m: 0.5, e: -99 }; // 2^-100
        let sum = big + tiny;
        assert!(approx(sum.to_f64(), 1.0, 1e-6));
        // The tiny value on its own keeps its deep exponent.
        assert_eq!(tiny.e, -99);
    }

    #[test]
    fn fexp_survives_deep_exponents() {
        // 1e-300 squared underflows f64 (~1e-600) but FExp keeps it: mantissa
        // finite & non-zero, exponent far below f64's floor.
        let small = FExp::from_f64(1e-300);
        let sq = small * small;
        assert!(sq.m != 0.0 && sq.m.is_finite());
        assert!(sq.e < -1900, "exponent should be ~ -1993, got {}", sq.e);
        // And it normalizes back: sqrt(1e-300^2) == 1e-300.
        assert!(approx(sq.sqrt().to_f64(), 1e-300, 1e-5));
    }

    #[test]
    fn fexp_sqrt_and_compare() {
        assert!(approx(FExp::from_f64(16.0).sqrt().to_f64(), 4.0, 1e-6));
        assert!(approx(FExp::from_f64(2.0).sqrt().to_f64(), std::f64::consts::SQRT_2, 1e-6));
        assert!(FExp::from_f64(3.0).lt_pos(FExp::from_f64(5.0)));
        assert!(!FExp::from_f64(5.0).lt_pos(FExp::from_f64(3.0)));
        assert_eq!(FExp::from_f64(3.0).min_pos(FExp::from_f64(5.0)).to_f64(), 3.0);
    }

    #[test]
    fn cfexp_mul_and_mag() {
        // (1 + 2i)(3 + 4i) = -5 + 10i
        let a = ComplexFExp::from_f64_pair(1.0, 2.0);
        let b = ComplexFExp::from_f64_pair(3.0, 4.0);
        let p = a * b;
        assert!(approx(p.re.to_f64(), -5.0, 1e-6));
        assert!(approx(p.im.to_f64(), 10.0, 1e-6));
        // |3 + 4i| = 5
        assert!(approx(ComplexFExp::from_f64_pair(3.0, 4.0).mag().to_f64(), 5.0, 1e-6));
        // double
        let d = ComplexFExp::from_f64_pair(1.5, -2.0).double();
        assert!(approx(d.re.to_f64(), 3.0, 1e-6));
        assert!(approx(d.im.to_f64(), -4.0, 1e-6));
    }
}
