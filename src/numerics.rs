use std::error::Error;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::num::ParseIntError;
use bytemuck;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::str::FromStr;
use num_bigint::{BigInt, ParseBigIntError};
use num_traits::{FromPrimitive, Signed, ToPrimitive};

pub const DEFAULT_BAILOUT: f64 = 4.0;
pub const MAX_SAFE_DF_MAG: f64 = 1e30;

/// Wraps a BigInt and a u32 to represent a fixed-point value.
/// The final number is represented as mantissa * 2^shift, where
/// mantissa is a BigInt - which is internally a Vec<u32> - and
/// shift is a u32. Math operations on this type are kept minimal,
/// and the shift never changes. Instead, the shift is only updated
/// on zoom operations and on consideration of the viewport scale.
/// From the perspective of this class, shift should never be
/// modified once the object is allocated.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FixedReal {
    pub mantissa: BigInt,
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
    InvalidBits(ParseBigIntError),
    InvalidShift(ParseIntError),
}

impl FixedReal {
    pub fn new(bits: BigInt, shift: u32) -> Self {
        Self { mantissa: bits, shift }
    }

    pub fn zero(shift: u32) -> Self {
        Self::new(BigInt::ZERO, shift)
    }

    pub fn from_f32(float: f32, shift: u32) -> Self {
        Self::from_f64(float as f64, shift)
    }

    pub fn from_f64(float: f64, shift: u32) -> Self {
        let scaled = float * 2.0_f64.powi(shift as i32);

        Self::new(
            BigInt::from_f64(scaled.round()).unwrap(),
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

            if bits != BigInt::ZERO {
                shift = MAX_SHIFT;
            } else {
                bits = original_bits;
            }
        }

        let value = bits.to_f64().unwrap_or_else(|| {
            if bits.sign() == num_bigint::Sign::Minus {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
        });

        value / 2.0_f64.powi(shift as i32)
    }

    pub fn to_f32_lossy(&self) -> f32 {
        const TARGET_BITS: u64 = 30;

        let mut bits = self.mantissa.clone();
        let mut shift = self.shift;

        let bit_count = bits.bits();

        if bit_count > TARGET_BITS {
            let reduction = bit_count - TARGET_BITS;

            bits >>= reduction;
            shift -= reduction as u32;
        }

        let numerator = bits.to_f64().unwrap_or(0.0);

        (numerator / 2.0_f64.powi(shift as i32)) as f32
    }
    
    pub fn abs(&self) -> Self {
        Self::new(self.mantissa.abs(), self.shift)
    }

    pub fn square_ref(x: &Self) -> Self {
        Self::new(
            (&x.mantissa * &x.mantissa) >> x.shift,
            x.shift
        )
    }

    pub fn scale_by_f64(&self, factor: f64, up: bool) -> Self {
        let rhs = FixedReal::from_f64(factor, self.shift);
        if up {self.clone() * rhs} else {self.clone() / rhs}
    }

    pub fn double(self) -> Self {
        Self::new(self.mantissa << 1, self.shift)
    }

    pub fn sqrt(self) -> Self {
        Self::new(
            (&self.mantissa << self.shift).sqrt(),
            self.shift
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
            mantissa: BigInt::from_str(bits)?,
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
            &self.mantissa + &rhs.mantissa,
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
            &self.mantissa - &rhs.mantissa,
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
            (&self.mantissa * &rhs.mantissa) >> self.shift,
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
            (&self.mantissa << self.shift) / &rhs.mantissa,
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
                write!(f, "invalid BigInt bits: {}", err)
            }
            Self::InvalidShift(err) => {
                write!(f, "invalid shift value: {}", err)
            }
        }
    }
}

impl Error for ParseFixedError {}

impl From<ParseBigIntError> for ParseFixedError {
    fn from(err: ParseBigIntError) -> Self {
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
