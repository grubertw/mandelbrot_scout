use bytemuck;
use std::ops::{Add, Mul};
use rug::{Float, Complex};

pub const DEFAULT_BAILOUT: f64 = 4.0;
pub const MAX_SAFE_DF_MAG: f64 = 1e30;

// Double-float, which is our 'bypass' of WGSL's lack of f64
// On that note, using two floats in this way is far more robust 
// accross a wider set of GPUs. While not giving 53 bits of precision,
// this can theoreticly give us up to 48 bits - i.e. 24+24 as f32 
// has 24 bits - though in practice it will likely yeild only 40-44.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Df {
    pub hi: f32,
    pub lo: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComplexDf {
    pub re: Df,
    pub im: Df,
}

impl Df {
    pub fn new(hi: f32, lo: f32) -> Self {
        Self { hi, lo }
    }

    // Convert a Rug Float to double-float
    // First rounds the arbitary precision to fit into f64, and then seeks to 
    // preserve what's lost with the initial rounding, and perserves that in 
    // another f64. Ultilate, these high and low f64s are trunkated again 
    // before finally being returns as a Df. This strategy tries to preserve
    // 'the most meaningful' significant digets at the beginning and end of
    // the arbitraty precision value.
    pub fn from_float(x: &Float) -> Self {
        let hi = x.to_f64();

        // residual = x - hi
        let mut residual = Float::with_val(x.prec(), x);
        residual -= hi;

        let lo = residual.to_f64();
        
        Self {hi: hi as f32, lo: lo as f32}
    }

    pub fn to_float(&self, prec: u32) -> Float {
        let mut res = Float::with_val(prec, self.hi);
        res += self.lo;
        res
    }
}

impl Add for Df {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        let s = self.hi + rhs.hi;
        let e = (self.hi - s) + rhs.hi + self.lo + rhs.lo;
        Self::new(s, e)
    }
}

impl Mul for Df {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let p = self.hi * rhs.hi;
        let e = self.hi * rhs.lo + self.lo * rhs.hi;
        Self::new(p, e)
    }
}

impl ComplexDf {
    pub fn new(re: Df, im: Df) -> Self {
        Self { re, im }
    }

    pub fn from_complex(c: &Complex) -> Self {
        let real_df = Df::from_float(c.real());
        let imag_df = Df::from_float(c.imag());

        Self {re: real_df, im: imag_df}
    }
    
    pub fn to_complex(&self, prec: u32) -> Complex {
        let mut res = Complex::with_val(prec, (self.re.hi, self.im.hi));
        let (real, imag) = res.as_mut_real_imag();
        *real += self.re.lo;
        *imag += self.im.lo;
        res
    }
}
