//! Bivariate Linear Approximation (BLA) for the perturbed Mandelbrot iteration.
//!
//! The perturbation step `dz' = 2*Z*dz + dz^2 + dc` is linear in `dz` except for
//! the `dz^2` term. While `|dz|` is small relative to `|Z|`, that term is
//! negligible and the step is approximately `dz' = A*dz + B*dc` with `A = 2Z`,
//! `B = 1` — a linear map. Linear maps compose, so a run of `l` such steps
//! collapses into a single `A*dz + B*dc` (see `merge`). A binary merge tree over
//! the reference orbit lets a pixel skip the largest valid run in one step.
//!
//! Two halves, deliberately separated by their dependence on the view:
//!   * `BlaTable` (A, B, l) depends only on the reference orbit `Z` — built once
//!     per anchored orbit, immutable thereafter.
//!   * `BlaRadii` (the validity `r^2` per entry) depends on `delta_c_max` (the
//!     max |dc| over the image) — recomputed from the table when the view moves,
//!     cheaply (no complex multiplies), without rebuilding the tree.
//!
//! Mandelbrot is holomorphic, so `A`/`B` are complex scalars (`ComplexFExp`),
//! not the 2x2 real matrices a general (e.g. Burning Ship) formula would need.
//! Everything is `FExp` so a deeply merged `A` (the orbit derivative, which
//! over/underflows plain f64) stays representable at any zoom.

use std::sync::Arc;
use std::thread::available_parallelism;

use futures::executor::ThreadPool;
use futures::future::join_all;
use futures::task::SpawnExt;

use crate::numerics::{ComplexFExp, FExp, FixedComplex, FixedReal};
use crate::scout_engine::orbit::{OrbitId, WeakOrbit};
// The GPU upload contract lives with the other GPU structs; we build instances of
// it directly so there's a single source of truth for the entry layout.
pub use crate::gpu_pipeline::structs::BlaEntry;

/// BLA epsilon: the relative tolerance on the dropped `dz^2` term. 2^-24 matches
/// the f32 mantissa (and Fraktaler-3's default). Smaller => safer but shorter
/// jumps; larger => longer jumps but visible error.
pub const BLA_EPSILON: f32 = 1.0 / (1u32 << 24) as f32;

/// Below this many entries in a level, build it serially — the per-task spawn
/// overhead isn't worth it for the small upper levels of the tree.
const PAR_CUTOFF: usize = 2048;

/// The view-independent merge tree. `levels[0]` holds one leaf per reference step
/// `m` in `1..=M` (M = orbit length - 1); `levels[k][dst]` merges two adjacent
/// entries from `levels[k-1]`, covering `2^k` steps starting at `m = (dst<<k)+1`.
#[derive(Clone, Debug, Default)]
pub struct BlaTable {
    pub levels: Vec<Vec<BlaEntry>>,
    /// Number of reference steps the table spans (orbit length - 1).
    pub m: usize,
    /// |Z_m|^2 per leaf (view-independent), one entry per `levels[0]` leaf. The
    /// leaf validity radius is eps*|Z|/(power-1), which can't be recovered from
    /// the leaf `A = power*Z^(power-1)` without a (power-1)-th root, so we keep
    /// |Z|^2 here for `compute_radii`. Not uploaded to the GPU (radii are).
    pub leaf_z2: Vec<FExp>,
}

/// The view-dependent validity radii, shaped exactly like `BlaTable.levels`:
/// `radii.levels[k][dst]` is the squared radius `r^2` of `table.levels[k][dst]`.
#[derive(Clone, Debug, Default)]
pub struct BlaRadii {
    pub levels: Vec<Vec<FExp>>,
}

/// Binds a built BLA table to the orbit it describes, plus the view-dependent
/// state. The `table` (A/B/l) is immutable and shared; `radii` and `delta_c_max`
/// are refreshed together when the view moves. `orbit_id`/`built_len` are the
/// staleness key (checked without upgrading the weak ref); `orbit_ref` lets a
/// radii refresh re-read `c_ref` without touching the orbit pool.
#[derive(Clone)]
pub struct QualifiedOrbitBLAInfo {
    pub orbit_id: OrbitId,
    pub built_len: u32,
    /// The formula power the table was built with; the renderer feeds it back to
    /// `compute_radii` so the leaf radii stay consistent with the leaf `A`.
    pub power: u32,
    pub orbit_ref: WeakOrbit,
    // Only the view-INDEPENDENT table (A/B/l), wrapped so a reader can clone the
    // whole struct cheaply. The validity radii are view-dependent (they scale with
    // delta_c_max), so the renderer computes and uploads them itself for the exact
    // view being drawn — see scene::refresh_bla. Keeping them out of here avoids a
    // stale-radii hazard when the same orbit persists across a pan/zoom.
    pub table: Arc<BlaTable>,
}

// Summarize rather than dump — the table can hold millions of entries, so a
// derived Debug could explode a single log line.
impl std::fmt::Debug for QualifiedOrbitBLAInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QualifiedOrbitBLAInfo")
            .field("orbit_id", &self.orbit_id)
            .field("built_len", &self.built_len)
            .field("steps", &self.table.m)
            .field("levels", &self.table.levels.len())
            .finish()
    }
}

/// Conservative upper bound on `|delta_c|` over the image: the center offset
/// (`center - c_ref`, the value the GPU already uses per orbit) plus the view
/// half-extent (which upper-bounds the center-to-corner pixel offset). Sizes the
/// BLA merge radii; overestimating is safe (shorter but always-valid jumps).
/// Taking the precomputed offset (rather than center & c_ref) keeps this in lock
/// step with what the renderer uploads and sidesteps any shift mismatch.
pub fn delta_c_max_bound(center_offset: &FixedComplex, half_extent: &FixedReal) -> FExp {
    ComplexFExp::from_fixed(center_offset).mag() + FExp::from_fixed(half_extent)
}

/// A single-step leaf at reference point `z = Z_m` for f(z) = z^power + c:
/// `A = f'(Z) = power * Z^(power-1)`, `B = 1`, `l = 1`. Power 2 keeps the exact
/// `2Z` form (byte-identical to the original power-2 path).
#[inline]
fn make_leaf(z: ComplexFExp, power: u32) -> BlaEntry {
    let a = if power == 2 {
        z.double() // 2Z
    } else {
        // power * Z^(power-1)
        let mut zpow = z; // Z^1
        for _ in 2..power { zpow = zpow * z; } // -> Z^(power-1)
        zpow * ComplexFExp::from_f64_pair(power as f64, 0.0)
    };
    BlaEntry { a, b: ComplexFExp::one(), l: 1 }
}

/// Compose `x` (the earlier block) then `y` (the later block) into one BLA:
///   dz_out = A_y(A_x dz + B_x c) + B_y c = (A_y A_x) dz + (A_y B_x + B_y) c
#[inline]
fn merge(x: &BlaEntry, y: &BlaEntry) -> BlaEntry {
    BlaEntry {
        a: y.a * x.a,
        b: y.a * x.b + y.b,
        l: x.l + y.l,
    }
}

/// Merge children `2*dst` and `2*dst+1` from `prev`, or carry the left child
/// alone when the right one is past the end (ragged right edge of the tree).
#[inline]
fn merge_or_carry(prev: &[BlaEntry], dst: usize) -> BlaEntry {
    let xi = dst * 2;
    let yi = xi + 1;
    if yi < prev.len() {
        merge(&prev[xi], &prev[yi])
    } else {
        prev[xi]
    }
}

/// Map `f` over `0..n` across the thread pool, chunked into ~one task per core.
/// The pool requires `'static` futures, so any shared input must be captured by
/// `Arc` (the caller clones it into `f`); results are concatenated in order.
async fn parallel_map<T, F>(pool: &ThreadPool, n: usize, f: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(usize) -> T + Send + Sync + Clone + 'static,
{
    if n == 0 {
        return Vec::new();
    }
    let workers = available_parallelism().map(|p| p.get()).unwrap_or(4).max(1);
    let chunk = n.div_ceil(workers);

    let mut handles = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + chunk).min(n);
        let f = f.clone();
        let handle = pool
            .spawn_with_handle(async move { (start..end).map(|i| f(i)).collect::<Vec<T>>() })
            .expect("ScoutEngine ThreadPool failed to spawn a BLA build chunk");
        handles.push(handle);
        start = end;
    }

    let mut out = Vec::with_capacity(n);
    for part in join_all(handles).await {
        out.extend(part);
    }
    out
}

/// Build the view-independent BLA tree from a truncated reference orbit.
/// `orbit[0]` is the critical point (0) and is skipped; leaf `dst` uses
/// `orbit[dst+1]`. Each level is computed in parallel after its child level is
/// complete (a barrier between levels; ~log2(M) of them).
pub async fn build_bla(orbit: Arc<Vec<ComplexFExp>>, power: u32, pool: ThreadPool) -> BlaTable {
    let m = orbit.len().saturating_sub(1);
    if m == 0 {
        return BlaTable::default();
    }

    let leaves = parallel_map(&pool, m, {
        let orbit = orbit.clone();
        move |dst| make_leaf(orbit[dst + 1], power)
    })
    .await;

    // |Z_m|^2 per leaf — view-independent, needed by compute_radii for the leaf
    // radius. Cheap (a magnitude each), so just gather serially.
    let leaf_z2: Vec<FExp> = (0..m).map(|dst| orbit[dst + 1].mag2()).collect();

    let mut levels: Vec<Arc<Vec<BlaEntry>>> = vec![Arc::new(leaves)];
    while levels.last().unwrap().len() > 1 {
        let prev = levels.last().unwrap().clone();
        let size = prev.len().div_ceil(2);
        let next: Vec<BlaEntry> = if size >= PAR_CUTOFF {
            parallel_map(&pool, size, {
                let prev = prev.clone();
                move |dst| merge_or_carry(&prev, dst)
            })
            .await
        } else {
            (0..size).map(|dst| merge_or_carry(&prev, dst)).collect()
        };
        levels.push(Arc::new(next));
    }

    BlaTable {
        levels: levels
            .into_iter()
            .map(|lvl| Arc::try_unwrap(lvl).unwrap_or_else(|a| (*a).clone()))
            .collect(),
        m,
        leaf_z2,
    }
}

/// Compute the validity radii for `table` against a given `delta_c_max` (the max
/// |dc| over the image). Cheap: only magnitudes/mins, no complex multiplies — so
/// it can be re-run when the view moves while the tree itself is reused.
///
/// Leaf radius (view-independent) for f(z)=z^power: the linear step
/// `dz' = A*dz` breaks when the leading nonlinear term C(p,2) Z^(p-2) dz^2
/// overtakes it, giving `r = epsilon * |Z| / (power - 1)`. For power 2 this is
/// `epsilon * |Z|` (= the original `epsilon*|A|/2` since A=2Z). |Z| comes from
/// `table.leaf_z2` — it is NOT `|A|/2` once A = power*Z^(power-1). The
/// `delta_c_max` dependence enters only when merging:
///   r = min( r_x, max(0, (r_y - |B_x|*c) / |A_x|) )
/// i.e. `dz` must stay inside `x`'s radius *and* land inside `y`'s after `x`.
pub fn compute_radii(table: &BlaTable, epsilon: f32, power: u32, delta_c_max: FExp) -> BlaRadii {
    let mut levels: Vec<Vec<FExp>> = Vec::with_capacity(table.levels.len());
    if table.levels.is_empty() {
        return BlaRadii { levels };
    }

    // r = (epsilon/(power-1)) * |Z|  =>  r^2 = k^2 * |Z|^2.
    let k = FExp::from_f64(epsilon as f64 / (power.max(2) - 1) as f64);
    let k2 = k * k;
    let leaf_r2: Vec<FExp> = table.leaf_z2.iter().map(|z2| k2 * *z2).collect();
    levels.push(leaf_r2);

    for k in 1..table.levels.len() {
        let prev_tab = &table.levels[k - 1];
        let size = table.levels[k].len();
        let mut lvl = Vec::with_capacity(size);
        for dst in 0..size {
            let xi = dst * 2;
            let yi = xi + 1;
            if yi < prev_tab.len() {
                let x = &prev_tab[xi];
                let rx = levels[k - 1][xi].sqrt();
                let ry = levels[k - 1][yi].sqrt();
                let sup_ax = x.a.mag();
                let sup_bx = x.b.mag();
                let num = ry - sup_bx * delta_c_max;
                // max(0, num/|A_x|); if |A_x| is zero the x-block is constant, so
                // its own radius governs.
                let t = if num.m <= 0.0 || sup_ax.is_zero() {
                    FExp::zero()
                } else {
                    num / sup_ax
                };
                let r = rx.min_pos(t);
                lvl.push(r * r);
            } else {
                lvl.push(levels[k - 1][xi]); // carry
            }
        }
        levels.push(lvl);
    }

    BlaRadii { levels }
}

/// Find the largest valid BLA step starting at reference index `m` (1-based)
/// given the current `|dz|^2`. Climbs levels while the entry stays aligned to `m`
/// and `z2` is inside its radius; returns `(level, index)` into the table/radii,
/// or `None` to fall back to a single perturbation step. Mirrors F3's lookup and
/// is the reference for the eventual GPU port.
pub fn lookup(table: &BlaTable, radii: &BlaRadii, m: usize, z2: FExp) -> Option<(usize, usize)> {
    if m == 0 || m >= table.m {
        return None;
    }
    let mut ix = m - 1;
    let mut found = None;
    for level in 0..table.levels.len() {
        if ix >= table.levels[level].len() {
            break;
        }
        let ixm = (ix << level) + 1;
        if m == ixm && z2.lt_pos(radii.levels[level][ix]) {
            found = Some((level, ix));
        } else {
            break;
        }
        ix >>= 1;
    }
    found
}

impl BlaTable {
    /// Flatten the level-of-levels into one contiguous `entries` array (the GPU
    /// upload form) plus a `dims` header the shader uses to index it:
    ///   dims = [level_count, steps, off_0=0, off_1, ..., off_L = total]
    /// so `offset(k) = dims[2+k]` and `count(k) = dims[3+k] - dims[2+k]`.
    /// The radii (see [`BlaRadii::flatten`]) share this exact layout/offsets.
    pub fn flatten(&self) -> (Vec<BlaEntry>, Vec<u32>) {
        let total: usize = self.levels.iter().map(|lvl| lvl.len()).sum();
        let mut entries = Vec::with_capacity(total);
        let mut dims = Vec::with_capacity(self.levels.len() + 3);
        dims.push(self.levels.len() as u32); // level_count
        dims.push(self.m as u32); // steps (M)
        let mut off = 0u32;
        for lvl in &self.levels {
            dims.push(off);
            entries.extend_from_slice(lvl);
            off += lvl.len() as u32;
        }
        dims.push(off); // sentinel total = off_L
        (entries, dims)
    }
}

impl BlaRadii {
    /// Flatten radii in the same level order as [`BlaTable::flatten`], so a flat
    /// index from the table's `dims` selects the matching `r^2`.
    pub fn flatten(&self) -> Vec<FExp> {
        let total: usize = self.levels.iter().map(|lvl| lvl.len()).sum();
        let mut out = Vec::with_capacity(total);
        for lvl in &self.levels {
            out.extend_from_slice(lvl);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::executor::block_on;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    /// A real bounded Mandelbrot reference orbit: Z[0]=0, Z[1]=c, ...
    fn ref_orbit(cre: f64, cim: f64, n: usize) -> Vec<ComplexFExp> {
        let mut v = Vec::with_capacity(n);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for _ in 0..n {
            v.push(ComplexFExp::from_f64_pair(re, im));
            let (r2, i2) = (re * re - im * im + cre, 2.0 * re * im + cim);
            re = r2;
            im = i2;
        }
        v
    }

    fn pool() -> ThreadPool {
        ThreadPool::new().expect("test thread pool")
    }

    /// Core build/merge correctness: a fully-merged top entry must reproduce the
    /// LINEAR recurrence dz' = 2*Z*dz + dc exactly (decoupled from the dz^2
    /// approximation error). M = 16 is a power of two, so the top level is a
    /// single entry spanning all 16 steps from m=1.
    #[test]
    fn merge_reproduces_linear_recurrence() {
        let n = 17; // M = 16
        let orbit = Arc::new(ref_orbit(-0.75, 0.06, n));
        let table = block_on(build_bla(orbit.clone(), 2, pool()));

        let dz0 = ComplexFExp::from_f64_pair(0.5, 0.3);
        let dc = ComplexFExp::from_f64_pair(0.1, -0.2);

        // Direct linear propagation over steps m = 1..=16.
        let mut dz = dz0;
        for i in 1..n {
            dz = orbit[i].double() * dz + dc;
        }

        // Single BLA jump via the top entry.
        let top = table.levels.last().unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].l, 16);
        let dz_bla = top[0].a * dz0 + top[0].b * dc;

        assert!(approx(dz_bla.re.to_f64(), dz.re.to_f64(), 1e-3),
            "re: bla={} direct={}", dz_bla.re.to_f64(), dz.re.to_f64());
        assert!(approx(dz_bla.im.to_f64(), dz.im.to_f64(), 1e-3),
            "im: bla={} direct={}", dz_bla.im.to_f64(), dz.im.to_f64());
    }

    /// Power-3: leaf A = 3*Z^2, the merged top reproduces the LINEAR recurrence
    /// dz' = 3*Z^2*dz + dc, and the leaf radius uses the eps/(p-1) factor.
    #[test]
    fn power3_leaf_merge_and_radius() {
        // Reference orbit for z^3 + c (bounded near the origin).
        let n = 17; // M = 16
        let (cre, cim) = (0.1_f64, 0.05_f64);
        let mut v = Vec::with_capacity(n);
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for _ in 0..n {
            v.push(ComplexFExp::from_f64_pair(re, im));
            let (r2, i2) = (re * re - im * im, 2.0 * re * im); // z^2
            let (r3, i3) = (r2 * re - i2 * im, r2 * im + i2 * re); // z^3
            re = r3 + cre;
            im = i3 + cim;
        }
        let orbit = Arc::new(v);
        let table = block_on(build_bla(orbit.clone(), 3, pool()));

        let three = ComplexFExp::from_f64_pair(3.0, 0.0);

        // Leaf A for m=1 must be 3*Z[1]^2.
        let expect_a = (orbit[1] * orbit[1]) * three;
        let leaf_a = table.levels[0][0].a;
        assert!(approx(leaf_a.re.to_f64(), expect_a.re.to_f64(), 1e-9));
        assert!(approx(leaf_a.im.to_f64(), expect_a.im.to_f64(), 1e-9));

        // Merged top == direct linear propagation dz' = 3*Z^2*dz + dc.
        let dz0 = ComplexFExp::from_f64_pair(0.02, -0.01);
        let dc = ComplexFExp::from_f64_pair(0.003, 0.004);
        let mut dz = dz0;
        for i in 1..n {
            dz = ((orbit[i] * orbit[i]) * three) * dz + dc;
        }
        let top = table.levels.last().unwrap();
        assert_eq!(top[0].l, 16);
        let dz_bla = top[0].a * dz0 + top[0].b * dc;
        assert!(approx(dz_bla.re.to_f64(), dz.re.to_f64(), 1e-3));
        assert!(approx(dz_bla.im.to_f64(), dz.im.to_f64(), 1e-3));

        // Leaf radius: r^2 = (eps/(p-1))^2 * |Z1|^2, p=3 => (eps/2)^2 |Z1|^2.
        let radii = compute_radii(&table, BLA_EPSILON, 3, FExp::from_f64(1e-30));
        let expect_r2 = (BLA_EPSILON as f64 / 2.0).powi(2) * orbit[1].mag2().to_f64();
        assert!(approx(radii.levels[0][0].to_f64(), expect_r2, 1e-2),
            "leaf r^2 = {} expected {}", radii.levels[0][0].to_f64(), expect_r2);
    }

    /// A two-step (level-1) entry must equal applying its two leaves in sequence.
    #[test]
    fn level1_equals_two_leaves() {
        let orbit = Arc::new(ref_orbit(-0.12, 0.75, 9)); // M = 8
        let table = block_on(build_bla(orbit.clone(), 2, pool()));

        let dz0 = ComplexFExp::from_f64_pair(0.2, -0.1);
        let dc = ComplexFExp::from_f64_pair(0.05, 0.07);

        // Two direct linear steps using Z[1], Z[2].
        let dz1 = orbit[1].double() * dz0 + dc;
        let dz2 = orbit[2].double() * dz1 + dc;

        let e = table.levels[1][0]; // covers m = 1,2
        assert_eq!(e.l, 2);
        let dz_bla = e.a * dz0 + e.b * dc;
        assert!(approx(dz_bla.re.to_f64(), dz2.re.to_f64(), 1e-4));
        assert!(approx(dz_bla.im.to_f64(), dz2.im.to_f64(), 1e-4));
    }

    /// Level sizes follow the binary halving, and leaf radius = epsilon*|Z|.
    #[test]
    fn structure_and_leaf_radius() {
        let orbit = Arc::new(ref_orbit(-0.75, 0.06, 17)); // M = 16
        let table = block_on(build_bla(orbit.clone(), 2, pool()));
        let sizes: Vec<usize> = table.levels.iter().map(|l| l.len()).collect();
        assert_eq!(sizes, vec![16, 8, 4, 2, 1]);
        assert_eq!(table.m, 16);

        let radii = compute_radii(&table, BLA_EPSILON, 2, FExp::from_f64(1e-30));
        // leaf 0 covers m=1 -> Z[1] = c. r = epsilon * |c|.
        let z1_mag = orbit[1].mag().to_f64();
        let expect_r2 = (BLA_EPSILON as f64 * z1_mag).powi(2);
        assert!(approx(radii.levels[0][0].to_f64(), expect_r2, 1e-2),
            "leaf r^2 = {} expected {}", radii.levels[0][0].to_f64(), expect_r2);
    }

    /// flatten() must preserve every entry and produce a dims header whose
    /// offsets reproduce the original (level, index) addressing.
    #[test]
    fn flatten_roundtrips_indexing() {
        let orbit = Arc::new(ref_orbit(-0.75, 0.06, 17)); // M = 16
        let table = block_on(build_bla(orbit.clone(), 2, pool()));
        let radii = compute_radii(&table, BLA_EPSILON, 2, FExp::from_f64(1e-30));
        let (entries, dims) = table.flatten();
        let flat_radii = radii.flatten();

        let level_count = dims[0] as usize;
        let steps = dims[1] as usize;
        assert_eq!(level_count, table.levels.len());
        assert_eq!(steps, table.m);
        assert_eq!(entries.len(), flat_radii.len());

        for level in 0..level_count {
            let off = dims[2 + level] as usize;
            let count = dims[3 + level] as usize - off;
            assert_eq!(count, table.levels[level].len());
            for ix in 0..count {
                let e = entries[off + ix];
                let src = table.levels[level][ix];
                assert_eq!(e.l, src.l);
                assert_eq!(e.a.re.m, src.a.re.m);
                assert_eq!(flat_radii[off + ix].e, radii.levels[level][ix].e);
            }
        }
        // Last sentinel == total entry count.
        assert_eq!(dims[2 + level_count] as usize, entries.len());
    }

    /// Lookup climbs to a bigger jump when the radius allows and falls back to a
    /// smaller one when |dz| is too large.
    #[test]
    fn lookup_picks_by_radius() {
        let orbit = Arc::new(ref_orbit(-0.75, 0.06, 17));
        let table = block_on(build_bla(orbit.clone(), 2, pool()));
        let radii = compute_radii(&table, BLA_EPSILON, 2, FExp::from_f64(1e-30));

        // Tiny |dz|^2 at m=1 should resolve to the highest aligned level.
        let tiny = FExp::from_f64(1e-90);
        let hit = lookup(&table, &radii, 1, tiny).expect("should find a step");
        let (level, ix) = hit;
        assert_eq!(ix, 0);

        // A |dz|^2 larger than every radius at m=1 yields no step (fall back).
        let huge = FExp::from_f64(1e30);
        assert!(lookup(&table, &radii, 1, huge).is_none());

        // The found level is the largest whose radius still contains `tiny`.
        assert!(tiny.lt_pos(radii.levels[level][ix]));
    }
}
