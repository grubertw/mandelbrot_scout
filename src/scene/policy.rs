// Scout Config
pub const ORBIT_RNG_SEED: u64 = 12345;
pub const INIT_RUG_PRECISION: u32 = 128;
pub const MAX_LIVE_ORBITS: u32 = 1000;
pub const MAX_REF_ORBIT: u32 = 8192;
pub const NUM_ORBITS_PER_TILE_SPAWN: u32 = 2;
pub const MAX_TILE_ANCHOR_FAILURE_ATTEMPTS: u32 = 8;
pub const LEVEL_ZERO_TILE_CONSTRAINT_BEFORE_EVAL: u32 = 5;
pub const EXPLORATION_BUDGET: f64 = 5.0;

// Initial Scene config/start location
// Start with the center of the mandelbrot shifted slightly to the left
pub const CENTER: (f64, f64) = (-0.75, 0.0);
// In a square, ensure the initial span is from -2 to 1
pub const COMPLEX_SPAN: f64 = 3.0;

// Controlls WGPU Texture allocation
pub const MAX_ORBITS_PER_FRAME: u32 = 264;
pub const ROWS_PER_ORBIT: u32 = 4;
pub const MAX_SCREEN_WIDTH:  u32 = 3840; // Support for a 4k display
pub const MAX_SCREEN_HEIGHT: u32 = 2160; // Support for a 4k display