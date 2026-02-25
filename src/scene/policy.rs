// Scout Config
pub const ORBIT_RNG_SEED: u64 = 12345;
pub const INIT_RUG_PRECISION: u32 = 128;
pub const MAX_LIVE_ORBITS: u32 = 1000;
pub const MAX_REF_ORBIT: u32 = 8192;
pub const AUTO_START: bool = false;
pub const STARTING_SCALE: f64 = 1e-9;
pub const STARTING_TILE_PIXEL_SPAN: f64 = 256.0;
pub const SMALLEST_TILE_PIXEL_SPAN: f64 = 10.0;
pub const COVERAGE_TO_ANCHOR: f64 = 1.0;
pub const NUM_ORBITS_PER_TILE_SPAWN: u32 = 2;
pub const MAX_TILE_ANCHOR_FAILURE_ATTEMPTS: u32 = 40;
pub const SPLIT_TILE_ON_POOR_COVERAGE: u32 = 16;
pub const EXPLORATION_BUDGET: i32 = 2;

// Initial Scene config/start location
// Start with the center of the mandelbrot shifted slightly to the left
pub const CENTER: (f64, f64) = (-0.75, 0.0);
// In a square, ensure the initial span is from -2 to 1
pub const COMPLEX_SPAN: f64 = 3.0;
// Maps to User Widgets
pub const MAX_USER_ITER: u32 = 500;

// Controlls WGPU Texture allocation
pub const MAX_ORBITS_PER_FRAME: u32 = 264;
pub const ROWS_PER_ORBIT: u32 = 4;
pub const MAX_SCREEN_WIDTH:  u32 = 3840; // Support for a 4k display
pub const MAX_SCREEN_HEIGHT: u32 = 2160; // Support for a 4k display