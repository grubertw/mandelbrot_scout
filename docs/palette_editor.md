# Custom Palette Editor — Design

Status: **COMPLETE — all 7 steps implemented** (2026-07-10)

A dedicated editor for inspecting, modifying, generating, and saving color
palettes. Inspired by Fractal-Zoomer's "Custom Palette Editor" but deliberately
simpler, and adapted to Mandelbrot Scout's GPU-interpolated palette pipeline.

## Goals

- **Inspect** an imported MAP palette: see its colors, run structure, a live
  preview, R/G/B channel graphs, and length.
- **Edit in place**: recolor / resize / insert / delete colors, with edits
  applied **live** to the currently rendered fractal.
- **Generate** new palettes from 16 algorithms (ported from FZ), each reroll
  using a fresh seed.
- **Save** to a (renamed) `.map` file and register it in the palette pick-list.

## Non-goals (v1)

- Editable R/G/B curves (drag-to-reshape channels). Graphs are **read-only** in
  v1; editing happens via the stop list + color picker.
- Persisting stop/run structure to disk (MAP can't hold it; not needed — see
  Stops).
- `same_hues` / `pastel` post-processors from FZ (redundant, dropped).
- Count randomization from FZ (that's the length-randomization we intentionally
  avoid).

---

## How it plugs into the existing pipeline

The current palette path stays the source of record and is **not** restructured:

- Storage: `Rgba8Palette { name, palette: Vec<[u8;4]> }` in
  `Scene.color_palettes: HashMap<String, _>` (`scene.rs`), selected by key with a
  `palette_changed` dirty flag.
- Upload (`scene.rs::upload_color_palette`): source colors are *repeated* into a
  1024-wide 1-D texture (`src[i % len]`); `palette_len = src.len()`.
- Sampling (`color.wgsl::palette_lookup`, ~line 158): GPU does linear `mix`
  between adjacent source colors, then applies `cycles` / `offset` / `gamma`.
  `t % tex_width` means short palettes **cycle seamlessly** across the repeat
  boundary. Histogram equalization happens upstream on `t_in`, so it composes.

The editor produces a flat `Vec<[u8;4]>` and hands it to the existing upload
path. Nothing about the GPU texture/upload changes except a new interpolation
mode uniform (below).

---

## Data model — stops are a soft, session-only view

The flat `Vec<[u8;4]>` remains authoritative on disk. **Stops** are GUI
scaffolding derived once, on load or generation, by run-length grouping adjacent
identical colors.

```rust
struct EditablePalette { name: String, stops: Vec<Stop> }
struct Stop { color: [u8;4], count: u32 }   // count >= 1
// equivalently a Vec<[u8;4]> of identical entries; flatten = concat
```

- **Load** (MAP import or generator output): scan the flat vec; start a new stop
  on any RGB change. Dead simple.
- Stops are independent after creation — recoloring two adjacent stops to the
  same value keeps them separate (no re-merging).
- **Every edit** → flatten (`stops.concat()`) → replace the selected palette's
  `Vec<[u8;4]>` → set `palette_changed = true` → request redraw on the main
  window. Live, in place, through the existing upload path.
  `palette_len` = flattened length.
- **Save** = flatten → MAP `R G B` lines. Stops never touch disk; re-loading
  re-derives them.

### Edit operations

- Recolor a stop → recolors its whole run.
- Grow / shrink `count` (min 1) → the star control for short palettes, where a
  count bump has outsized, easy-to-judge visual effect. Stepper on each row;
  optional drag-to-resize.
- Insert a stop anywhere → starts `count = 1`, **color copies the previous stop**
  (not black), growable.
- Delete a stop.
- Total length = Σ count, shown live, **capped at 1024** (texture width).

### Short palettes are preferred

Count-1 stops + GPU interpolation already look smooth (the default 3-color R/G/B
palette proves it). The 1024 texture is headroom, not a target. Shorter palettes
make each `count` change more impactful (easier to customize) and make the
non-linear interp modes matter more (each segment covers more visual range).

---

## Interpolation — GPU-side, systemwide uniform

Interpolation stays entirely on the GPU. Add one `palette_interp_mode` scene
uniform + enum, mirroring `color_scalar_mapping_mode`, consumed by a `switch` in
`palette_lookup` (`color.wgsl`). The commented smoothstep / gamma-correct lines
just below the current `mix` become the enum arms.

Modes: `Linear` (current) · `Smoothstep` · `Gamma-correct` (lerp in linear light)
· `Cosine` · optionally `OkLab-in-shader`.

Set from a dropdown in the Color panel (or the editor). One consistency note: the
**trap** color path (`color.wgsl`, ~line 315) should read the same uniform so
both lookups stay consistent.

No per-palette interpolation and no CPU rasterization of curves.

---

## Generators — 16 algorithms, ported from FZ

Source of truth studied: `Fractal-Zoomer/src/fractalzoomer/gui/
CustomPaletteEditorDialog.java::randomPalette(...)` (alg 0–15) and helper classes
in `.../utils/`.

**Key reframe:** FZ's generators emit **control colors** (`palette.length` ≈ 32),
not the final palette; FZ then assigns each a random count of 7–18 and
interpolates on the CPU, so total length ≈ 398 — the length randomization we
avoid.

**Our model:**
- **Generation input is `K` = number of colors**, not length. Default K modest
  (~8–16 per algo; Triad/Tetrad keep their natural 32).
- Each generated color becomes a **count-1 stop**; GPU interpolation smooths
  across them. Total length is deterministic (= K) and edited afterward by
  growing stops.
- Every algo takes a fresh **seed** on "Roll" (reproducible). Expose the 1–2
  genuinely meaningful params only where they exist (below); leave the rest
  seed-driven.

| # | Name | What it is | Port |
|---|------|-----------|------|
| 0 | Golden Ratio | `brightness += 0.61803… mod 1`, random hue/sat → HSB→RGB | arithmetic + HSB |
| 1 | Waves | per-channel `0.5*(sin(π/coeff·(m+1)+phase)+1)` | arithmetic |
| 2 | Distance | local `ColorGenerator` (137 lines): max-distinct via 3D subdivision in RYB | port class |
| 3 | Triad | HSB, 3 hues 120° apart, hardcoded 32 | arithmetic |
| 4 | Tetrad | HSB, 4 hues 90° apart, hardcoded 32 | arithmetic |
| 5 | Google Material | local static Material color table | port data |
| 6 | ColorBrewer 1 | local ColorBrewer table (`generate2`) | port data |
| 7 | ColorBrewer 2 | same table (`generate`) | port data |
| 8 | Google-ColorBrewer | merges the two tables | port data |
| 9 | Cubehelix | D.A. Green (2011); params start/rotation/gamma/sat | port ~60-line formula |
| 10 | IQ-Cosines | `a+b·cos(2π(c·t+d)+g)` per channel, normalized | arithmetic |
| 11 | Perlin | jnoise Perlin, sampled around a 2π circle, normalized | `noise` crate |
| 12 | Simplex | jnoise FastSimplex | `noise` crate |
| 13 | Perlin+Simplex | average of 11 & 12 | `noise` crate |
| 14 | Random Walk | ±step per channel, reflect at bounds | arithmetic |
| 15 | Simple Random | random color per slot in one of 11 random color spaces | arithmetic + conversions |

**Exposed params (beyond seed):** Cubehelix (start / rotation / gamma),
IQ-Cosines (the 4 `a,b,c,d` vec3s, or at minimum frequency), noise (scale /
frequency). Everything else seed-only.

**Noise detail:** FZ walks the noise field around a full `2π` circle
(`cos/sin(a+phase)`) so the palette loops seamlessly — replicate so cycling stays
continuous. Seed each channel's noise generator from the master RNG.

---

## Dependencies

Existing and kept: `iced_widget`/`iced_winit`/`iced_wgpu` 0.14 (hand-integrated),
`rand` 0.8, `rand_chacha` 0.3.

New crates (both pure-Rust, no system deps):
- **`noise`** — Perlin + OpenSimplex fields for algos 11–13. Direct jnoise
  replacement. `rand_chacha` alone can't produce coherent gradient noise.
- **`palette`** — HSB/HSL/Lab/Lch/**Oklab/Oklch**/XYZ conversions used by Golden
  Ratio variants and Simple Random, and by the picker. Also useful if we ever do
  OkLab work outside the shader.

RNG split (mirrors FZ's single `java.util.Random`):
- `ChaCha8Rng` is the **master `generator`** — its seed is the Roll button
  (reproducible), it draws all algo params and *seeds* the `noise` generators.
- `noise` provides only the coherent-noise fields.

Ported as data/formula (no dep): Material + ColorBrewer tables (5–8), Cubehelix
(9), `ColorGenerator` for Distance (2). Porting FZ's exact tables gives
byte-identical output to the MAP files already recognized by name.

---

## UI shell — a separate dialog window

The editor is too large for an inline overlay panel (unlike the other Iced
panels). It lives in a **second winit window**, which is cheap here because
`main.rs` is a hand-rolled `ApplicationHandler` (`Runner`) with a manual iced
runtime, and `window_event` already receives a `WindowId` (currently ignored).

- Add `Option<PaletteWindow { window, surface, renderer, ui_state }>` to
  `Runner::Ready`; create it on demand via `event_loop.create_window`.
- Route by `window_id` in `window_event` (main UI vs editor UI).
- Shared `Rc<RefCell<Scene>>` — the second window is on the **same event loop /
  thread**, so no `Arc<Mutex>` is required. On edit: `scene.borrow_mut()` →
  flatten → set palette + `palette_changed` → `main_window.request_redraw()`.

### Widgets

- **Color picker: hand-rolled**, not `iced_aw` (which builds against the umbrella
  `iced` crate and lags 0.14 — real conflict risk with the split-crate manual
  setup). Build from existing widgets + `canvas`: RGB + HSV sliders, hex field,
  `canvas` SV-square, hue strip. `palette` crate does the conversions.
- **Preview bar + R/G/B graphs: `canvas`** (built into `iced_widget`, no new
  dep). Same widget backs the picker's SV-square. Graphs read-only in v1.

### Layout (reference)

Stop list (swatch + hex + count stepper + delete) · full-width preview bar ·
R/G/B channel graphs · generator panel (algorithm dropdown + Colors K + Seed /
Roll + exposed params) · interp mode dropdown · Apply / Export MAP. Palette name
field feeds the MAP filename and the pick-list key.

---

## Workflow

1. Open editor → loads a working copy (stops) of the currently selected palette.
2. Inspect: stop list, preview bar, R/G/B graphs, length.
3. Edit: recolor / resize / insert (copies previous) / delete; change interp
   mode (systemwide).
4. Generate: pick algorithm + Colors K + Roll (fresh seed) → fills stops; then
   hand-tweak.
5. Apply → updates/creates the in-memory palette (live on the fractal).
   Export MAP → `rfd` save dialog, writes the flattened file and registers it.

## Implementation order (suggested)

1. **[DONE 2026-07-10]** `palette_interp_mode` uniform + `color.wgsl` switch +
   Color-panel dropdown (independent, immediately useful, no window work).
   Shipped 5 modes (Linear/Smoothstep/Gamma-correct/Cosine/OkLab) via a shared
   `interp_palette()` used by both the main and trap lookups. Struct padded to
   256 bytes (3 reserved u32s). Field also added to `histogram.wgsl`'s Uniforms.
2. **[DONE 2026-07-10]** `EditablePalette` / `Stop` model + RLE grouping +
   flatten + edit ops, in `src/palette_editor.rs` with 8 unit tests (Fire Ice +
   run-bearing fixtures). `mod palette_editor` is `#[allow(dead_code)]` until the
   window wires it. Live re-upload hook (Scene method) deferred to step 3, since
   it needs a caller. `flatten()` is uncapped; caller caps to `max_palette_colors`.
3. **[DONE 2026-07-10]** Second window plumbing. `Runner::Ready` gained
   `instance`/`adapter`/`palette_window: Option<PaletteWindow>`; `window_event`
   routes by `window_id`. New `src/palette_window.rs` holds `PaletteEditor` (UI,
   mirrors `Controls`) + `PaletteWindow` (own surface/renderer/viewport/cache,
   clears to a dark bg via `present(Some(bg), ..)`). Opened by an "Edit" button
   (`Message::OpenPaletteEditor` → `controls.take_open_palette_editor()` → Runner
   creates the window). Shares `Rc<RefCell<Scene>>`; live edits call
   `Scene::set_selected_palette_colors` (+`selected_palette_colors`/`_display_name`
   getters) and redraw the main window. Placeholder view proves the live loop with
   a real **Reverse** edit. `winit` imported via `iced_winit::winit`.
4. **[DONE 2026-07-10]** Read-only preview bar + R/G/B graphs on `canvas` (in
   `src/palette_window.rs`). Added the `"canvas"` feature to `iced_widget` (pulls
   lyon). Two `canvas::Program`s parameterized over the app's `iced_wgpu::Renderer`
   (which impls `graphics::geometry::Renderer`): `PalettePreview` (per-column
   linear gradient of the flattened palette) and `ChannelGraphs` (3 stacked R/G/B
   polylines, value 1.0 at band top). Rebuilt each `view()` from
   `palette.flatten()`. Preview is linear-only for now (doesn't yet mirror the
   GPU interp-mode/cycles/offset).
5. **[DONE 2026-07-10]** Stop-list editing UI + hand-rolled color picker (in
   `src/palette_window.rs`). Stop rows: color swatch (click to select) + hex +
   count `-`/`+` stepper + Ins/Del, wired to the model edit ops. Picker for the
   selected stop: interactive `SvSquare` + `HueStrip` canvases (drag → publish
   `Action`), RGB + HSV sliders, big swatch + hex. `picker_hsv` is held
   independent of the stop RGB so hue survives S/V→0. Hand-rolled `rgb_to_hsv` /
   `hsv_to_rgb` (no `palette` crate yet; add it with the generators). Whole editor
   wrapped in a `scrollable`. Every edit applies live via `apply()`.
6. **[COMPLETE 2026-07-10]** Generators — all 16 in `src/palette_generators.rs`.
   The 11 procedural: Golden Ratio, Waves, Triad, Tetrad, Cubehelix (D.A. Green formula
   ported), IQ-Cosines, Perlin, Simplex, Perlin+Simplex (via the new `noise`
   crate, circle-sampled so they loop), Random Walk, Simple Random. `Generator`
   enum (strum) + `GenParams` (colors K, seed, cubehelix start/rotation/gamma, IQ
   frequency, noise scale). `ChaCha8Rng` master seed = the Roll button; param
   tweaks re-run with the same seed. Color helpers `hsv/hsl/rgb` now live here
   (picker imports them). UI: a "Generate" panel (dropdown + Colors slider + Roll
   + conditional param sliders). NOTE: edition-2024 makes `gen` a keyword — use
   `gen_range`/`gen_bool`, never bare `gen`. No `palette` crate — HSV/HSL
   hand-rolled; only `noise` added.
   The data-heavy 5: Distance (`ColorGenerator` farthest-point subdivision over an
   RYB grid + `ryb_to_rgb`), Google Material, ColorBrewer 1 (`generate2`),
   ColorBrewer 2 (`generate`/MIXED), Google-ColorBrewer. FZ tables transcribed
   verbatim (Material 0xRRGGBB, ColorBrewer signed-ARGB; both via `argb_to_rgb`,
   alpha ignored). `category_select` (category→ramp→run of ≥2, refill to guarantee
   K) + `ramp_select` (whole diverging schemes asc/desc). Enum reordered to FZ's
   dropdown order. Tables baked into the module for now (could move to toml later).
7. **[COMPLETE 2026-07-10]** Export MAP + pick-list registration. "Save MAP"
   button → rfd `save_file` dialog (default dir = new `default_export_palette_directory`
   setting, "$HOME/Pictures/Fractals/Palettes"; default filename = palette name +
   `.map`). Writes one `R G B` line per flattened slot. Filename stem → palette
   name, full filename → key (matches import). `Scene::save_palette` registers +
   selects it and sets a `palettes_dirty` flag; the main window's redraw calls
   `Controls::refresh_palettes()` (cross-window sync, since the editor is a
   separate window). Colors slider max also bumped 48 → 64.
