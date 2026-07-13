//! Session-only editable palette model for the Custom Palette Editor.
//!
//! See `docs/palette_editor.md`. A palette on disk (and on the GPU) is always a
//! flat `Vec<[u8; 4]>`. `Stop`s are a soft, run-length *view* over that flat
//! list — derived once via [`EditablePalette::from_colors`] by grouping runs of
//! identical adjacent colors, and collapsed back with
//! [`EditablePalette::flatten`]. Stops never persist; they exist only while
//! editing, to make "extend a color across N slots" and per-color editing
//! natural in the GUI.

/// One run of identical color occupying `count` consecutive palette slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stop {
    pub color: [u8; 4],
    pub count: u32, // invariant: >= 1
}

impl Stop {
    fn new(color: [u8; 4]) -> Self {
        Stop { color, count: 1 }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EditablePalette {
    pub name: String,
    pub stops: Vec<Stop>,
}

impl EditablePalette {
    /// Group a flat color list into stops by runs of identical adjacent colors.
    /// e.g. `[a, a, b, a]` -> `[(a,2), (b,1), (a,1)]`.
    pub fn from_colors(name: impl Into<String>, colors: &[[u8; 4]]) -> Self {
        let mut stops: Vec<Stop> = Vec::new();
        for &color in colors {
            match stops.last_mut() {
                Some(last) if last.color == color => last.count += 1,
                _ => stops.push(Stop::new(color)),
            }
        }
        EditablePalette { name: name.into(), stops }
    }

    /// Total number of slots the palette expands to (Σ count == `flatten().len()`).
    pub fn total_len(&self) -> u32 {
        self.stops.iter().map(|s| s.count).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.stops.is_empty()
    }

    /// Collapse stops back to a flat color list (the GPU / disk representation).
    /// The caller caps to the GPU texture width (`max_palette_colors`) at upload.
    pub fn flatten(&self) -> Vec<[u8; 4]> {
        let mut out = Vec::with_capacity(self.total_len() as usize);
        for s in &self.stops {
            for _ in 0..s.count {
                out.push(s.color);
            }
        }
        out
    }

    /// Recolor stop `i`'s entire run. No-op if out of range.
    pub fn set_color(&mut self, i: usize, color: [u8; 4]) {
        if let Some(s) = self.stops.get_mut(i) {
            s.color = color;
        }
    }

    /// Set stop `i`'s slot count (clamped to >= 1). No-op if out of range.
    pub fn set_count(&mut self, i: usize, count: u32) {
        if let Some(s) = self.stops.get_mut(i) {
            s.count = count.max(1);
        }
    }

    /// Grow stop `i` by one slot.
    pub fn grow(&mut self, i: usize) {
        if let Some(s) = self.stops.get_mut(i) {
            s.count += 1;
        }
    }

    /// Shrink stop `i` by one slot (never below 1).
    pub fn shrink(&mut self, i: usize) {
        if let Some(s) = self.stops.get_mut(i) {
            s.count = s.count.saturating_sub(1).max(1);
        }
    }

    /// Insert a new count-1 stop after `i`, copying `i`'s color (not black), per
    /// the design. If the palette is empty, inserts a single black stop.
    pub fn insert_after(&mut self, i: usize) {
        match self.stops.get(i) {
            Some(s) => {
                let color = s.color;
                self.stops.insert(i + 1, Stop::new(color));
            }
            None => self.stops.push(Stop::new([0, 0, 0, 255])),
        }
    }

    /// Delete stop `i`, keeping at least one stop. No-op if it would empty the list.
    pub fn delete(&mut self, i: usize) {
        if self.stops.len() > 1 && i < self.stops.len() {
            self.stops.remove(i);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(r: u8, g: u8, b: u8) -> [u8; 4] {
        [r, g, b, 255]
    }

    // The user's real "Fire Ice" palette from settings.toml: 9 distinct colors.
    fn fire_ice() -> Vec<[u8; 4]> {
        vec![
            c(0, 0, 0),
            c(255, 0, 0),
            c(255, 255, 0),
            c(255, 255, 255),
            c(128, 255, 255),
            c(0, 255, 255),
            c(0, 128, 255),
            c(0, 0, 255),
            c(0, 0, 128),
        ]
    }

    #[test]
    fn from_colors_distinct_gives_count1_stops() {
        let p = EditablePalette::from_colors("Fire Ice", &fire_ice());
        assert_eq!(p.stops.len(), 9);
        assert!(p.stops.iter().all(|s| s.count == 1));
        assert_eq!(p.total_len(), 9);
    }

    #[test]
    fn flatten_roundtrips_distinct() {
        let colors = fire_ice();
        let p = EditablePalette::from_colors("Fire Ice", &colors);
        assert_eq!(p.flatten(), colors);
    }

    #[test]
    fn from_colors_groups_runs_and_roundtrips() {
        let a = c(1, 2, 3);
        let b = c(4, 5, 6);
        let colors = vec![a, a, b, a];
        let p = EditablePalette::from_colors("t", &colors);
        assert_eq!(
            p.stops,
            vec![
                Stop { color: a, count: 2 },
                Stop { color: b, count: 1 },
                Stop { color: a, count: 1 },
            ]
        );
        assert_eq!(p.total_len(), 4);
        assert_eq!(p.flatten(), colors);
    }

    #[test]
    fn insert_after_copies_previous_color() {
        let mut p = EditablePalette::from_colors("t", &fire_ice());
        let red = p.stops[1].color;
        p.insert_after(1);
        assert_eq!(p.stops.len(), 10);
        assert_eq!(p.stops[2], Stop { color: red, count: 1 });
    }

    #[test]
    fn insert_after_on_empty_inserts_black() {
        let mut p = EditablePalette { name: "t".into(), stops: vec![] };
        p.insert_after(0);
        assert_eq!(p.stops, vec![Stop { color: [0, 0, 0, 255], count: 1 }]);
    }

    #[test]
    fn set_count_and_shrink_clamp_to_one() {
        let mut p = EditablePalette::from_colors("t", &[c(1, 1, 1)]);
        p.set_count(0, 0);
        assert_eq!(p.stops[0].count, 1);
        p.set_count(0, 5);
        assert_eq!(p.stops[0].count, 5);
        p.shrink(0);
        assert_eq!(p.stops[0].count, 4);
        p.set_count(0, 1);
        p.shrink(0);
        assert_eq!(p.stops[0].count, 1); // never below 1
    }

    #[test]
    fn grow_extends_run_and_flatten_reflects_it() {
        let mut p = EditablePalette::from_colors("t", &[c(9, 9, 9), c(1, 2, 3)]);
        p.grow(0);
        p.grow(0); // count 3
        assert_eq!(p.total_len(), 4);
        assert_eq!(
            p.flatten(),
            vec![c(9, 9, 9), c(9, 9, 9), c(9, 9, 9), c(1, 2, 3)]
        );
    }

    #[test]
    fn delete_keeps_at_least_one() {
        let mut p = EditablePalette::from_colors("t", &[c(1, 1, 1)]);
        p.delete(0);
        assert_eq!(p.stops.len(), 1); // refused

        let mut p2 = EditablePalette::from_colors("t", &[c(1, 1, 1), c(2, 2, 2)]);
        p2.delete(0);
        assert_eq!(p2.stops, vec![Stop { color: c(2, 2, 2), count: 1 }]);
    }
}
