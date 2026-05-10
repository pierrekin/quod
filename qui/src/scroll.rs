//! Pure helper for vim-style "scrolloff" scrolling: keep the cursor at
//! least `SCROLLOFF` rows from each viewport edge, until you hit the
//! actual top or bottom of the list (where the cursor can reach the
//! edge).
//!
//! Every scrollable surface owns its own `(cursor, offset, len)` state
//! and calls `compute_offset` at render time; this module is stateless.
pub const SCROLLOFF: usize = 3;

pub fn compute_offset(
    current_offset: usize,
    cursor: usize,
    viewport_h: usize,
    list_len: usize,
) -> usize {
    if list_len == 0 || viewport_h == 0 || list_len <= viewport_h {
        return 0;
    }
    let max_offset = list_len - viewport_h;
    let scrolloff = SCROLLOFF.min(viewport_h.saturating_sub(1) / 2);

    let new_offset = if cursor < current_offset + scrolloff {
        cursor.saturating_sub(scrolloff)
    } else if cursor + scrolloff + 1 > current_offset + viewport_h {
        (cursor + scrolloff + 1).saturating_sub(viewport_h)
    } else {
        current_offset
    };

    new_offset.min(max_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_scroll_when_list_fits() {
        assert_eq!(compute_offset(0, 5, 20, 10), 0);
    }

    #[test]
    fn cursor_at_very_top_of_list_can_touch_edge() {
        assert_eq!(compute_offset(0, 0, 10, 40), 0);
    }

    #[test]
    fn cursor_at_very_bottom_of_list_can_touch_edge() {
        assert_eq!(compute_offset(0, 39, 10, 40), 30);
    }

    #[test]
    fn scrolling_down_starts_when_cursor_within_scrolloff_of_bottom() {
        // viewport=10, scrolloff=3 → cursor=7 should scroll
        assert_eq!(compute_offset(0, 7, 10, 40), 1);
    }

    #[test]
    fn scrolling_up_starts_when_cursor_within_scrolloff_of_top() {
        // current_offset=10, scrolloff=3 → cursor=12 should scroll back to 9
        assert_eq!(compute_offset(10, 12, 10, 40), 9);
    }

    #[test]
    fn middle_cursor_doesnt_change_offset() {
        // current=10, viewport=10, cursor=15 → still in the safe zone (>=13, <=16)
        assert_eq!(compute_offset(10, 15, 10, 40), 10);
    }

    #[test]
    fn small_viewport_clamps_scrolloff_to_half() {
        // viewport=4 → scrolloff capped at (4-1)/2 = 1
        assert_eq!(compute_offset(0, 2, 4, 20), 0);
        assert_eq!(compute_offset(0, 3, 4, 20), 1);
    }

    #[test]
    fn never_exceeds_max_offset() {
        // cursor at end with scrolloff would push past list end
        let n = 40;
        let viewport = 10;
        let off = compute_offset(0, n - 1, viewport, n);
        assert!(off <= n - viewport);
    }
}
