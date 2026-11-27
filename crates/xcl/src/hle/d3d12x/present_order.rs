//! PresentX frame-token order (`XCL_PRESENT_TOKEN_ORDER`, default 1; 0: every present is shown).
//!
//! Each `PresentX` plane starts with the frame token `WaitFrameEventX` handed out at that
//! frame's origin (`frame`; plane words `[token, resource count, resources, …]`, the planes of
//! one present carry the same token). A present whose token is older than the newest one shown
//! is not shown, so a late frame never replaces a newer one on screen. Its flip is still
//! reported (`frame::mark_flipped`): the title's frame statistics, and with them its buffer
//! reuse and pacing, go on as if it had been shown. XCL composes to one output (one presenter
//! and window; a PresentX from another queue moves the composition there), so one order holds.
//!
//! Tokens are `frame::TOKEN_MASK` bits wide and compared as serial numbers (RFC 1982), so the
//! order holds across a wrap. A token further behind the newest shown than the frame clock
//! keeps tokens for (`frame::KNOWN_TOKENS`) is no frame still in flight but a count that started
//! again (a mode change, a device reset): it is shown and the order starts from it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use super::frame::{KNOWN_TOKENS, TOKEN_MASK};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// as new as the newest shown, or newer
    Show,
    /// older than the newest shown
    Drop,
    /// behind by more than any frame in flight: a new count
    Reset,
    /// the null token (no frame named), or the order is off
    Unordered,
}

/// The newest token shown.
pub struct Order {
    newest: Option<u64>,
}

impl Order {
    pub const fn new() -> Self {
        Order { newest: None }
    }

    pub fn present(&mut self, token: u64) -> Verdict {
        let token = token & TOKEN_MASK;
        if token == 0 {
            return Verdict::Unordered;
        }
        let behind = self.newest.map_or(0, |n| n.wrapping_sub(token) & TOKEN_MASK);
        let verdict = if behind == 0 || behind > TOKEN_MASK >> 1 {
            Verdict::Show
        } else if behind <= KNOWN_TOKENS as u64 {
            return Verdict::Drop;
        } else {
            Verdict::Reset
        };
        self.newest = Some(token);
        verdict
    }
}

static ORDER: Mutex<Order> = Mutex::new(Order::new());
static DROPPED: AtomicU64 = AtomicU64::new(0);
static LOGGED: AtomicU64 = AtomicU64::new(0);
/// the first presents' tokens are logged (`present token` lines)
const LOG_FIRST: u64 = 12;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_PRESENT_TOKEN_ORDER").map_or(true, |v| v.trim() != "0"))
}

/// Presents dropped as older than the newest shown, so far (`XCL_ACTIVITY`).
pub fn dropped() -> u64 {
    DROPPED.load(Ordering::Relaxed)
}

/// `PresentX` of planes whose tokens are `tokens` (plane 0's names the frame): `frame` runs with
/// whether the frame is shown, then its flip is reported, so a dropped present completes as a
/// shown one does.
pub fn present(tokens: &[u64], frame: impl FnOnce(bool)) -> Verdict {
    present_in(&ORDER, enabled(), tokens, frame)
}

fn present_in(order: &Mutex<Order>, on: bool, tokens: &[u64], frame: impl FnOnce(bool)) -> Verdict {
    let token = tokens.first().copied().unwrap_or(0);
    let (verdict, newest) = {
        let mut o = order.lock().unwrap();
        let verdict = if on { o.present(token) } else { Verdict::Unordered };
        (verdict, o.newest)
    };
    if verdict == Verdict::Drop {
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
    let n = LOGGED.fetch_add(1, Ordering::Relaxed);
    if n < LOG_FIRST {
        crate::xlog_info!("D3D12.X: present token #{n}: {token:#x} (planes {tokens:x?}): {verdict:?}, newest shown {newest:x?}");
    }
    frame(verdict != Verdict::Drop);
    super::frame::mark_flipped(token);
    verdict
}

#[cfg(test)]
mod tests {
    use super::Verdict::*;
    use super::*;

    fn verdicts(tokens: &[u64]) -> Vec<Verdict> {
        let mut order = Order::new();
        tokens.iter().map(|&t| order.present(t)).collect()
    }

    #[test]
    fn in_order_presents_are_all_shown() {
        assert_eq!(verdicts(&[1, 2, 3, 3, 4]), [Show; 5]);
    }

    #[test]
    fn an_older_token_is_dropped_and_the_newest_stays() {
        assert_eq!(verdicts(&[5, 7, 6, 8, 7, 9]), [Show, Show, Drop, Show, Drop, Show]);
    }

    /// The frame clock skips the null token when its count wraps.
    #[test]
    fn the_order_holds_across_the_token_wrap() {
        assert_eq!(verdicts(&[TOKEN_MASK - 1, TOKEN_MASK, 1, TOKEN_MASK, 2]), [Show, Show, Show, Drop, Show]);
    }

    #[test]
    fn a_jump_back_past_the_frames_in_flight_is_a_reset() {
        assert_eq!(verdicts(&[5000, 1, 2, 3]), [Show, Reset, Show, Show]);
        let k = KNOWN_TOKENS as u64;
        assert_eq!(verdicts(&[1000, 1000 - k]), [Show, Drop]);
        assert_eq!(verdicts(&[1000, 1000 - k - 1, 1000]), [Show, Reset, Show]);
    }

    #[test]
    fn the_null_token_is_shown_and_keeps_the_order() {
        assert_eq!(verdicts(&[5, 0, 4, 6]), [Show, Unordered, Drop, Show]);
    }

    /// The dropped frame's own bookkeeping runs (told it is not shown) and its flip is reported,
    /// so the title's GetFrameStatisticsX for it succeeds as for a shown frame.
    #[test]
    fn a_dropped_present_still_reports_its_flip() {
        let order = Mutex::new(Order::new());
        let (newer, older) = (0x5EED_0002, 0x5EED_0001);
        let mut shown = Vec::new();
        assert_eq!(present_in(&order, true, &[newer, newer], |s| shown.push(s)), Show);
        assert_eq!(present_in(&order, true, &[older, older], |s| shown.push(s)), Drop);
        assert_eq!(shown, [true, false]);
        let (mut n, mut st) = (1u32, [0u8; 0x40]);
        let hr = unsafe { super::super::frame::frame_statistics_x(std::ptr::null_mut(), older, 0, &mut n, st.as_mut_ptr()) };
        assert_eq!(hr, crate::util::S_OK);
        let flip = i64::from_le_bytes(st[0x18..0x20].try_into().unwrap());
        assert!(n == 1 && flip > 0, "{n} {flip}");
    }

    #[test]
    fn with_the_order_off_every_present_is_shown() {
        let order = Mutex::new(Order::new());
        for t in [0x5EED_0107, 0x5EED_0106, 0x5EED_0105] {
            assert_eq!(present_in(&order, false, &[t], |s| assert!(s)), Unordered);
        }
    }
}
