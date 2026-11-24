//! `std::sort` as the C++ standard libraries DXC is built with run it. DXC sorts the rows of a
//! signature part with `std::sort` and a comparison that leaves ties (one semantic name packed
//! twice into a register), the validator sorts again and compares bytes, so the order of ties
//! is whatever the library's algorithm makes of the input, and a different order is a
//! "container part does not match" error. Both libraries are deterministic; they differ.
//!
//! - libstdc++ (DXC on Linux): introsort, an insertion sort up to 16 elements.
//! - MSVC's STL (dxcompiler.dll / dxil.dll): an insertion sort up to 32 elements, which keeps
//!   ties in input order. Beyond that it is a quicksort of its own that is not reproduced
//!   here: ties stay in input order, which may not be what dxil.dll computes for a signature
//!   of more than 32 rows with ties.

/// Whose `std::sort`.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Flavor {
    /// GCC's libstdc++ (bits/stl_algo.h)
    Libstdcxx,
    /// MSVC's STL
    Msvc,
}

impl Flavor {
    /// The library of the DXC builds for the platform this crate is compiled for.
    pub const HOST: Flavor = if cfg!(windows) { Flavor::Msvc } else { Flavor::Libstdcxx };
}

/// Sort `v` by the strict order `less`, moving elements exactly as `flavor`'s std::sort does.
pub fn sort<T: Copy>(v: &mut [T], flavor: Flavor, less: impl Fn(&T, &T) -> bool) {
    match flavor {
        Flavor::Msvc => {
            // _Insertion_sort_unchecked moves an element only past strictly greater ones
            for i in 1..v.len() {
                let val = v[i];
                let mut hole = i;
                while hole > 0 && less(&val, &v[hole - 1]) {
                    v[hole] = v[hole - 1];
                    hole -= 1;
                }
                v[hole] = val;
            }
        }
        Flavor::Libstdcxx => {
            if !v.is_empty() {
                // std::__lg(n) * 2 levels of quicksort before it gives up on a range
                let depth = 2 * (usize::BITS - 1 - v.len().leading_zeros());
                introsort_loop(v, depth, &less);
                final_insertion_sort(v, &less);
            }
        }
    }
}

/// libstdc++'s _S_threshold: ranges up to this are left to the final insertion sort.
const THRESHOLD: usize = 16;

/// `std::__introsort_loop`: partition around a median of three, recurse into the right part.
fn introsort_loop<T: Copy>(v: &mut [T], mut depth: u32, less: &impl Fn(&T, &T) -> bool) {
    let mut last = v.len();
    while last > THRESHOLD {
        if depth == 0 {
            heap_sort(&mut v[..last], less);
            return;
        }
        depth -= 1;
        let cut = partition_pivot(&mut v[..last], less);
        introsort_loop(&mut v[cut..last], depth, less);
        last = cut;
    }
}

/// `std::__unguarded_partition_pivot`: the median of the second, middle and last element goes
/// to the front as the pivot (`__move_median_to_first`), then `__unguarded_partition` of the
/// rest around it.
fn partition_pivot<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) -> usize {
    let (a, b, c) = (1, v.len() / 2, v.len() - 1);
    let median = if less(&v[a], &v[b]) {
        if less(&v[b], &v[c]) {
            b
        } else if less(&v[a], &v[c]) {
            c
        } else {
            a
        }
    } else if less(&v[a], &v[c]) {
        a
    } else if less(&v[b], &v[c]) {
        c
    } else {
        b
    };
    v.swap(0, median);
    let (mut first, mut last) = (1, v.len());
    loop {
        while less(&v[first], &v[0]) {
            first += 1;
        }
        last -= 1;
        while less(&v[0], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `std::__final_insertion_sort`: the first 16 elements with the guarded insertion sort (a new
/// minimum moves to the front in one step), the rest unguarded (the minimum is in place).
fn final_insertion_sort<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
    for i in 1..v.len() {
        let val = v[i];
        if i < THRESHOLD && less(&val, &v[0]) {
            v.copy_within(0..i, 1);
            v[0] = val;
        } else {
            let mut hole = i;
            while hole > 0 && less(&val, &v[hole - 1]) {
                v[hole] = v[hole - 1];
                hole -= 1;
            }
            v[hole] = val;
        }
    }
}

/// `std::__partial_sort(first, last, last)`, what introsort falls back to: `__make_heap`, then
/// `__sort_heap`.
fn heap_sort<T: Copy>(v: &mut [T], less: &impl Fn(&T, &T) -> bool) {
    let len = v.len();
    if len >= 2 {
        for parent in (0..=(len - 2) / 2).rev() {
            let value = v[parent];
            adjust_heap(v, parent, len, value, less);
        }
    }
    for last in (1..len).rev() {
        let value = v[last];
        v[last] = v[0];
        adjust_heap(v, 0, last, value, less);
    }
}

/// `std::__adjust_heap`: sift the hole down to a leaf along the larger children, then
/// `__push_heap` the value up from there.
fn adjust_heap<T: Copy>(v: &mut [T], mut hole: usize, len: usize, value: T, less: &impl Fn(&T, &T) -> bool) {
    let top = hole;
    let mut child = hole;
    while child < (len - 1) / 2 {
        child = 2 * (child + 1);
        if less(&v[child], &v[child - 1]) {
            child -= 1;
        }
        v[hole] = v[child];
        hole = child;
    }
    if len & 1 == 0 && child == (len - 2) / 2 {
        child = 2 * (child + 1);
        v[hole] = v[child - 1];
        hole = child - 1;
    }
    while hole > top {
        let parent = (hole - 1) / 2;
        if !less(&v[parent], &value) {
            break;
        }
        v[hole] = v[parent];
        hole = parent;
    }
    v[hole] = value;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The generator of the reference program (below): keys with many ties, each element
    /// tagged with its position.
    fn case(seed: u32, n: usize, keys: u32) -> Vec<(u32, u32)> {
        let mut x = seed;
        (0..n as u32)
            .map(|i| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                ((x >> 16) % keys, i)
            })
            .collect()
    }

    fn fnv(h: &mut u32, v: &[(u32, u32)]) {
        for &(_, tag) in v {
            *h = (*h ^ tag).wrapping_mul(16777619);
        }
    }

    fn by_key(a: &(u32, u32), b: &(u32, u32)) -> bool {
        a.0 < b.0
    }

    /// Against g++ 13's std::sort (the libstdc++ the Linux DXC is built with): the tags after
    /// sorting 4000 arrays of 0..=130 elements with 1 to 9 distinct keys, hashed. The reference:
    ///
    /// ```text
    /// uint32_t h = 2166136261u;
    /// for (uint32_t s = 0; s < 4000; s++) {
    ///   uint32_t x = s * 2654435761u + 1; size_t n = s % 131; uint32_t keys = 1 + s % 9;
    ///   std::vector<std::pair<uint32_t, uint32_t>> v;
    ///   for (uint32_t i = 0; i < n; i++) { x = x * 1664525u + 1013904223u; v.push_back({(x >> 16) % keys, i}); }
    ///   std::sort(v.begin(), v.end(), [](auto &a, auto &b) { return a.first < b.first; });
    ///   for (auto &e : v) h = (h ^ e.second) * 16777619u;
    /// }
    /// ```
    #[test]
    fn libstdcxx_tie_order() {
        let mut h = 2166136261u32;
        for s in 0..4000u32 {
            let mut v = case(s.wrapping_mul(2654435761).wrapping_add(1), (s % 131) as usize, 1 + s % 9);
            sort(&mut v, Flavor::Libstdcxx, by_key);
            assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
            fnv(&mut h, &v);
        }
        assert_eq!(h, LIBSTDCXX_HASH);
    }

    /// The heap sort introsort falls back to, on its own against g++ 13's
    /// `std::partial_sort(first, last, last)` over the same arrays.
    #[test]
    fn libstdcxx_heap_sort() {
        let mut h = 2166136261u32;
        for s in 0..4000u32 {
            let mut v = case(s.wrapping_mul(2654435761).wrapping_add(1), (s % 131) as usize, 1 + s % 9);
            heap_sort(&mut v, &by_key);
            assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
            fnv(&mut h, &v);
        }
        assert_eq!(h, LIBSTDCXX_HEAP_HASH);
    }

    #[test]
    fn msvc_keeps_ties_in_order() {
        for s in 0..200u32 {
            let mut v = case(s + 7, (s % 33) as usize, 1 + s % 5);
            let mut stable = v.clone();
            stable.sort_by_key(|e| e.0);
            sort(&mut v, Flavor::Msvc, by_key);
            assert_eq!(v, stable);
        }
    }

    const LIBSTDCXX_HASH: u32 = 0x873fadd5;
    const LIBSTDCXX_HEAP_HASH: u32 = 0x5941b85b;
}
