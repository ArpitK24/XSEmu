//! Control-flow analysis for the wave scheduler: dominators, natural loops, and the block
//! order the reconvergence rule is stated in.
//!
//! The scheduler ([`super::exec`]) needs one property of the order: it is a topological
//! order of the CFG without its back edges, in which every natural loop is a contiguous run
//! of positions starting at its header. Then "the lanes at the lowest position run next"
//! is maximal reconvergence: nothing downstream of a merge point runs before every lane
//! that can still reach it (in this loop iteration) has arrived, and nothing after a loop
//! runs before the last lane has left it.
//!
//! Only reducible control flow is accepted (both DXC and this crate produce it from
//! structured HLSL): a retreating edge whose target does not dominate its source is an
//! error, never a guess.

/// The schedule order of a function's blocks.
pub struct Order {
    /// position → block; unreachable blocks are absent
    pub order: Vec<u32>,
    /// block → position, u32::MAX if unreachable
    pub pos: Vec<u32>,
    /// position → the last position of the loop this block heads (its own position for a
    /// block that is not a loop header)
    pub loop_end: Vec<u32>,
    /// number of natural loops
    pub loops: usize,
}

const NONE: u32 = u32::MAX;

/// `succs(b)` lists the successors of block `b` (duplicates are fine); block 0 is the entry.
pub fn order(n: usize, succs: &dyn Fn(u32, &mut Vec<u32>)) -> Result<Order, String> {
    // successor lists in CSR form
    let mut s_first = Vec::with_capacity(n + 1);
    let mut s_list: Vec<u32> = Vec::new();
    let mut tmp = Vec::new();
    for b in 0..n {
        s_first.push(s_list.len() as u32);
        tmp.clear();
        succs(b as u32, &mut tmp);
        for &t in &tmp {
            if t as usize >= n {
                return Err("branch to a block that does not exist".into());
            }
            s_list.push(t);
        }
    }
    s_first.push(s_list.len() as u32);
    let succ = |b: u32| &s_list[s_first[b as usize] as usize..s_first[b as usize + 1] as usize];

    // depth-first search from the entry: reverse post-order
    let mut rpo_of = vec![NONE; n];
    let mut post: Vec<u32> = Vec::with_capacity(n);
    {
        let mut seen = vec![false; n];
        let mut stack: Vec<(u32, u32)> = vec![(0, 0)];
        seen[0] = true;
        while let Some(&mut (b, ref mut k)) = stack.last_mut() {
            let ss = succ(b);
            if (*k as usize) < ss.len() {
                let t = ss[*k as usize];
                *k += 1;
                if !seen[t as usize] {
                    seen[t as usize] = true;
                    stack.push((t, 0));
                }
            } else {
                post.push(b);
                stack.pop();
            }
        }
    }
    let m = post.len();
    let rpo: Vec<u32> = post.iter().rev().copied().collect();
    for (i, &b) in rpo.iter().enumerate() {
        rpo_of[b as usize] = i as u32;
    }

    // predecessor lists (reachable blocks only), CSR
    let mut p_count = vec![0u32; n + 1];
    for &b in &rpo {
        for &t in succ(b) {
            p_count[t as usize + 1] += 1;
        }
    }
    for i in 0..n {
        p_count[i + 1] += p_count[i];
    }
    let p_first = p_count.clone();
    let mut p_list = vec![0u32; s_list.len()];
    let mut fill = p_first.clone();
    for &b in &rpo {
        for &t in succ(b) {
            p_list[fill[t as usize] as usize] = b;
            fill[t as usize] += 1;
        }
    }
    let pred = |b: u32| &p_list[p_first[b as usize] as usize..p_first[b as usize + 1] as usize];

    // immediate dominators (Cooper, Harvey, Kennedy), in reverse post-order numbers
    let mut idom = vec![NONE; m];
    idom[0] = 0;
    loop {
        let mut changed = false;
        for i in 1..m {
            let mut new = NONE;
            for &p in pred(rpo[i]) {
                let mut p = rpo_of[p as usize];
                if idom[p as usize] == NONE {
                    continue;
                }
                if new == NONE {
                    new = p;
                    continue;
                }
                let mut q = new;
                while p != q {
                    while p > q {
                        p = idom[p as usize];
                    }
                    while q > p {
                        q = idom[q as usize];
                    }
                }
                new = p;
            }
            if new != idom[i] {
                idom[i] = new;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    // dominator tree intervals for constant-time dominance tests
    let mut tin = vec![0u32; m];
    let mut tout = vec![0u32; m];
    {
        let mut c_count = vec![0u32; m + 1];
        for i in 1..m {
            c_count[idom[i] as usize + 1] += 1;
        }
        for i in 0..m {
            c_count[i + 1] += c_count[i];
        }
        let c_first = c_count.clone();
        let mut c_list = vec![0u32; m.saturating_sub(1)];
        let mut fill = c_first.clone();
        for i in 1..m {
            let d = idom[i] as usize;
            c_list[fill[d] as usize] = i as u32;
            fill[d] += 1;
        }
        let mut clock = 0u32;
        let mut stack: Vec<(u32, u32)> = vec![(0, c_first[0])];
        tin[0] = 0;
        while let Some(&mut (v, ref mut k)) = stack.last_mut() {
            if *k < c_first[v as usize + 1] {
                let c = c_list[*k as usize];
                *k += 1;
                clock += 1;
                tin[c as usize] = clock;
                stack.push((c, c_first[c as usize]));
            } else {
                tout[v as usize] = clock;
                stack.pop();
            }
        }
    }
    let dominates = |a: u32, b: u32| tin[a as usize] <= tin[b as usize] && tout[b as usize] <= tout[a as usize];

    // natural loops, innermost first (headers in decreasing reverse post-order)
    let mut is_header = vec![false; m];
    for i in 0..m {
        for &t in succ(rpo[i]) {
            let t = rpo_of[t as usize];
            if t as usize <= i {
                if !dominates(t, i as u32) {
                    return Err("irreducible control flow".into());
                }
                is_header[t as usize] = true;
            }
        }
    }
    // loop ids are their header's reverse post-order number; loop_of: innermost loop of a block
    let mut loop_of = vec![NONE; m];
    let mut parent = vec![NONE; m];
    let mut work: Vec<u32> = Vec::new();
    let mut loops = 0usize;
    for h in (0..m).rev() {
        if !is_header[h] {
            continue;
        }
        loops += 1;
        let h = h as u32;
        work.clear();
        for &p in pred(rpo[h as usize]) {
            let p = rpo_of[p as usize];
            if p >= h && dominates(h, p) {
                work.push(p);
            }
        }
        while let Some(x) = work.pop() {
            if x == h {
                continue;
            }
            // the outermost loop found so far that holds x
            let mut top = loop_of[x as usize];
            if top == NONE {
                loop_of[x as usize] = h;
                for &p in pred(rpo[x as usize]) {
                    work.push(rpo_of[p as usize]);
                }
                continue;
            }
            while parent[top as usize] != NONE {
                top = parent[top as usize];
            }
            if top == h {
                continue;
            }
            parent[top as usize] = h;
            for &p in pred(rpo[top as usize]) {
                let p = rpo_of[p as usize];
                // the inner loop's own latches are inside it already
                if !(p >= top && dominates(top, p)) {
                    work.push(p);
                }
            }
        }
        if loop_of[h as usize] == NONE || loop_of[h as usize] != h {
            // the header belongs to its own loop; an enclosing loop links through 'parent'
            loop_of[h as usize] = h;
        }
    }

    // members of each loop in reverse post-order, a nested loop standing where its header is
    const LOOP: u32 = 1 << 31;
    let mut count = vec![0u32; m + 2];
    let slot = |l: u32| if l == NONE { m } else { l as usize };
    for i in 0..m as u32 {
        let l = loop_of[i as usize];
        count[slot(l) + 1] += 1;
        if l == i {
            count[slot(parent[i as usize]) + 1] += 1;
        }
    }
    for i in 0..=m {
        count[i + 1] += count[i];
    }
    let first = count.clone();
    let mut members = vec![0u32; count[m + 1] as usize];
    let mut fill = first.clone();
    for i in 0..m as u32 {
        let l = loop_of[i as usize];
        if l == i {
            let p = slot(parent[i as usize]);
            members[fill[p] as usize] = LOOP | i;
            fill[p] += 1;
        }
        members[fill[slot(l)] as usize] = i;
        fill[slot(l)] += 1;
    }
    let mut order = Vec::with_capacity(m);
    let mut pos = vec![NONE; n];
    let mut loop_end = vec![0u32; m];
    // (loop slot, next member, position of the header)
    let mut stack: Vec<(usize, u32, u32)> = vec![(m, first[m], NONE)];
    while let Some(&mut (l, ref mut k, head)) = stack.last_mut() {
        if *k < first[l + 1] {
            let x = members[*k as usize];
            *k += 1;
            if x & LOOP != 0 {
                let inner = (x & !LOOP) as usize;
                stack.push((inner, first[inner], order.len() as u32));
            } else {
                pos[rpo[x as usize] as usize] = order.len() as u32;
                loop_end[order.len()] = order.len() as u32;
                order.push(rpo[x as usize]);
            }
        } else {
            if head != NONE {
                loop_end[head as usize] = order.len() as u32 - 1;
            }
            stack.pop();
        }
    }
    Ok(Order { order, pos, loop_end, loops })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(edges: &[(u32, u32)], n: usize) -> Result<Order, String> {
        let e = edges.to_vec();
        order(n, &move |b, out| out.extend(e.iter().filter(|x| x.0 == b).map(|x| x.1)))
    }

    #[test]
    fn diamond_is_topological() {
        let o = run(&[(0, 1), (0, 2), (1, 3), (2, 3)], 4).unwrap();
        assert_eq!(o.pos[0], 0);
        assert_eq!(o.pos[3], 3);
        assert_eq!(o.loops, 0);
    }

    #[test]
    fn loop_is_contiguous_and_exit_follows() {
        // 0 → 1 (header) → 2 → 1, 1 → 3 (exit), 2 → 4 (break) → 3
        let o = run(&[(0, 1), (1, 3), (1, 2), (2, 1), (2, 4), (4, 3)], 5).unwrap();
        assert_eq!(o.loops, 1);
        let h = o.pos[1] as usize;
        assert_eq!(o.loop_end[h], o.pos[2]);
        assert!(o.pos[3] > o.pos[2] && o.pos[4] > o.pos[2]);
        assert!(o.pos[4] < o.pos[3]);
    }

    #[test]
    fn nested_loops_nest() {
        // 0 → 1 (outer) → 2 (inner) → 3 → 2, 3 → 4 → 1, 4 → 5; an exit from the inner loop
        // straight out of both: 2 → 5
        let o = run(&[(0, 1), (1, 2), (2, 3), (3, 2), (3, 4), (4, 1), (4, 5), (2, 5)], 6).unwrap();
        assert_eq!(o.loops, 2);
        let (outer, inner) = (o.pos[1] as usize, o.pos[2] as usize);
        assert_eq!(o.loop_end[outer], o.pos[4]);
        assert_eq!(o.loop_end[inner], o.pos[3]);
        assert_eq!(o.pos[5], 5);
    }

    #[test]
    fn irreducible_is_refused() {
        assert!(run(&[(0, 1), (0, 2), (1, 2), (2, 1), (1, 3)], 4).is_err());
    }

    #[test]
    fn unreachable_blocks_are_dropped() {
        let o = run(&[(0, 2), (1, 2)], 3).unwrap();
        assert_eq!(o.order, vec![0, 2]);
        assert_eq!(o.pos[1], u32::MAX);
    }
}
