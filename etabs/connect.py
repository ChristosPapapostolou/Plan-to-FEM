"""Joint connectivity for the exported FE model (called from ETABSExporter._connect_joints).

Before this step a beam end counted as supported if a column, a wall edge or another beam lay within 0.20 m
of it, without being *on* a shared joint.  An FE package connects elements only through shared joints, so
such ends were cantilevers, and chains of them formed sub-assemblies with no element path to the base (shown
by an independent OpenSees solve: 16.6 % of beams floating in the median evaluation plan).  Steps:

  1. connect_model  : every dangling beam end (on no column or wall joint, shared with no other member) is
                      moved onto the nearest support within `tol` (column joint > wall edge > other beam).
                      A wall edge receives a joint at the projection point and every pier panel of the
                      vertical stack is divided there; a supporting beam is divided at the projection point.
                      Sub-assemblies still without a path to the base receive posts at free ends and corners.
  2. support_joints : beam-to-beam joints within `tol` of a column or wall joint are merged onto it
                      (typically slab-outline corners a few mm off a wall corner); remaining unsupported joints
                      where the framing changes direction receive a gravity post continued to the base.
  3. limit_spans    : straight unsupported beam runs longer than `max_span` (default 6.0 m, the grid span
                      limit) receive intermediate posts at equal spacing.
  4. drop_orphan_nodes : joints referenced by no element are removed (once, after the last node is created;
                      node ids are issued as N_{len(nodes)+1}).
Counts are returned per step and stored in exp.connect_stats.
"""
from __future__ import annotations
import math, uuid
import numpy as np


def _xyz(exp, nid):
    n = exp.nodes[nid]
    return np.array([n.x, n.y, n.z], float)


def _proj(p, a, b):
    ab = b - a; L2 = float(ab @ ab)
    if L2 < 1e-12:
        return float(np.linalg.norm(p - a)), 0.0, a
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), t, q


def _split_panels_at(exp, q2, merge):
    """Divide every wall panel whose base segment passes through plan point q2 (all storeys)."""
    n_split = 0
    WP = type(exp.wall_panels[0]) if exp.wall_panels else None
    out = []
    for wp in exp.wall_panels:
        bl, br, tr, tl = (_xyz(exp, n) for n in wp.nodes)
        d, t, q = _proj(q2, bl[:2], br[:2])
        L = float(np.linalg.norm(br[:2] - bl[:2]))
        if d > merge or t * L <= merge or (1 - t) * L <= merge:
            out.append(wp); continue
        nb = exp._get_or_create_node(q[0], q[1], bl[2])
        nt = exp._get_or_create_node(q[0], q[1], tl[2])
        common = dict(section=wp.section, thickness_m=wp.thickness_m, is_shear_wall=wp.is_shear_wall,
                      story=wp.story, wall_id=wp.wall_id, rect_id=wp.rect_id)
        out.append(WP(id=wp.id + "a", nodes=[wp.nodes[0], nb.id, nt.id, wp.nodes[3]], **common))
        out.append(WP(id=wp.id + "b", nodes=[nb.id, wp.nodes[1], wp.nodes[2], nt.id], **common))
        n_split += 1
    exp.wall_panels = out
    return n_split


def _components(exp):
    par = {k: k for k in exp.nodes}
    def f(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for fm in exp.frame_members:
        par[f(fm.start_node)] = f(fm.end_node)
    for wp in exp.wall_panels:
        for k in wp.nodes[1:]:
            par[f(wp.nodes[0])] = f(k)
    based = {f(k) for k, n in exp.nodes.items() if n.z <= exp.base_z + 1e-3}
    return f, based


def _post_stack(exp, x, y, top, near: float = 0.005):
    """Gravity post from storey `top` down to the base on one column line.  (x, y) is the exact joint position; storeys that
    already have a column on that line (within `near`, the node-merge tolerance) are skipped."""
    cols = [(fm, _xyz(exp, fm.start_node)) for fm in exp.frame_members if fm.member_type == "column"]
    for fm, p in cols:
        if np.hypot(p[0] - x, p[1] - y) <= near:
            x, y = float(p[0]), float(p[1]); break
    have = {fm.story for fm, p in cols if np.hypot(p[0] - x, p[1] - y) <= near}
    n = 0
    for s_ in range(top, -1, -1):
        if s_ in have: continue
        exp._add_column(x, y, exp.base_z + s_ * exp.floor_h, exp.base_z + (s_ + 1) * exp.floor_h, s_); n += 1
    return n


def connect_model(exp, tol: float = 0.20, merge: float = 0.005) -> dict:
    st = dict(snap_column=0, snap_wall=0, snap_beam=0, wall_splits=0, beam_splits=0, removed_degenerate=0,
              posts_floating=0, floating_after=0)
    FM = type(exp.frame_members[0]) if exp.frame_members else None
    for _pass in range(4):
        changed = 0
        col_nodes = {n for fm in exp.frame_members if fm.member_type == "column" for n in (fm.start_node, fm.end_node)}
        wall_nodes = {n for wp in exp.wall_panels for n in wp.nodes}
        cols_at = {}
        for n in col_nodes:
            p = _xyz(exp, n); cols_at.setdefault(round(p[2], 3), []).append((n, p[:2]))
        deg = {}
        for fm in exp.frame_members:
            for n in (fm.start_node, fm.end_node): deg[n] = deg.get(n, 0) + 1
        for bm in list(exp.frame_members):
            if bm.member_type != "beam" or bm not in exp.frame_members:
                continue
            for attr in ("start_node", "end_node"):
                nid = getattr(bm, attr)
                # only dangling ends: not on a column or wall joint and not shared with any other member
                if nid in col_nodes or nid in wall_nodes or deg.get(nid, 0) > 1:
                    continue
                p = _xyz(exp, nid); z = round(p[2], 3)
                other = bm.end_node if attr == "start_node" else bm.start_node
                target = None
                # (a) column joint
                best = (tol + 1e-9, None)
                for cn, cxy in cols_at.get(z, []):
                    d = float(np.linalg.norm(p[:2] - cxy))
                    if d < best[0] and cn != other: best = (d, cn)
                if best[1] is not None:
                    target = best[1]; st["snap_column"] += 1
                # (b) wall edge (top edge of the pier below or bottom edge of the pier above)
                if target is None:
                    bw = (tol + 1e-9, None)
                    for wp in exp.wall_panels:
                        bl, br, tr, tl = (_xyz(exp, n) for n in wp.nodes)
                        if abs(tl[2] - p[2]) > 1e-3 and abs(bl[2] - p[2]) > 1e-3:
                            continue
                        d, t, q = _proj(p[:2], bl[:2], br[:2])
                        if d < bw[0]: bw = (d, (wp, q))
                    if bw[1] is not None:
                        wp, q = bw[1]
                        st["wall_splits"] += _split_panels_at(exp, q, merge)
                        target = exp._get_or_create_node(q[0], q[1], p[2]).id
                        wall_nodes = {n for w in exp.wall_panels for n in w.nodes}
                        st["snap_wall"] += 1
                # (c) another beam at the same level
                if target is None:
                    bb = (tol + 1e-9, None)
                    for ob in exp.frame_members:
                        if ob is bm or ob.member_type != "beam" or nid in (ob.start_node, ob.end_node):
                            continue
                        a, b = _xyz(exp, ob.start_node), _xyz(exp, ob.end_node)
                        if abs(a[2] - p[2]) > 1e-3: continue
                        d, t, q = _proj(p[:2], a[:2], b[:2])
                        if d < bb[0]: bb = (d, (ob, a, b, q))
                    if bb[1] is not None:
                        ob, a, b, q = bb[1]
                        if np.linalg.norm(q - a[:2]) <= merge: target = ob.start_node
                        elif np.linalg.norm(q - b[:2]) <= merge: target = ob.end_node
                        else:
                            nq = exp._get_or_create_node(q[0], q[1], p[2]).id
                            exp.frame_members.append(FM(id=ob.id + "s", start_node=nq, end_node=ob.end_node,
                                                        section=ob.section, member_type="beam", story=ob.story))
                            ob.end_node = nq; target = nq; st["beam_splits"] += 1
                        st["snap_beam"] += 1
                if target is not None and target != nid:
                    setattr(bm, attr, target); changed += 1
                    deg[target] = deg.get(target, 0) + 1; deg[nid] = deg.get(nid, 1) - 1
            if bm.start_node == bm.end_node and bm in exp.frame_members:
                exp.frame_members.remove(bm); st["removed_degenerate"] += 1
        if not changed:
            break

    # ---- posts under sub-assemblies that still have no path to the base
    exp._in_repair = True
    for _it in range(exp.n_stories + 1):
        f, based = _components(exp)
        floating = [fm for fm in exp.frame_members if f(fm.start_node) not in based]
        if not floating:
            break
        inc = {}
        for fm in floating:
            for a, b in ((fm.start_node, fm.end_node), (fm.end_node, fm.start_node)):
                inc.setdefault(a, []).append(_xyz(exp, b) - _xyz(exp, a))
        for nid, dirs in inc.items():
            p = _xyz(exp, nid)
            corner = len(dirs) == 1 or any(abs(float(np.cross(d1[:2], d2[:2]))) > 1e-3 * np.linalg.norm(d1) * np.linalg.norm(d2)
                                            for i, d1 in enumerate(dirs) for d2 in dirs[i + 1:])
            if not corner or p[2] <= exp.base_z + 1e-3:
                continue
            story = int(round((p[2] - exp.base_z) / exp.floor_h)) - 1
            if story < 0: continue
            z_top = exp.base_z + (story + 1) * exp.floor_h
            exp._add_column(float(p[0]), float(p[1]), z_top - exp.floor_h, z_top, story)
            st["posts_floating"] += 1
    exp._in_repair = False
    f, based = _components(exp)
    st["floating_after"] = sum(1 for fm in exp.frame_members if f(fm.start_node) not in based)
    exp.connect_stats = st
    return st


def drop_orphan_nodes(exp):
    """Remove joints no element references (old beam-end nodes left behind by snapping).
    Call only once, after the last node has been created: node ids are issued as N_{len(nodes)+1}."""
    used = {n for fm in exp.frame_members for n in (fm.start_node, fm.end_node)}
    used |= {n for wp in exp.wall_panels for n in wp.nodes}
    used |= {n for sl in getattr(exp, "slabs", []) for n in sl.nodes}
    gone = [n for n in exp.nodes if n not in used]
    for nid in gone:
        del exp.nodes[nid]
    return len(gone)


def support_joints(exp, tol: float = 0.20, merge: float = 0.005) -> dict:
    """Step 3: beam-to-beam joints that carry no vertical support.

    (a) merge : a beam joint that is not a column or wall joint but lies within `tol` of one (typically a
                slab-outline corner a few millimetres off a wall corner) is moved onto it; every beam and slab
                that references the joint is re-pointed.  A wall edge receives a joint at the projection point
                and the pier stack is divided there.
    (b) posts : a remaining unsupported joint at which the incident beams change direction (an outline corner,
                an L-joint) is a hinge-like point of the floor framing; it receives a gravity post continued to
                the base.  Straight joints and T-joints (a secondary beam framing into a continuous primary)
                are kept as beam-on-beam joints.
    """
    st = dict(merge_column=0, merge_wall=0, wall_splits=0, posts_corner=0)
    def refs():
        col = {n for fm in exp.frame_members if fm.member_type == "column" for n in (fm.start_node, fm.end_node)}
        wall = {n for wp in exp.wall_panels for n in wp.nodes}
        return col, wall
    def repoint(old, new):
        for fm in exp.frame_members:
            if fm.start_node == old: fm.start_node = new
            if fm.end_node == old: fm.end_node = new
        for sl in getattr(exp, "slabs", []):
            sl.nodes = [new if n == old else n for n in sl.nodes]
        exp.frame_members[:] = [fm for fm in exp.frame_members if fm.start_node != fm.end_node]
    col, wall = refs()
    joints = sorted({n for fm in exp.frame_members if fm.member_type == "beam" for n in (fm.start_node, fm.end_node)}
                    - col - wall)
    for nid in joints:
        if nid not in exp.nodes: continue
        p = _xyz(exp, nid)
        best = (tol + 1e-9, None)
        for cn in col:
            c = _xyz(exp, cn)
            if abs(c[2] - p[2]) > 1e-3: continue
            d = float(np.linalg.norm(p[:2] - c[:2]))
            if d < best[0]: best = (d, cn)
        if best[1] is not None:
            repoint(nid, best[1]); st["merge_column"] += 1; continue
        bw = (tol + 1e-9, None)
        for wp in exp.wall_panels:
            bl, br, tr, tl = (_xyz(exp, n) for n in wp.nodes)
            if abs(tl[2] - p[2]) > 1e-3 and abs(bl[2] - p[2]) > 1e-3: continue
            d, t, q = _proj(p[:2], bl[:2], br[:2])
            if d < bw[0]: bw = (d, q)
        if bw[1] is not None:
            q = bw[1]
            st["wall_splits"] += _split_panels_at(exp, q, merge)
            tgt = exp._get_or_create_node(q[0], q[1], p[2]).id
            if tgt != nid:
                repoint(nid, tgt)
            st["merge_wall"] += 1
            col, wall = refs()
    # (b) posts under unsupported direction-change joints
    col, wall = refs()
    inc = {}
    for fm in exp.frame_members:
        if fm.member_type != "beam": continue
        for a, b in ((fm.start_node, fm.end_node), (fm.end_node, fm.start_node)):
            if a in col or a in wall: continue
            v = _xyz(exp, b) - _xyz(exp, a); v = v[:2] / (np.linalg.norm(v[:2]) + 1e-12)
            inc.setdefault(a, []).append(v)
    stacks = {}
    for nid, dirs in inc.items():
        # a continuous primary through the joint = a pair of opposite directions
        through = any(float(d1 @ d2) < -0.985 for i, d1 in enumerate(dirs) for d2 in dirs[i + 1:])
        if through: continue
        p = _xyz(exp, nid)
        story = int(round((p[2] - exp.base_z) / exp.floor_h)) - 1
        if story < 0: continue
        key = (round(p[0], 3), round(p[1], 3))
        prev = stacks.get(key)
        stacks[key] = (max(prev[0], story) if prev else story, float(p[0]), float(p[1]))
    exp._in_repair = True
    for top, x, y in stacks.values():
        _post_stack(exp, x, y, top)
        st["posts_corner"] += 1
    exp._in_repair = False
    return st


def limit_spans(exp, max_span: float = 6.0, near: float = 1.0) -> dict:
    """Step 4: unsupported beam runs longer than `max_span`.

    A run is a straight chain of beams through joints without a column or wall (beam-on-beam joints).  A run
    longer than `max_span` (6.0 m, the span limit used by the structural grid and by the slab-span check)
    receives intermediate gravity posts at equal spacing <= max_span, re-using an existing joint within
    `near` of each target point or dividing the beam there.  Posts are continued down to the base.
    """
    st = dict(long_runs=0, posts_span=0, beam_splits=0)
    FM = type(exp.frame_members[0])
    col = {n for fm in exp.frame_members if fm.member_type == "column" for n in (fm.start_node, fm.end_node)}
    wall = {n for wp in exp.wall_panels for n in wp.nodes}
    sup = col | wall
    beams = [fm for fm in exp.frame_members if fm.member_type == "beam"]
    inc = {}
    for bm in beams:
        inc.setdefault(bm.start_node, []).append(bm); inc.setdefault(bm.end_node, []).append(bm)
    def unit(a, b):
        v = _xyz(exp, b)[:2] - _xyz(exp, a)[:2]; return v / (np.linalg.norm(v) + 1e-12)
    seen = set(); posts = {}
    for bm in beams:
        if id(bm) in seen: continue
        seen.add(id(bm))
        nodes = [bm.start_node, bm.end_node]; members = [bm]
        for side in (0, 1):
            while True:
                end = nodes[-1] if side else nodes[0]
                prev = nodes[-2] if side else nodes[1]
                if end in sup: break
                d = unit(prev, end); nxt = None
                for ob in inc.get(end, []):
                    if id(ob) in seen: continue
                    far = ob.end_node if ob.start_node == end else ob.start_node
                    if float(unit(end, far) @ d) > 0.985: nxt = (ob, far); break
                if nxt is None: break
                seen.add(id(nxt[0])); members.append(nxt[0])
                if side: nodes.append(nxt[1])
                else: nodes.insert(0, nxt[1])
        P = np.array([_xyz(exp, n) for n in nodes])
        seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]; L = float(s[-1])
        if L <= max_span + 1e-6: continue
        st["long_runs"] += 1
        k = int(math.ceil(L / max_span))
        for i in range(1, k):
            si = L * i / k
            j = int(np.argmin(np.abs(s - si)))
            if 0 < j < len(nodes) - 1 and abs(s[j] - si) <= near:
                q = P[j]
            else:
                q = None
                for jj in range(len(seg)):
                    if s[jj] - 1e-9 <= si <= s[jj + 1] + 1e-9:
                        t = (si - s[jj]) / max(seg[jj], 1e-9); q = P[jj] + t * (P[jj + 1] - P[jj]); break
                a, b = nodes[jj], nodes[jj + 1]
                mb = next((m for m in exp.frame_members if m.member_type == "beam" and {m.start_node, m.end_node} == {a, b}), None)
                nq = exp._get_or_create_node(q[0], q[1], q[2]).id
                if mb is not None and nq not in (a, b):
                    exp.frame_members.append(FM(id=mb.id + "p", start_node=nq, end_node=mb.end_node,
                                                section=mb.section, member_type="beam", story=mb.story))
                    mb.end_node = nq; st["beam_splits"] += 1
                    nodes.insert(jj + 1, nq); P = np.insert(P, jj + 1, q, axis=0)
                    seg = np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
            story = int(round((q[2] - exp.base_z) / exp.floor_h)) - 1
            if story < 0: continue
            key = (round(float(q[0]), 3), round(float(q[1]), 3))
            prev = posts.get(key)
            posts[key] = (max(prev[0], story) if prev else story, float(q[0]), float(q[1]))
    exp._in_repair = True
    for top, x, y in posts.values():
        _post_stack(exp, x, y, top)
        st["posts_span"] += 1
    exp._in_repair = False
    return st
