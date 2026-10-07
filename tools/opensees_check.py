"""Independent solver check of Plan-to-FEM models with OpenSeesPy (v3: slab load carried by beams and supports;
beam strong axis vertical).

Maps model_fem.json (the pipeline's interchange schema) to an OpenSees 3D model and runs
  (0) a connectivity audit        -> beam ends not connected to any support, floating sub-assemblies
  (1) linear gravity analysis     -> convergence, equilibrium (sum of reactions = applied load)
  (2) modal analysis              -> periods and effective modal mass on the diaphragm DOFs
  (3) lateral-force pattern X / Y -> base-shear share carried by walls, torsion ratio d_max/d_centre

Two connectivity modes:
  'as_exported' : elements are connected only where they share a node in model_fem.json (as an FE
                  package would read the file). Floating sub-assemblies (no element path to the base)
                  are reported and removed so that the rest can be solved.
  'snap'        : a beam end that is not on a support is snapped to the nearest column (point) or wall
                  edge (projection) within `snap_tol` (0.20 m, the tolerance the coherence check uses);
                  wall meshes receive a node at every snapped point. Remaining floating parts are removed.

Idealization (kept close to the ETABS export described in the paper):
  * walls: ShellMITC4 + ElasticMembranePlateSection, each pier meshed into ~0.75 m elements
  * columns, beams: elasticBeamColumn, gross rectangular sections (no cracking factors)
  * slabs: not modelled as elements; one rigid diaphragm per storey (as in the export);
    seismic mass (slab + SDL + 0.3 live + self-weight of the storey) lumped at the master node
  * slab gravity load distributed to the vertical-support nodes of each storey (nearest node, 0.25 m grid)
  * base nodes (z < 1 mm) fully fixed
Units: kN, m, s, t.  Loads: concrete 25 kN/m3; SDL 1.5 kPa; live 2.0 kPa; seismic mass G + 0.3 Q.
"""
from __future__ import annotations
import json, math, sys, time
import numpy as np

G_ACC, GAMMA_C = 9.81, 25.0
SDL, LIVE, PSI2 = 1.5, 2.0, 0.3


def _poly_props(xy):
    x, y = xy[:, 0], xy[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2.0
    if abs(A) < 1e-9:
        return 0.0, xy.mean(0), 0.0
    cx = ((x + x1) * cr).sum() / (6 * A); cy = ((y + y1) * cr).sum() / (6 * A)
    Ixx = abs(((y * y + y * y1 + y1 * y1) * cr).sum() / 12.0)
    Iyy = abs(((x * x + x * x1 + x1 * x1) * cr).sum() / 12.0)
    A = abs(A)
    return A, np.array([cx, cy]), abs(Ixx + Iyy - A * (cx * cx + cy * cy))


def _point_in_poly(px, py, poly):
    x, y = poly[:, 0], poly[:, 1]
    inside = np.zeros(px.shape, bool); j = len(poly) - 1
    for i in range(len(poly)):
        inside ^= ((y[i] > py) != (y[j] > py)) & (px < (x[j] - x[i]) * (py - y[i]) / (y[j] - y[i] + 1e-12) + x[i])
        j = i
    return inside


def _seg_proj(p, a, b):
    ab = b - a; L2 = float(ab @ ab)
    t = 0.0 if L2 < 1e-12 else float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), t, q


def analyse(model_path: str, *, connect='as_exported', snap_tol=0.20, target_el=0.75, elf_coeff=0.10, slab_load='members') -> dict:
    import openseespy.opensees as ops
    t0 = time.time()
    M = json.load(open(model_path))
    out = dict(model=model_path, connect=connect, ok=False)
    mat = M['materials'][0]
    E = mat['E_mpa'] * 1e3; nu = mat.get('poisson', 0.2); Gm = E / (2 * (1 + nu))
    secs = {s['name']: s for s in M.get('column_sections', []) + M.get('beam_sections', [])}
    nodes = {n['id']: np.array([n['x'], n['y'], n['z']], float) for n in M['nodes']}
    levels = sorted({round(s['z_top'], 3) for s in M['stories']})
    H = max(levels)
    fms = M.get('frame_members', [])

    # ---------------- wall geometry
    walls = []
    for wp in M.get('wall_panels', []):
        P = np.array([nodes[n] for n in wp['nodes']])
        zb, zt = P[:, 2].min(), P[:, 2].max()
        bot = P[np.isclose(P[:, 2], zb)]
        if len(bot) != 2: continue
        a, b = bot[0, :2].copy(), bot[1, :2].copy()
        if np.linalg.norm(b - a) < 1e-3 or zt - zb < 1e-3: continue
        walls.append(dict(a=a, b=b, zb=float(zb), zt=float(zt), t=float(wp.get('thickness_m', 0.2)), ids=set(wp['nodes']), extra=[]))
    wall_node_ids = set().union(*[w['ids'] for w in walls]) if walls else set()

    # ---------------- (0) connectivity audit of beam ends
    deg = {}
    for f in fms:
        for n in (f['start_node'], f['end_node']): deg[n] = deg.get(n, 0) + 1
    col_xy = {}
    for f in fms:
        if f['type'] == 'column':
            for n in (f['start_node'], f['end_node']):
                col_xy.setdefault(round(nodes[n][2], 3), []).append((n, nodes[n][:2]))
    beam_ends = [n for f in fms if f['type'] == 'beam' for n in (f['start_node'], f['end_node'])]
    col_nodes = {n for f in fms if f['type'] == 'column' for n in (f['start_node'], f['end_node'])}
    on_support = lambda n: n in col_nodes or n in wall_node_ids
    loose = sorted({n for n in beam_ends if not on_support(n)})
    out['n_beam_ends'] = len(beam_ends)
    out['n_beam_end_nodes_off_support'] = len(loose)
    out['n_dangling_beam_ends'] = sum(1 for n in beam_ends if deg.get(n, 0) == 1 and n not in wall_node_ids)
    moved = {}
    n_snap_col = n_snap_wall = 0
    if connect == 'snap':
        for n in loose:
            p = nodes[n]; z = round(p[2], 3)
            best = (snap_tol + 1e-9, None, None)
            for cid, cxy in col_xy.get(z, []):
                d = float(np.linalg.norm(p[:2] - cxy))
                if d < best[0]: best = (d, ('col', cid), np.r_[cxy, p[2]])
            for w in walls:
                for edge_z in (w['zt'], w['zb']):
                    if abs(edge_z - p[2]) > 1e-3: continue
                    d, t, q = _seg_proj(p[:2], w['a'], w['b'])
                    if d < best[0]: best = (d, ('wall', w, t), np.r_[q, p[2]])
            if best[1] is None: continue
            moved[n] = best[2]
            if best[1][0] == 'col': n_snap_col += 1
            else:
                n_snap_wall += 1
                w, t = best[1][1], best[1][2]
                if 1e-3 < t < 1 - 1e-3: w['extra'].append(t)
        # a point snapped onto the top edge of one storey's pier is also on the bottom edge of the pier above
        for w in walls:
            for w2 in walls:
                if w2 is w or abs(w2['zb'] - w['zt']) > 1e-3: continue
                if np.allclose(w2['a'], w['a'], atol=5e-3) and np.allclose(w2['b'], w['b'], atol=5e-3):
                    w2['extra'] = sorted(set(w2['extra']) | set(w['extra']))
    out['n_snapped_to_column'] = n_snap_col; out['n_snapped_to_wall'] = n_snap_wall
    pos = lambda n: moved.get(n, nodes[n])

    # ---------------- build OpenSees model
    ops.wipe(); ops.model('basic', '-ndm', 3, '-ndf', 6)
    key2tag, xyz = {}, {}
    def node_at(p):
        k = (round(p[0] / 0.005), round(p[1] / 0.005), round(p[2] / 0.005))
        t = key2tag.get(k)
        if t is None:
            t = len(key2tag) + 1; key2tag[k] = t; xyz[t] = np.asarray(p, float)
            ops.node(t, float(p[0]), float(p[1]), float(p[2]))
        return t
    ele_nodes, ele_kind, ele_w, ele_dims = {}, {}, {}, {}
    sec_tags = {}; wall_base = set(); support = set()
    for w in walls:
        L = float(np.linalg.norm(w['b'] - w['a'])); h = w['zt'] - w['zb']
        if w['t'] not in sec_tags:
            sec_tags[w['t']] = len(sec_tags) + 10
            ops.section('ElasticMembranePlateSection', sec_tags[w['t']], E, nu, w['t'], 0.0)
        nh = max(1, math.ceil(L / target_el)); nv = max(1, math.ceil(h / target_el))
        ts = sorted(set([i / nh for i in range(nh + 1)] + list(w['extra'])))
        ts = [t for k, t in enumerate(ts) if k == 0 or t - ts[k - 1] > 0.02 / L] if L > 0 else ts
        if ts[-1] < 1.0: ts[-1] = 1.0
        g = [[node_at(np.r_[w['a'] + (w['b'] - w['a']) * t, w['zb'] + h * j / nv]) for t in ts] for j in range(nv + 1)]
        for j in range(nv):
            for i in range(len(ts) - 1):
                tag = 100000 + len(ele_nodes) + 1
                q4 = (g[j][i], g[j][i + 1], g[j + 1][i + 1], g[j + 1][i])
                ops.element('ShellMITC4', tag, *q4, sec_tags[w['t']])
                ele_nodes[tag] = q4; ele_kind[tag] = 'wall'
                ele_w[tag] = GAMMA_C * w['t'] * L * (ts[i + 1] - ts[i]) * h / nv
        if w['zb'] < 1e-3: wall_base.update(g[0])
        for row in g: support.update(row)
    ops.geomTransf('Linear', 1, 1.0, 0.0, 0.0)
    ops.geomTransf('Linear', 2, 0.0, 0.0, 1.0)
    for f in fms:
        p1, p2 = pos(f['start_node']), pos(f['end_node'])
        s = secs[f['section']]; bw, dd = s['width_m'], s['depth_m']
        # local axes: beams use vecxz = global Z, so local z is vertical and vertical bending is about local y;
        # Iy therefore takes the strong-axis value b*d^3/12 (depth d vertical). Columns are square.
        A = bw * dd; Iy = bw * dd ** 3 / 12; Iz = dd * bw ** 3 / 12
        aa, bb = max(bw, dd), min(bw, dd)
        J = aa * bb ** 3 * (1 / 3 - 0.21 * bb / aa * (1 - bb ** 4 / (12 * aa ** 4)))
        i1, i2 = node_at(p1), node_at(p2)
        if i1 == i2: continue
        vert = np.linalg.norm(p2[:2] - p1[:2]) < 1e-3
        tag = 200000 + len(ele_nodes) + 1
        ops.element('elasticBeamColumn', tag, i1, i2, A, E, Gm, J, Iy, Iz, 1 if vert else 2)
        ele_dims[tag] = (bw, dd)
        ele_nodes[tag] = (i1, i2); ele_kind[tag] = f['type']; ele_w[tag] = GAMMA_C * A * float(np.linalg.norm(p2 - p1))
        if f['type'] == 'column': support.update((i1, i2))

    # ---------------- remove floating sub-assemblies (no element path to the base)
    par = {t: t for t in xyz}
    def fnd(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for en in ele_nodes.values():
        for k in en[1:]: par[fnd(en[0])] = fnd(k)
    based = {fnd(t) for t, p in xyz.items() if p[2] < 1e-3}
    float_ele = [e for e, en in ele_nodes.items() if fnd(en[0]) not in based]
    out['n_float_members'] = sum(1 for e in float_ele if ele_kind[e] in ('beam', 'column'))
    out['n_float_beams'] = sum(1 for e in float_ele if ele_kind[e] == 'beam')
    out['n_beams'] = sum(1 for f in fms if f['type'] == 'beam')
    out['float_frac'] = out['n_float_beams'] / max(out['n_beams'], 1)
    out['float_weight_kN'] = sum(ele_w[e] for e in float_ele)
    for e in float_ele:
        ops.remove('element', e); ele_nodes.pop(e)
    used = {t for en in ele_nodes.values() for t in en}
    for t in [t for t in list(xyz) if t not in used]:
        ops.remove('node', t); xyz.pop(t)
    support &= set(xyz); wall_base &= set(xyz)

    # ---------------- gravity loads (self-weight to element nodes)
    load = {}
    def add(t, w): load[t] = load.get(t, 0.0) + w
    for e, en in ele_nodes.items():
        for t in en: add(t, ele_w[e] / len(en))
    W_wall = sum(ele_w[e] for e in ele_nodes if ele_kind[e] == 'wall')
    W_frame = sum(ele_w[e] for e in ele_nodes if ele_kind[e] != 'wall')
    base = [t for t, p in xyz.items() if p[2] < 1e-3]
    for t in base: ops.fix(t, 1, 1, 1, 1, 1, 1)

    # ---------------- diaphragms, lumped mass, slab gravity load
    slabs = {}
    for sl in M.get('slabs', []):
        P = np.array([nodes[n] for n in sl['nodes']])
        slabs.setdefault(round(P[:, 2].mean(), 3), []).append(P[:, :2])
    slab_t = M['slab_sections'][0]['thickness_m'] if M.get('slab_sections') else 0.15
    q_mass = GAMMA_C * slab_t + SDL + PSI2 * LIVE
    q_grav = GAMMA_C * slab_t + SDL + LIVE
    masters, smass, W_slab, A_floor = {}, {}, 0.0, 0.0
    beam_load = {}; beams_at = {}
    for e, en in ele_nodes.items():
        if ele_kind[e] == 'beam':
            p1, p2 = xyz[en[0]], xyz[en[1]]
            if abs(p1[2] - p2[2]) < 1e-3 and np.linalg.norm(p2[:2] - p1[:2]) > 1e-3:
                beams_at.setdefault(round(float(p1[2]), 3), []).append((e, p1[:2], p2[:2]))
    self_load = dict(load)
    for z in levels:
        lvl = [t for t, p in xyz.items() if abs(p[2] - z) < 1e-3]
        polys = slabs.get(z, [])
        if not lvl or not polys: continue
        props = [_poly_props(pp) for pp in polys]
        At = sum(p[0] for p in props)
        cen = sum(p[0] * p[1] for p in props) / At
        Ip = sum(p[2] + p[0] * float(np.sum((p[1] - cen) ** 2)) for p in props)
        iz = levels.index(z)
        z_lo = levels[iz - 1] if iz > 0 else 0.0
        z_hi = levels[iz + 1] if iz + 1 < len(levels) else z
        w_self = sum(w for t, w in self_load.items() if (z_lo + z) / 2 - 1e-6 < xyz[t][2] <= (z + z_hi) / 2 + 1e-6)
        m = (q_mass * At + w_self) / G_ACC
        mt = 900000 + len(masters) + 1
        ops.node(mt, float(cen[0]), float(cen[1]), float(z)); ops.fix(mt, 0, 0, 1, 1, 1, 0)
        ops.mass(mt, m, m, 0.0, 0.0, 0.0, m * Ip / At)
        ops.rigidDiaphragm(3, mt, *lvl)
        masters[z] = mt; smass[z] = m; A_floor += At
        sup_l = [t for t in lvl if t in support] or lvl
        sxy = np.array([xyz[t][:2] for t in sup_l])
        for poly, (A, _, _) in zip(polys, props):
            xs = np.arange(poly[:, 0].min(), poly[:, 0].max() + 0.25, 0.25)
            ys = np.arange(poly[:, 1].min(), poly[:, 1].max() + 0.25, 0.25)
            X, Y = [v.ravel() for v in np.meshgrid(xs, ys)]
            ins = _point_in_poly(X, Y, poly)
            if not ins.any(): continue
            pts = np.c_[X[ins], Y[ins]]
            idx = np.empty(len(pts), int)
            for k0 in range(0, len(pts), 2000):
                blk = pts[k0:k0 + 2000]
                idx[k0:k0 + 2000] = np.argmin(((blk[:, None, :] - sxy[None]) ** 2).sum(-1), 1)
            q = q_grav * A / len(pts)
            if slab_load == 'members' and beams_at.get(z):
                # each slab sample goes to the nearest load-carrying member: a vertical-support joint or a beam
                bt = beams_at[z]; Aa = np.array([b[1] for b in bt]); Bb = np.array([b[2] for b in bt])
                AB = Bb - Aa; L2 = np.maximum((AB ** 2).sum(1), 1e-12)
                for k0 in range(0, len(pts), 2000):
                    blk = pts[k0:k0 + 2000]
                    ds = np.sqrt(((blk[:, None, :] - sxy[None]) ** 2).sum(-1).min(1))
                    tt = np.clip(((blk[:, None, :] - Aa[None]) * AB[None]).sum(-1) / L2[None], 0, 1)
                    db = np.sqrt((((Aa[None] + tt[..., None] * AB[None]) - blk[:, None, :]) ** 2).sum(-1))
                    jb = db.argmin(1); dbm = db[np.arange(len(blk)), jb]
                    for kk in range(len(blk)):
                        if dbm[kk] < ds[kk] - 1e-9:
                            beam_load[bt[jb[kk]][0]] = beam_load.get(bt[jb[kk]][0], 0.0) + q
                        else:
                            add(sup_l[idx[k0 + kk]], q)
            else:
                for k in idx: add(sup_l[k], q)
            W_slab += q_grav * A
    out.update(n_nodes=len(xyz), n_shell=sum(1 for e in ele_nodes if ele_kind[e] == 'wall'),
               n_frame=sum(1 for e in ele_nodes if ele_kind[e] != 'wall'), n_dof=6 * len(xyz), H=H,
               n_stories=len(levels), floor_area_m2=A_floor, W_gravity_kN=W_wall + W_frame + W_slab)
    out['W_seismic_kN'] = sum(smass.values()) * G_ACC

    def _setup():
        ops.wipeAnalysis()
        ops.constraints('Transformation'); ops.numberer('RCM'); ops.system('UmfPack')
        ops.test('NormDispIncr', 1e-8, 10); ops.algorithm('Linear')

    # ---------------- (1) gravity
    _setup(); ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)
    applied = 0.0
    for t, w in load.items():
        if xyz[t][2] > 1e-3:
            ops.load(t, 0, 0, -w, 0, 0, 0); applied += w
    for e, wtot in beam_load.items():
        en = ele_nodes[e]; Lb = float(np.linalg.norm(xyz[en[1]][:2] - xyz[en[0]][:2]))
        ops.eleLoad('-ele', e, '-type', '-beamUniform', 0.0, -wtot / Lb); applied += wtot
    out['slab_share_on_beams'] = sum(beam_load.values()) / max(W_slab, 1e-9)
    ops.integrator('LoadControl', 1.0); ops.analysis('Static')
    out['gravity_ok'] = ops.analyze(1) == 0
    if out['gravity_ok']:
        ops.reactions()
        Rz = sum(ops.nodeReaction(t, 3) for t in base)
        out['equilibrium_err'] = abs(Rz - applied) / applied
        uz_all = np.array([-1000 * ops.nodeDisp(t, 3) for t in xyz])
        uz_sup = np.array([-1000 * ops.nodeDisp(t, 3) for t in support if xyz[t][2] > 1e-3])
        out['max_uz_support_mm'] = float(uz_sup.max()) if len(uz_sup) else float('nan')
        out['max_uz_mm'] = float(uz_all.max()); out['n_uz_gt25mm'] = int((uz_all > 25).sum())
        # beam mid-span deflection under the slab line load (simply-supported bound on the span term)
        mids, ratios = [], []
        for e, wtot in beam_load.items():
            en = ele_nodes[e]; Lb = float(np.linalg.norm(xyz[en[1]][:2] - xyz[en[0]][:2]))
            bw_, dd_ = ele_dims[e]; Ib = bw_ * dd_ ** 3 / 12
            dm = -500 * (ops.nodeDisp(en[0], 3) + ops.nodeDisp(en[1], 3)) + 1000 * 5 * (wtot / Lb) * Lb ** 4 / (384 * E * Ib)
            mids.append(dm); ratios.append(Lb * 1000 / max(dm, 1e-9))
        out['max_beam_mid_mm'] = float(max(mids)) if mids else 0.0
        out['min_span_over_defl'] = float(min(ratios)) if ratios else float('nan')
        out['wall_share_gravity'] = sum(ops.nodeReaction(t, 3) for t in wall_base) / Rz
        tags = list(xyz); worst = np.argsort(-uz_all)[:5]
        inc = {}
        for e, en in ele_nodes.items():
            for t in en: inc.setdefault(t, []).append(ele_kind[e])
        out['worst_uz'] = [(round(float(uz_all[i]), 1), [round(float(c), 2) for c in xyz[tags[i]]], inc.get(tags[i], [])) for i in worst]
    ops.remove('loadPattern', 1); ops.reset()

    # ---------------- (2) modal analysis on condensed diaphragm DOFs (mass lumped at masters only)
    mdofs = [(z, mt, d) for z, mt in masters.items() for d in (1, 2, 6)]
    Fm = np.zeros((len(mdofs), len(mdofs))); ok_all = True
    for j, (_, mtj, dj) in enumerate(mdofs):
        _setup(); ops.timeSeries('Linear', 100 + j); ops.pattern('Plain', 100 + j, 100 + j)
        ops.load(mtj, *[1.0 if k == dj else 0.0 for k in range(1, 7)])
        ops.integrator('LoadControl', 1.0); ops.analysis('Static')
        if ops.analyze(1) != 0: ok_all = False; break
        for i, (_, mti, di) in enumerate(mdofs): Fm[i, j] = ops.nodeDisp(mti, di)
        ops.remove('loadPattern', 100 + j); ops.reset()
    out['eigen_ok'] = False
    if ok_all:
        from scipy.linalg import eigh
        Kc = np.linalg.inv((Fm + Fm.T) / 2)
        Mc = np.diag([smass[z] if d in (1, 2) else ops.nodeMass(mt, 6) for z, mt, d in mdofs])
        lam, phi = eigh(Kc, Mc)
        if lam.min() > 0:
            T = 2 * np.pi / np.sqrt(lam)
            out['T1'], out['T2'], out['T3'] = (float(x) for x in T[:3])
            def eff(dsel):
                r = np.array([1.0 if d == dsel else 0.0 for _, _, d in mdofs])
                Lm = phi.T @ Mc @ r; mm = np.einsum('ji,jk,ki->i', phi, Mc, phi)
                return Lm ** 2 / mm / (r @ Mc @ r)
            ex, ey, er = eff(1), eff(2), eff(6)
            out['mode_dir'] = ''.join('X' if ex[i] >= max(ey[i], er[i]) else 'Y' if ey[i] >= er[i] else 'T' for i in range(3))
            out['mode1_mx'], out['mode1_my'], out['mode1_rz'] = float(ex[0]), float(ey[0]), float(er[0])
            out['T_X'] = float(T[int(np.argmax(ex))]); out['T_Y'] = float(T[int(np.argmax(ey))])
            out['T_rz'] = float(T[int(np.argmax(er))])
            out['eigen_ok'] = True

    # ---------------- (3) lateral force pattern (EN 1998 distribution, V = elf_coeff * W)
    smz = sum(smass[z] * z for z in masters)
    for d, dof in (('X', 1), ('Y', 2)):
        _setup(); ops.timeSeries('Linear', 10 + dof); ops.pattern('Plain', 10 + dof, 10 + dof)
        for z, mt in masters.items():
            F = elf_coeff * out['W_seismic_kN'] * smass[z] * z / smz
            ops.load(mt, F * (dof == 1), F * (dof == 2), 0, 0, 0, 0)
        ops.integrator('LoadControl', 1.0); ops.analysis('Static')
        if ops.analyze(1) == 0:
            ops.reactions()
            V = -sum(ops.nodeReaction(t, dof) for t in base)
            out[f'wall_shear_share_{d}'] = -sum(ops.nodeReaction(t, dof) for t in wall_base) / V
            zr = max(masters); u_c = ops.nodeDisp(masters[zr], dof)
            roof = [t for t, p in xyz.items() if abs(p[2] - zr) < 1e-3]
            u_max = max(abs(ops.nodeDisp(t, dof)) for t in roof)
            out[f'roof_drift_{d}'] = abs(u_c) / H
            out[f'torsion_ratio_{d}'] = u_max / max(abs(u_c), 1e-12)
        ops.remove('loadPattern', 10 + dof); ops.reset()

    out['EC8_T1_walls'] = 0.05 * H ** 0.75
    out['ok'] = bool(out.get('gravity_ok') and out.get('eigen_ok'))
    out['seconds'] = time.time() - t0
    ops.wipe()
    return out


if __name__ == '__main__':
    mode = sys.argv[2] if len(sys.argv) > 2 else 'as_exported'
    print(json.dumps(analyse(sys.argv[1], connect=mode), indent=1, default=float))
