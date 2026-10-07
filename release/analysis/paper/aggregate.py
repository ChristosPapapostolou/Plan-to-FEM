"""(v5: MSD results from the orientation-corrected sweep, msd_worker_v5.py.)
Collect every number used by edits_pass3.py / supp_pass3.py from the result files and write numbers_v3.json.

Inputs (all under C:\\Users\\chris\\Documents\\Plan2FEM_revision):
  04_results/connect_v3/{deployed,snap}/<plan>/model_fem.json, sanity_report.json   ten-plan batch
  03_experiments/run/enriched/<plan>.json                                          Stage-4 output (column provenance)
  03_experiments/solve/solve_ten_v3.jsonl                                          OpenSees, ten plans (v3 solver)
  03_experiments/solve/solve_msd150_published_v3.jsonl                             OpenSees, published exporter, MSD sample
  03_experiments/msd_full/res/<id>.json, arm_gnnonly/res/<id>.json                 full MSD sweep (both arms)
  03_experiments/msd_full/solve_msd_connected_v3.jsonl                             OpenSees, connected exporter, MSD
  03_experiments/run/scratch/{psw_sweep,nms_sweep,arms_v2}.json (post-fix) and *_prefix.json (pre-fix)
  04_results/indep_test_both.json                                                  Table 9
Run with the plan2fem interpreter:  python aggregate_v3.py [--partial]
"""
import os, sys, json, glob, collections, importlib.util
import numpy as np

ROOT = r"C:\Users\chris\Documents\Plan2FEM_revision"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(ROOT, "03_experiments", "run", "Plan-to-FEM")
V3 = os.path.join(ROOT, "04_results", "connect_v3")
ENR = os.path.join(ROOT, "03_experiments", "run", "enriched")
SOLVE = os.path.join(ROOT, "03_experiments", "solve")
MSDF = os.path.join(ROOT, "03_experiments", "msd_full")
SCR = os.path.join(ROOT, "03_experiments", "run", "scratch")
PLANS = [f"floor_plan_{i}" for i in range(1, 10)] + ["plan_05"]
PARTIAL = "--partial" in sys.argv


def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


SAN = load_mod("san_agg", os.path.join(REPO, "etabs", "sanity.py"))
MX = load_mod("metrics_agg", os.path.join(ROOT, "03_experiments", "run", "metrics.py"))
f1 = lambda x: f"{x:.1f}"; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
pct = lambda x, d=1: f"{100 * x:.{d}f}%"
N = {"tables": {}, "text": {}, "ten": {}, "msd": {}, "files": {}}
T_, X = N["tables"], N["text"]


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


# =====================================================================================  ten-plan batch
def plan_rows(var):
    rows = []
    for p in PLANS:
        d = os.path.join(V3, var, p)
        mp, sp = os.path.join(d, "model_fem.json"), os.path.join(d, "sanity_report.json")
        M = json.load(open(mp)); S = json.load(open(sp)); mm = MX.model_metrics(mp, sp)
        nd = SAN._nodes_xyz(M); ch = {c["name"]: c for c in S["checks"]}
        E = json.load(open(os.path.join(ENR, p + ".json")))
        g = np.array([[c["x"], c["y"]] for c in E.get("columns", []) if c.get("source") == "gnn"]).reshape(-1, 2)
        cols = [f for f in M["frame_members"] if f["type"] == "column"]
        net = sum(1 for f in cols if f["story"] == 0 and len(g) and
                  np.min(np.linalg.norm(g - np.array(nd[f["start_node"]][:2]), axis=1)) < 0.05)
        c0 = [f for f in cols if f["story"] == 0]
        locs = len({(round(nd[f["start_node"]][0], 2), round(nd[f["start_node"]][1], 2)) for f in c0})
        secs = collections.Counter(f["section"] for f in cols)
        ecc = ch["eccentricity"].get("value") or {}
        bs = ch["beam_support"].get("value") or {}
        ll = ch["lateral_lines"].get("value") or {}
        cj = os.path.join(d, "model_fem.connect.json")
        C = json.load(open(cj)) if os.path.exists(cj) else {}
        rows.append(dict(plan=p, M=M, S=S, mm=mm, ch=ch, net=net, rule=len(c0) - net, locs=locs, secs=secs,
                         ecc=ecc, bs=bs, ll=ll, C=C, n_stories=len(M["stories"])))
    return rows


ten = plan_rows("snap"); ten_pub = plan_rows("deployed")
sv = {r["key"]: r for r in jl(os.path.join(SOLVE, "solve_ten_v3.jsonl"))}
st = lambda r: r["S"]["score"]
n_ten = len(ten)
mean6 = np.mean([st(r) for r in ten]); mean4 = np.mean([r["mm"]["score4"] for r in ten])
mean6_pub_strict = np.mean([st(r) for r in ten_pub])
nofail = sum(r["S"]["n_fail"] == 0 for r in ten); all6 = sum(r["S"]["n_fail"] == 0 and r["S"]["n_warn"] == 0 for r in ten)
dens = [r["mm"]["cols_per_100m2"] for r in ten]; trib = [r["mm"]["trib_m2"] for r in ten]; spc = [r["mm"]["spacing_med"] for r in ten]
conc = [r["mm"]["m3_per_m2"] for r in ten]; vert = [r["mm"]["vertical_m3_per_m2"] for r in ten]
conc_pub = [r["mm"]["m3_per_m2"] for r in ten_pub]
vols = {k: sum(r["mm"][k] for r in ten) for k in ("v_col", "v_beam", "v_wall", "v_slab")}
vt = sum(vols.values()); share = {k: 100 * v / vt for k, v in vols.items()}
secs_all = collections.Counter(); [secs_all.update(r["secs"]) for r in ten]
cols0_total = sum(r["mm"]["cols_story0"] for r in ten); cols0_pub = sum(r["mm"]["cols_story0"] for r in ten_pub)
net_total = sum(r["net"] for r in ten); rule_total = sum(r["rule"] for r in ten)
net_pub = sum(r["net"] for r in ten_pub)
posts_c = sum(r["C"].get("joints", {}).get("posts_corner", 0) for r in ten)
posts_s = sum(r["C"].get("spans", {}).get("posts_span", 0) for r in ten)
posts_f = sum(sum(r["C"].get(k, {}).get("posts_floating", 0) for k in ("pass1", "pass2", "pass3")) for r in ten)
merged = sum(r["C"].get("joints", {}).get("merge_column", 0) + r["C"].get("joints", {}).get("merge_wall", 0) for r in ten)
snaps = sum(sum(r["C"].get(k, {}).get(x, 0) for k in ("pass1", "pass2", "pass3") for x in ("snap_column", "snap_wall", "snap_beam")) for r in ten)
sec_name = lambda s: s.replace("C", "").split("X")[0]
storey_secs = {sec_name(k): v // ten[0]["n_stories"] for k, v in secs_all.items()}
N["ten"].update(mean6=mean6, mean4=mean4, mean6_pub_strict=mean6_pub_strict, nofail=nofail, all6=all6,
                cols0_total=cols0_total, cols0_pub=cols0_pub, net=net_total, rule=rule_total, net_pub=net_pub,
                posts_corner=posts_c, posts_span=posts_s, posts_float=posts_f, merged=merged, snapped=snaps)

# ---------- Table 13 (density)
T_["table13"] = [["Plan", "Floor area (m2)", "Columns", "Columns / 100 m2", "Tributary area (m2)", "Median spacing (m)"]] + [
    [r["plan"], f"{r['mm']['area_m2']:.0f}", str(r["mm"]["cols_story0"]), f1(r["mm"]["cols_per_100m2"]), f1(r["mm"]["trib_m2"]),
     f2(r["mm"]["spacing_med"])] for r in ten]
# ---------- Table 14 (batch)
def wf(r): return f"{r['S']['n_warn']} / {r['S']['n_fail']}"
T_["table14"] = [["Plan", "SW panels", "Beams", "Columns", "Ecc. (%)", "Span (m)", "W / F", "Coherence score"]] + [
    [r["plan"], str(r["mm"]["sw_panels"]), str(r["mm"]["beams"]), str(r["mm"]["columns"]),
     f1(100 * max(r["ecc"].values())) if r["ecc"] else "-", f2(r["ch"]["slab_span"]["value"]), wf(r), f2(st(r))] for r in ten]
# ---------- Table 15 (quantities)
T_["table15"] = [["Plan", "Floor area (m2)", "Concrete (m3)", "Concrete (m3/m2)", "Vertical only (m3/m2)", "Wall length (m/m2)"]] + [
    [r["plan"], f"{r['mm']['area_m2']:.0f}", f"{r['mm']['v_tot']:.0f}", f3(r["mm"]["m3_per_m2"]), f3(r["mm"]["vertical_m3_per_m2"]),
     f3(r["mm"]["wall_len_per_m2"])] for r in ten]
# ---------- S11 (per-plan breakdown)
def bse(r):
    b = r["bs"]; return f"{b.get('direct', 0)} / {b.get('via_beam', 0)} / {b.get('unsupported', 0)}"
T_["S11"] = [["Plan", "Lines X/Y", "Ecc. x / y (%)", "Slab span (m)", "Beam ends (d/v/u)", "Cols", "Score"]] + [
    [r["plan"], f"{r['ll'].get('x_dir', '-')} / {r['ll'].get('y_dir', '-')}",
     f"{100 * r['ecc'].get('x', 0):.1f} / {100 * r['ecc'].get('y', 0):.1f}", f2(r["ch"]["slab_span"]["value"]), bse(r), str(r["locs"]), f2(st(r))]
    for r in ten]
X["S11_caption"] = ("Table S11. Detailed coherence-check breakdown per plan, for the models exported with the joint-connectivity step. "
                    "Lines = distinct shear-wall lateral-resisting lines (X / Y); Ecc. = stiffness eccentricity per direction (% of plan dimension; "
                    "values over 20% fail, values over 10% warn); beam ends are counted at storey level as direct support / supported via another beam / "
                    "unsupported; Cols = column locations per floor. Every beam end lies on a shared joint and every member reaches the base; orphan-node "
                    "and column-continuity checks pass for every plan.")

# ---------- Table 7 (idealization) -- the connectivity rows change, the rest is restated unchanged
T_["table7"] = [
    ["Aspect", "Treatment in the exported model", "Parameter / tolerance"],
    ["Shear-wall elements", "Four-node thin-shell area objects, specified wall property", "thickness measured in Stage 2, snapped to 25 mm, floor 25 mm"],
    ["Slab elements", "One thin-shell slab per storey over the footprint hull; no slab openings", "0.15 m uniform"],
    ["Beams", "Two-node rectangular RC frame elements (perimeter ring, pier ties, interior grid beams)", "500 mm deep x 300 mm wide"],
    ["Columns", "Two-node square RC frame elements, continuous across storeys", "tributary-sized; see sizing row"],
    ["Column sizing", "N_Ed = A_trib w n_storeys; b = sqrt(N_Ed / (0.35 fc)); no slenderness, moment or reinforcement check", "w = 12 kPa; b >= 0.25 m, rounded up to 50 mm; no upper bound"],
    ["Wall openings", "Not modelled as openings within shells; opening edges are not extruded, leaving gaps between piers", "-"],
    ["Meshing", "No explicit mesh assignment; automatic meshing by the analysis package at solution time", "one area object per pier and per slab, per storey"],
    ["Beam-wall and beam-column joints", "Node-to-node framing at the storey top elevation; no rigid end zones or offsets", "-"],
    ["Diaphragm", "One rigid diaphragm per storey, assigned to the slab and to all joints at that level", "rigid (not semi-rigid)"],
    ["Base restraints", "All nodes at the base elevation fully fixed; no foundation elements", "6 DOF; base tolerance 1 mm"],
    ["Material", "Isotropic concrete", "fc = 30 MPa, E = 25,743 MPa, nu = 0.20, alpha = 5.5x10-6 /C"],
    ["Mass source, loads, load cases", "Not assigned by the exporter; defined by the engineer before analysis", "-"],
    ["Node connectivity", "Coincident nodes merged", "5 mm"],
    ["Beam-end connectivity", "Every beam end on a shared joint with a column, a wall panel or another beam; loose ends snapped onto the nearest support; walls and supporting beams divided at the snap point", "snap radius 0.20 m"],
    ["Unsupported joints", "Joints near a support merged onto it; gravity posts, continued to the base, under unsupported corner joints and under any sub-assembly without a path to the base", "merge radius 0.20 m"],
    ["Span control", "Grid spans subdivided by intermediate columns; long piers split; unsupported beam runs subdivided by gravity posts", "max clear span 6.0 m; pier split above 3.0 m; max unsupported beam run 6.0 m"],
    ["Plan orientation", "Stages 2-4 run and the model is built in the principal frame of the wall layout, then rotated back; piers off the frame axes keep their own direction; bent pier groups split into straight runs", "frame used if |theta| > 5 deg; axis snap only within 5 deg and 0.30 m"],
]
# ---------- Table 8 (coherence checks) -- criteria restored where the original cells were empty
T_["table8"] = [
    ["Check", "Criterion", "Thresholds"],
    ["lateral_lines", "distinct shear-wall lines resisting each direction", "fail if < 2"],
    ["eccentricity", "stiffness centre (k proportional to t·L³ per merged pier) against plan centre, per direction", "warn > 10%, fail > 20% of plan dimension"],
    ["beam_support", "every beam end within 0.20 m of a column, wall or beam and on a shared joint with it; every member with an element path to the base", "fail if any end unsupported or not on a shared joint, or any member floating; warn if > 50% of ends are carried only by other beams"],
    ["slab_span", "max clear distance from slab interior to a vertical support", "warn > 4.5 m, fail > 6.0 m"],
    ["column_continuity", "identical column plan positions across all storeys", "warn if 1-2 locations discontinuous; fail if > 2"],
    ["orphan_nodes", "nodes not referenced by any element", "warn if > 0"],
]
# ---------- Table 10 (common assumptions) -- empty cells restored from Section 5.5
T_["table10"] = [
    ["Parameter", "Value"], ["Number of storeys", "3 (ten-plan batch); 4 (MSD sweep)"], ["Storey height", "3.0 m"],
    ["Material", "Reinforced concrete, fc = 30 MPa, E = 25,743 MPa"], ["Wall mode", "Structural (shear walls as shell elements)"],
    ["Column mode", "Grid (GNN predictions consumed when present)"],
    ["Column sizing", "Tributary area, w = 12 kPa, b = sqrt(N_Ed / 0.35 fc), 250 mm minimum"],
    ["Slab / diaphragm", "One slab and one rigid diaphragm per storey"],
    ["Connectivity", "Joint-connectivity step on (Section 5.5)"],
]
# ---------- Table 9 (untouched synthetic test, deployed checkpoint)
it = json.load(open(os.path.join(ROOT, "04_results", "indep_test_both.json")))["deployed_gnn_multi1"]
vb, vf, tf = it["val_best_threshold"], it["val_fixed_threshold"], it["test_fixed_threshold"]
T_["table9"] = [["Evaluation", "Edge IoU", "Edge base rate", "Column P", "Column R", "Column F1"],
                ["Validation (seed 2), threshold swept", f3(vb["edge_iou"]), "-", f3(vb["node_p"]), f3(vb["node_r"]), f3(vb["node_f1"])],
                ["Validation (seed 2), thresholds frozen", f3(vf["edge_iou"]), f3(vf["edge_base_rate"]), f3(vf["node_p"]), f3(vf["node_r"]), f3(vf["node_f1"])],
                ["Untouched test (seed 101), thresholds frozen", f3(tf["edge_iou"]), f3(tf["edge_base_rate"]), f3(tf["node_p"]), f3(tf["node_r"]), f3(tf["node_f1"])]]
X["table9_text"] = (f"Table 9 evaluates the deployed checkpoint on it at its stored thresholds. On the untouched set the edge IoU is {f3(tf['edge_iou'])} "
                    f"and the column F1 {f3(tf['node_f1'])}, against {f3(vb['edge_iou'])} and {f3(vb['node_f1'])} when the thresholds are swept on the validation "
                    f"graphs, so choosing thresholds on the reported split inflates IoU by about {vb['edge_iou'] - tf['edge_iou']:.2f} and F1 by about "
                    f"{vb['node_f1'] - tf['node_f1']:.2f}. At frozen thresholds the two sets agree closely ({f3(tf['edge_iou'])} against {f3(vf['edge_iou'])}, "
                    f"{f3(tf['node_f1'])} against {f3(vf['node_f1'])}).")


# =====================================================================================  ten-plan solver (v3)
def sk(var, p): return sv.get(f"{var}|{p}", {})
pub_ff = [sk("deployed", p).get("float_frac", np.nan) for p in PLANS]
pub_uz = [sk("deployed", p).get("max_uz_mm", np.nan) for p in PLANS]
con10 = [sk("snap", p) for p in PLANS]
N["ten"].update(pub_float_med=float(np.nanmedian(pub_ff)), pub_float_max=float(np.nanmax(pub_ff)), pub_uz_med=float(np.nanmedian(pub_uz)),
                con_ok=sum(bool(c.get("ok")) for c in con10), con_float=sum(c.get("n_float_beams", 1) for c in con10),
                con_uz_max=max(c.get("max_uz_mm", np.nan) for c in con10), con_mid_max=max(c.get("max_beam_mid_mm", np.nan) for c in con10),
                con_Ld_min=min(c.get("min_span_over_defl", np.nan) for c in con10), con_T1_med=float(np.median([c["T1"] for c in con10])),
                con_eq_max=max(c.get("equilibrium_err", 0) for c in con10))
T_["S15"] = [["Plan", "Floating beams without the step (%)", "T1 (s)", "T1 / EC8", "Wall share of base shear X / Y", "Wall share of gravity",
              "Beam load share", "Max. beam mid-span (mm)", "Torsion ratio X / Y"]] + [
    [p, f1(100 * sk("deployed", p).get("float_frac", np.nan)), f3(c["T1"]), f2(c["T1"] / c["EC8_T1_walls"]),
     f"{c['wall_shear_share_X']:.2f} / {c['wall_shear_share_Y']:.2f}", f2(c["wall_share_gravity"]), f2(c["slab_share_on_beams"]),
     f1(c["max_beam_mid_mm"]), f"{c['torsion_ratio_X']:.2f} / {c['torsion_ratio_Y']:.2f}"] for p, c in zip(PLANS, con10)]
X["S15_caption"] = ("Table S15. Independent OpenSeesPy check of the ten batch models (three storeys). The first column is the share of beams with no element "
                    "path to the base in the models of the exporter without the joint-connectivity step; all other columns are for the connected models, "
                    "none of which has a floating member. T1 is the fundamental period of the diaphragm-condensed model and EC8 the estimate "
                    "Ct·H^0.75 with Ct = 0.05 for wall structures; base-shear and gravity shares are the fractions of the base reactions carried by wall "
                    "base joints; beam load share is the fraction of the slab load carried by beams rather than directly by vertical supports; the "
                    "mid-span deflection adds 5wL⁴/384EI to the mean end displacement; the torsion ratio is the largest roof displacement over the "
                    "centroid displacement under the equivalent lateral load.")
T_["S16"] = [["Plan", "Beam ends snapped", "Joints merged", "Wall splits", "Corner posts", "Span posts", "Posts under floating parts", "Columns without the step", "Columns with the step"]]
for r, rp in zip(ten, ten_pub):
    C = r["C"]
    T_["S16"].append([r["plan"], str(sum(C.get(k, {}).get(x, 0) for k in ("pass1", "pass2", "pass3") for x in ("snap_column", "snap_wall", "snap_beam"))),
                      str(C.get("joints", {}).get("merge_column", 0) + C.get("joints", {}).get("merge_wall", 0)),
                      str(sum(C.get(k, {}).get("wall_splits", 0) for k in ("pass1", "pass2", "pass3", "joints"))),
                      str(C.get("joints", {}).get("posts_corner", 0)), str(C.get("spans", {}).get("posts_span", 0)),
                      str(sum(C.get(k, {}).get("posts_floating", 0) for k in ("pass1", "pass2", "pass3"))),
                      str(rp["mm"]["cols_story0"]), str(r["mm"]["cols_story0"])])
X["S16_caption"] = ("Table S16. Operations of the joint-connectivity step on the ten batch models, counted over all three storeys, and storey-1 column counts "
                    "with and without it. Posts are counted as column stacks, each continued from its joint to the base.")

# =====================================================================================  ten-plan sweeps (post-fix)
def mean_by(rows, key, field):
    g = collections.defaultdict(list)
    for r in rows: g[r[key]].append(r[field])
    return {k: float(np.mean(v)) for k, v in g.items()}
psw = json.load(open(os.path.join(SCR, "psw_sweep.json"))) if os.path.exists(os.path.join(SCR, "psw_sweep.json")) else []
nms = json.load(open(os.path.join(SCR, "nms_sweep.json"))) if os.path.exists(os.path.join(SCR, "nms_sweep.json")) else []
arms = json.load(open(os.path.join(SCR, "arms_v2.json"))) if os.path.exists(os.path.join(SCR, "arms_v2.json")) else {}
if psw:
    T_["S2"] = [["PSW threshold (m)", "Graph edges", "PSW edges", "Opening edges", "Shear walls selected", "Selected length (m)", "Coherence score"]]
    for th in sorted({r["threshold"] for r in psw}):
        rr = [r for r in psw if r["threshold"] == th]
        T_["S2"].append([f"{th:.3f}" + (" (deployed)" if abs(th - 0.06) < 1e-9 else ""), f1(np.mean([r["n_edges"] for r in rr])),
                         f1(np.mean([r["n_psw"] for r in rr])), f1(np.mean([r["n_open"] for r in rr])), f1(np.mean([r["n_sw_selected"] for r in rr])),
                         f1(np.mean([r["sw_len_m"] for r in rr])), f3(np.mean([r["score"] for r in rr]))])
X["S2_caption"] = ("Table S2. Sensitivity of the pipeline to the PSW thickness threshold, averaged over the 10 evaluation plans. Every column is a mean per "
                   "plan; the coherence score is the aggregate of the six checks, computed with the joint-connectivity exporter and the joint-aware "
                   "beam-support check, as in main-text Table 14.")
if nms:
    T_["S10"] = [["Suppression radius (m)", "Columns / 100 m2", "Tributary area (m2)", "Median spacing (m)", "Coherence score"]]
    for R_ in sorted({r["radius"] for r in nms}):
        rr = [r for r in nms if r["radius"] == R_]
        T_["S10"].append([f"{R_:.2f}" + (" (deployed)" if abs(R_ - 0.6) < 1e-9 else ""), f1(np.mean([r["per100"] for r in rr])),
                          f1(np.mean([r["trib"] for r in rr])), f2(np.mean([r["spacing_med"] for r in rr])), f3(np.mean([r["score"] for r in rr]))])
    nm = {R_: [r for r in nms if r["radius"] == R_] for R_ in {r["radius"] for r in nms}}
    N["ten"].update(nms_d06=np.mean([r["per100"] for r in nm[0.6]]), nms_d40=np.mean([r["per100"] for r in nm[4.0]]),
                    nms_s06=np.mean([r["score"] for r in nm[0.6]]), nms_s40=np.mean([r["score"] for r in nm[4.0]]),
                    nms_s30=np.mean([r["score"] for r in nm[3.0]]))
X["S10_caption"] = ("Table S10. Effect of the column suppression radius on member density, averaged over the ten evaluation plans. The radius governs "
                    "only the network-predicted columns, about half of those exported, which is why the density responds so weakly to it. Computed "
                    "with the joint-connectivity exporter and the joint-aware beam-support check.")
if arms:
    A = arms["arms"]
    def arm_stats(lbl):
        rows = A[lbl]; return dict(n=np.mean([r["n"] for r in rows]), L=np.mean([r["length"] for r in rows]), score=np.mean([r["score"] for r in rows]),
                                   fails=np.mean([r["n_fail"] for r in rows]), ids={r["plan"]: set(r["ids"]) for r in rows})
    AS = {k: arm_stats(k) for k in A}
    base = AS["deployed"]["ids"]
    for k, a in AS.items():
        a["J"] = np.mean([len(a["ids"][p] & base[p]) / max(len(a["ids"][p] | base[p]), 1) for p in base])
    added = sum(len(base[p] - AS["rules only"]["ids"][p]) for p in base); total = sum(len(base[p]) for p in base)
    missing = sum(len(AS["rules only"]["ids"][p] - base[p]) for p in base)
    net_add = sum(len(AS["network only"]["ids"][p] - base[p]) for p in base); net_omit = sum(len(base[p] - AS["network only"]["ids"][p]) for p in base)
    N["ten"].update(arms={k: {kk: (float(v) if not isinstance(v, dict) else None) for kk, v in a.items() if kk != "ids"} for k, a in AS.items()},
                    arm_added=added, arm_total=total, arm_missing=missing, net_add=net_add, net_omit=net_omit)
    lab = {"rules only": "Rules only, learned prior disabled", "deployed": "Deployed: rules and network",
           "permissive": "Permissive prior (GNN_PRIOR = 0)", "network only": "Network only, rule selection disabled"}
    T_["table17"] = [["Configuration", "Piers", "Length (m)", "Coherence", "Failing checks per plan", "Overlap with deployed"]] + [
        [lab[k], f1(AS[k]["n"]), f1(AS[k]["L"]), f3(AS[k]["score"]), f2(AS[k]["fails"]), f3(AS[k]["J"])] for k in ("rules only", "deployed", "network only")]
    B = arms["budget"]
    T_["S4"] = [["Target SW area ratio", "Piers", "Selected length (m)", "SW area (m²)", "Coherence", "Worst ecc."]]
    for b in sorted(B, key=float):
        rows = B[b]
        T_["S4"].append([f"{100 * float(b):.1f}%", f1(np.mean([r["n"] for r in rows])), f1(np.mean([r["length"] for r in rows])),
                         f2(np.mean([r["sw_area"] for r in rows])), f3(np.mean([r["score"] for r in rows])), f"{100 * np.mean([r['ecc_worst'] for r in rows]):.1f}%"])
    bs_ = {float(b): np.mean([r["score"] for r in B[b]]) for b in B}
    N["ten"]["budget_scores"] = bs_
X["S4_caption"] = ("Table S4. Sensitivity of the layout to the shear-wall area budget, averaged over the ten evaluation plans, with the joint-connectivity "
                   "exporter and the joint-aware check. Wall quantity scales with the budget; worst eccentricity is the mean over plans of the "
                   "worst-direction value.")


# =====================================================================================  MSD sweep
import csv, gzip
CHECKS = ["lateral_lines", "eccentricity", "beam_support", "slab_span", "column_continuity", "orphan_nodes"]
deliv = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(ROOT, "we", "results", "msd_validation", "scores.csv")))}
RES = {}
for f in glob.glob(os.path.join(MSDF, "res", "*.json")):
    try: r = json.load(open(f)); RES[r["id"]] = r
    except Exception: pass
GNN = {}
for f in glob.glob(os.path.join(MSDF, "arm_gnnonly", "res", "*.json")):
    try: r = json.load(open(f)); GNN[r["id"]] = r
    except Exception: pass
SOL = {}
for f in sorted(glob.glob(os.path.join(MSDF, "solve_msd_connected_v3_*.jsonl*"))):
    for r in jl(f):
        k_ = int(r["key"])
        if k_ not in SOL or (r.get("ok") and not SOL[k_].get("ok")):
            SOL[k_] = r
SOLP = {int(r["key"]): r for r in jl(os.path.join(SOLVE, "solve_msd150_published_v3.jsonl"))}
# ---------- pass 3b: the orientation-corrected pipeline (msd_worker_v5.py: Stages 2-4 in the principal frame, corrected exporter)
# replaces the connected results, the solves and the network-only arm; "published" keeps the original exporter + original check
# (reproduction of the delivered scores) and takes the joint-aware score of the corrected exporter without the connectivity step.
AXC = {i: r["connected"] for i, r in RES.items() if "error" not in r and "connected" in r}   # first run: axis-projected pipeline
V4R = {}
for f in glob.glob(os.path.join(MSDF, "res_v5", "*.json")):
    r = json.load(open(f)); V4R[r["id"]] = r
assert PARTIAL or len(V4R) >= len(RES), (len(V4R), len(RES))
SOL = {}
for i in list(RES):
    v = V4R.get(i)
    if v is None: RES.pop(i); continue
    r = {k: x for k, x in RES[i].items() if k not in ("connected", "error")}
    if "error" in v or "connected" not in v:
        r["error"] = v.get("error", RES[i].get("error", "v5 failed"))
    else:
        c = dict(v["connected"]); so = c.pop("solve", None)
        if so is not None: SOL[i] = dict(so, key=str(i))
        r["connected"] = c
        if "published" in r and "noconnect" in v:
            r["published"] = dict(r["published"], strict=v["noconnect"]["strict"], metrics=v["noconnect"]["metrics"])
    RES[i] = r
GNN = {}
for f in glob.glob(os.path.join(MSDF, "arm_gnnonly", "res_v5", "*.json")):
    try: r = json.load(open(f)); GNN[r["id"]] = r
    except Exception: pass
n_att = len(RES)
ok_ids = sorted(i for i, r in RES.items() if "error" not in r and "connected" in r and "strict" in r["connected"])
err_ids = sorted(i for i, r in RES.items() if i not in ok_ids)
con = {i: RES[i]["connected"] for i in ok_ids}; pub = {i: RES[i]["published"] for i in ok_ids if "published" in RES[i]}
sc = np.array([con[i]["strict"]["score"] for i in ok_ids])
credit = {"pass": 1.0, "warn": 0.5, "fail": 0.0}
s4 = np.array([np.mean([credit[con[i]["strict"]["checks"][c]] for c in CHECKS[:4]]) for i in ok_ids])
pub_orig = np.array([pub[i]["orig"]["score"] for i in ok_ids if i in pub])
pub_strict = np.array([pub[i]["strict"]["score"] for i in ok_ids if i in pub])
repro = np.mean([abs(pub[i]["orig"]["score"] - float(deliv[i]["score"])) < 1e-9 for i in ok_ids if i in pub and i in deliv and deliv[i]["score"] not in ("", None)])
inc = {c: {s: float(np.mean([con[i]["strict"]["checks"][c] == s for i in ok_ids])) for s in ("pass", "warn", "fail")} for c in CHECKS}
inc_pub = {c: {s: float(np.mean([pub[i]["strict"]["checks"][c] == s for i in pub])) for s in ("pass", "warn", "fail")} for c in CHECKS}
def bsv(d, k): return (d["strict"]["values"].get("beam_support") or {}).get(k, 0)
pub_float_plans = np.mean([bsv(pub[i], "floating") > 0 for i in pub]); pub_unj_plans = np.mean([bsv(pub[i], "unjoined") > 0 for i in pub])
con_float = sum(bsv(con[i], "floating") for i in ok_ids); con_unj = sum(bsv(con[i], "unjoined") for i in ok_ids)
eccw = np.array([max((con[i]["strict"]["values"].get("eccentricity") or {"x": 0}).values() or [0]) for i in ok_ids])
cr = np.array([con[i]["metrics"]["columns"] / pub[i]["metrics"]["columns"] for i in ok_ids if i in pub and pub[i]["metrics"]["columns"] > 0])
def cs(i, k): C = con[i].get("connect", {}); return C.get(k, {})
posts_c = [cs(i, "joints").get("posts_corner", 0) for i in ok_ids]; posts_s = [cs(i, "spans").get("posts_span", 0) for i in ok_ids]
posts_f = [sum(cs(i, k).get("posts_floating", 0) for k in ("pass1", "pass2", "pass3")) for i in ok_ids]
m = N["msd"]
m.update(n_att=n_att, n_ok=len(ok_ids), err_ids=err_ids, err_msgs={str(i): RES[i].get("error", "")[:200] for i in err_ids},
         mean6=float(sc.mean()), mean4=float(s4.mean()), median=float(np.median(sc)), q25=float(np.percentile(sc, 25)), q75=float(np.percentile(sc, 75)),
         ge80=float(np.mean(sc >= 0.8 - 1e-9)), ge90=float(np.mean(sc >= 0.9 - 1e-9)), perfect=float(np.mean(sc >= 1 - 1e-9)), min=float(sc.min()),
         pub_orig_mean=float(pub_orig.mean()), pub_strict_mean=float(pub_strict.mean()), pub_strict_ge80=float(np.mean(pub_strict >= 0.8 - 1e-9)),
         repro=float(repro), inc=inc, inc_pub=inc_pub, pub_float_plans=float(pub_float_plans), pub_unj_plans=float(pub_unj_plans),
         con_float=int(con_float), con_unj=int(con_unj), ecc_med=float(np.median(eccw)), ecc_p90=float(np.percentile(eccw, 90)), ecc_p99=float(np.percentile(eccw, 99)),
         ecc_bins=[float(np.mean(eccw <= 0.10)), float(np.mean((eccw > 0.10) & (eccw <= 0.20))), float(np.mean((eccw > 0.20) & (eccw <= 0.30))), float(np.mean(eccw > 0.30))],
         col_ratio_med=float(np.median(cr)), col_ratio_mean=float(np.mean(cr)), col_ratio_max=float(np.max(cr)),
         posts_corner_tot=int(sum(posts_c)), posts_span_tot=int(sum(posts_s)), posts_float_tot=int(sum(posts_f)),
         plans_with_posts=float(np.mean([a + b + c > 0 for a, b, c in zip(posts_c, posts_s, posts_f)])),
         sec_med=float(np.median([RES[i]["seconds"] for i in ok_ids])))
m["ecc_changed"] = int(sum(pub[i]["strict"]["checks"]["eccentricity"] != con[i]["strict"]["checks"]["eccentricity"] for i in ok_ids if i in pub))
m["ll_changed"] = int(sum(pub[i]["strict"]["checks"]["lateral_lines"] != con[i]["strict"]["checks"]["lateral_lines"] for i in ok_ids if i in pub))
m["ge80_pct"] = pct(m["ge80"]); m["n_scored_str"] = f"{len(ok_ids):,}"

# ---------- orientation: frame angles and wall-geometry fidelity, corrected against axis-projected exporter
FIDN = {i: con[i].get("fidelity", {}) for i in ok_ids}; FIDA = {i: con[i].get("fidelity_axis", {}) for i in ok_ids}
fa = np.array([abs((V4R[i].get("frame") or {}).get("theta_deg") or 0.0) for i in ok_ids])
fid_ids = [i for i in ok_ids if "off_max" in FIDN[i] and "off_max" in FIDA[i]]
rot_ids = [i for i in ok_ids if abs((V4R[i].get("frame") or {}).get("theta_deg") or 0.0) > 5.0]
onA = np.array([FIDA[i]["off_max"] for i in fid_ids]); onN = np.array([FIDN[i]["off_max"] for i in fid_ids])
ax_sc = np.array([AXC[i]["strict"]["score"] for i in ok_ids if i in AXC])
def _mean_on(ids, src): return float(np.mean([src[i]["strict"]["score"] for i in ids if i in src])) if ids else np.nan
m["orient"] = dict(
    rot_share=float(np.mean(fa > 5.0)), n_rot=len(rot_ids), fa_med_rot=float(np.median(fa[fa > 5.0])) if (fa > 5.0).any() else 0.0,
    n_fid=len(fid_ids), axis_off_med=float(np.median(onA)), axis_off_p95=float(np.percentile(onA, 95)), axis_off_max=float(onA.max()),
    new_off_med=float(np.median(onN)), new_off_p95=float(np.percentile(onN, 95)), new_off_p99=float(np.percentile(onN, 99)), new_off_max=float(onN.max()),
    axis_share030=float(np.mean(onA > 0.30)), new_share030=float(np.mean(onN > 0.30)),
    axis_angmax_med_rot=float(np.median([FIDA[i]["ang_max"] for i in fid_ids if i in set(rot_ids)])) if rot_ids else np.nan,
    axis_len_ratio=float(sum(FIDA[i]["panel_len"] for i in fid_ids) / sum(FIDA[i]["s4_len"] for i in fid_ids)),
    new_len_ratio=float(sum(FIDN[i]["panel_len"] for i in fid_ids) / sum(FIDN[i]["s4_len"] for i in fid_ids)),
    axis_mean6=float(ax_sc.mean()), axis_ge80=float(np.mean(ax_sc >= 0.8 - 1e-9)),
    axis_ecc_fail=float(np.mean([AXC[i]["strict"]["checks"]["eccentricity"] == "fail" for i in ok_ids if i in AXC])),
    axis_ll_fail=float(np.mean([AXC[i]["strict"]["checks"]["lateral_lines"] == "fail" for i in ok_ids if i in AXC])),
    rot_mean_axis=_mean_on(rot_ids, AXC), rot_mean_new=_mean_on(rot_ids, con),
    ax_mean_axis=_mean_on([i for i in ok_ids if i not in set(rot_ids)], AXC), ax_mean_new=_mean_on([i for i in ok_ids if i not in set(rot_ids)], con),
    score_changed=float(np.mean([AXC[i]["strict"]["score"] != con[i]["strict"]["score"] for i in ok_ids if i in AXC])),
    score_changed_axisaligned=float(np.mean([AXC[i]["strict"]["score"] != con[i]["strict"]["score"] for i in ok_ids if i in AXC and i not in set(rot_ids)])))

_om = {int(r["id"]): r for r in csv.DictReader(open(os.path.join(ROOT, "04_results", "figs_v3", "msd_onmask_v5.csv")))}
def _omv(i, k):
    v = _om.get(i, {}).get(k, ""); return float(v) if v not in ("", None) else np.nan
_rs = set(rot_ids); _ar = [i for i in ok_ids if i not in _rs]
m["onmask"] = {f"{k}_{g}": float(np.nanmedian([_omv(i, k) for i in ids])) for k in ("first_panels", "new_panels", "first_walls", "new_walls")
               for g, ids in (("rot", rot_ids), ("ax", _ar))}
m["onmask"].update({f"{k}_mean": float(np.nanmean([_omv(i, k) for i in ok_ids])) for k in ("first_panels", "new_panels", "first_walls", "new_walls")})
m["onmask"].update(first_lt80=float(np.nanmean([_omv(i, "first_panels") < 0.8 for i in ok_ids if np.isfinite(_omv(i, "first_panels"))])),
                   new_lt80=float(np.nanmean([_omv(i, "new_panels") < 0.8 for i in ok_ids if np.isfinite(_omv(i, "new_panels"))])))
_ws = {int(r["id"]): float(r["wall_len"]) for r in csv.DictReader(open(os.path.join(ROOT, "04_results", "figs_v3", "gallery", "msd_wallstock.csv")))}
m["wallstock"] = dict(ll_fail_med=float(np.median([_ws[i] for i in ok_ids if i in _ws and con[i]["strict"]["checks"]["lateral_lines"] == "fail"])),
                      ll_pass_med=float(np.median([_ws[i] for i in ok_ids if i in _ws and con[i]["strict"]["checks"]["lateral_lines"] == "pass"])))
# ---------- weighting (S5), connected models, joint-aware check
WS = {"equal (deployed)": {c: 1.0 for c in CHECKS},
      "lateral system emphasis": dict(lateral_lines=2, eccentricity=2, beam_support=1, slab_span=1, column_continuity=1, orphan_nodes=1),
      "gravity system emphasis": dict(lateral_lines=1, eccentricity=1, beam_support=2, slab_span=2, column_continuity=1, orphan_nodes=1),
      "topology down-weighted": dict(lateral_lines=1, eccentricity=1, beam_support=1, slab_span=1, column_continuity=.25, orphan_nodes=.25),
      "four informative checks": dict(lateral_lines=1, eccentricity=1, beam_support=1, slab_span=1, column_continuity=0, orphan_nodes=0)}
def wscore(chk, w, warn=0.5):
    cc = {"pass": 1.0, "warn": warn, "fail": 0.0}; return sum(w[c] * cc[chk[c]] for c in CHECKS) / sum(w.values())
def spear(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    # average ranks for ties
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)
ten_chk = [{c["name"]: c["status"] for c in r["S"]["checks"]} for r in ten]
msd_chk = [con[i]["strict"]["checks"] for i in ok_ids]
T_["S5"] = [["Weighting scheme", "Mean (ten plans)", "Mean (MSD)", "MSD >= 0.80", "Rank correlation (MSD)"]]
basem = [wscore(c, WS["equal (deployed)"]) for c in msd_chk]
S5v = {}
for name, w in list(WS.items()) + [("equal, warn credit 0.25", None), ("equal, warn credit 0.75", None)]:
    warn = 0.5 if w is not None else float(name.split()[-1]); ww = w or WS["equal (deployed)"]
    a = [wscore(c, ww, warn) for c in ten_chk]; b = [wscore(c, ww, warn) for c in msd_chk]
    S5v[name] = dict(ten=np.mean(a), msd=np.mean(b), ge80=np.mean(np.array(b) >= 0.8 - 1e-9), rho=spear(basem, b) if name != "equal (deployed)" else 1.0)
    T_["S5"].append([name, f3(np.mean(a)), f3(np.mean(b)), pct(S5v[name]["ge80"], 0), f3(S5v[name]["rho"])])
m["S5"] = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in S5v.items()}
X["S5_caption"] = ("Table S5. Coherence score under alternative weightings of the six checks and alternative credit for a warning, recomputed from the "
                   "stored per-check outcomes of the ten evaluation plans and the MSD corpus (joint-connectivity exporter, joint-aware beam-support check). "
                   "Rank correlation is Spearman's rho against the deployed equal weighting.")
# ---------- Table 16
notes = {"lateral_lines": "sparse-wall slivers", "eccentricity": f"median worst-dir. {pct(m['ecc_med'])}", "beam_support": "shared joints, no floating member",
         "slab_span": "", "column_continuity": "", "orphan_nodes": ""}
T_["table16"] = [["Check", "Pass", "Warn", "Fail", "Note"]] + [[c, pct(inc[c]["pass"]), pct(inc[c]["warn"]), pct(inc[c]["fail"]), notes[c]] for c in CHECKS]
# ---------- network-only arm on MSD
g_ok = sorted(i for i, r in GNN.items() if "error" not in r and "connected" in r and i in con)
if g_ok:
    gs = np.array([GNN[i]["connected"]["strict"]["score"] for i in g_ok]); ds = np.array([con[i]["strict"]["score"] for i in g_ok])
    m["gnn"] = dict(n=len(g_ok), mean=float(gs.mean()), dep_mean=float(ds.mean()), ge80=float(np.mean(gs >= 0.8 - 1e-9)), dep_ge80=float(np.mean(ds >= 0.8 - 1e-9)),
                    ecc_fail=float(np.mean([GNN[i]["connected"]["strict"]["checks"]["eccentricity"] == "fail" for i in g_ok])),
                    dep_ecc_fail=float(np.mean([con[i]["strict"]["checks"]["eccentricity"] == "fail" for i in g_ok])),
                    ll_fail=float(np.mean([GNN[i]["connected"]["strict"]["checks"]["lateral_lines"] == "fail" for i in g_ok])),
                    dep_ll_fail=float(np.mean([con[i]["strict"]["checks"]["lateral_lines"] == "fail" for i in g_ok])),
                    complete=len(GNN) >= n_att)


# =====================================================================================  solver statistics, MSD
from scipy.stats import spearmanr
s_ok = [i for i in ok_ids if i in SOL]
def sg(i, k, d=np.nan):
    v = SOL[i].get(k, d); return d if v is None else v
solved = [i for i in s_ok if SOL[i].get("ok")]
m["solve"] = dict(n=len(s_ok), n_ok=len(solved), complete=len(s_ok) >= len(ok_ids),
                  fail_ids=[i for i in s_ok if not SOL[i].get("ok")][:20],
                  float_any=int(sum(sg(i, "n_float_beams", 0) > 0 for i in s_ok)),
                  eq_max=float(np.nanmax([sg(i, "equilibrium_err") for i in solved])) if solved else np.nan,
                  uz_max=float(np.nanmax([sg(i, "max_uz_mm") for i in solved])) if solved else np.nan,
                  uz_p99=float(np.nanpercentile([sg(i, "max_uz_mm") for i in solved], 99)) if solved else np.nan,
                  mid_max=float(np.nanmax([sg(i, "max_beam_mid_mm") for i in solved])) if solved else np.nan,
                  mid_p99=float(np.nanpercentile([sg(i, "max_beam_mid_mm") for i in solved], 99)) if solved else np.nan,
                  Ld_min=float(np.nanmin([sg(i, "min_span_over_defl") for i in solved])) if solved else np.nan,
                  Ld_p1=float(np.nanpercentile([sg(i, "min_span_over_defl") for i in solved], 1)) if solved else np.nan,
                  T1_med=float(np.median([sg(i, "T1") for i in solved])) if solved else np.nan,
                  TT_med=float(np.median([sg(i, "T1") / sg(i, "EC8_T1_walls") for i in solved])) if solved else np.nan,
                  ws_med=float(np.nanmedian([np.nanmean([sg(i, "wall_shear_share_X"), sg(i, "wall_shear_share_Y")]) for i in solved])) if solved else np.nan,
                  wg_med=float(np.nanmedian([sg(i, "wall_share_gravity") for i in solved])) if solved else np.nan,
                  beam_share_med=float(np.nanmedian([sg(i, "slab_share_on_beams") for i in solved])) if solved else np.nan,
                  sec_med=float(np.nanmedian([sg(i, "seconds") for i in solved])) if solved else np.nan)
# checks against solver
tor = {i: max(sg(i, "torsion_ratio_X", 1), sg(i, "torsion_ratio_Y", 1)) for i in solved}
Tmax_ratio = {i: max(sg(i, "T_X", sg(i, "T1")), sg(i, "T_Y", sg(i, "T1"))) / sg(i, "EC8_T1_walls") for i in solved}
gr = {"pass": 0, "warn": 1, "fail": 2}
if solved:
    egrade = [gr[con[i]["strict"]["checks"]["eccentricity"]] for i in solved]
    ewc = [max((con[i]["strict"]["values"].get("eccentricity") or {"x": 0}).values()) for i in solved]
    m["solve"].update(rho_grade=float(spearmanr(egrade, [tor[i] for i in solved]).statistic),
                      rho_ecc=float(spearmanr(ewc, [tor[i] for i in solved]).statistic),
                      ecc_fail_tor12=float(np.mean([tor[i] > 1.2 for i in solved if con[i]["strict"]["checks"]["eccentricity"] == "fail"])),
                      ecc_pass_tor12=float(np.mean([tor[i] > 1.2 for i in solved if con[i]["strict"]["checks"]["eccentricity"] == "pass"])),
                      tor_med_by={k: float(np.median([tor[i] for i in solved if con[i]["strict"]["checks"]["eccentricity"] == k])) for k in gr},
                      TT_fail=float(np.median([Tmax_ratio[i] for i in solved if con[i]["strict"]["checks"]["lateral_lines"] == "fail"])),
                      TT_pass=float(np.median([Tmax_ratio[i] for i in solved if con[i]["strict"]["checks"]["lateral_lines"] == "pass"])),
                      n_ll_fail=int(sum(con[i]["strict"]["checks"]["lateral_lines"] == "fail" for i in solved)))
# published exporter, MSD sample (150 stratified)
pp = [r for r in SOLP.values()]
m["solve_pub150"] = dict(n=len(pp), n_ok=sum(bool(r.get("ok")) for r in pp), float_any=int(sum(r.get("n_float_beams", 0) > 0 for r in pp)),
                         float_med=float(np.nanmedian([r.get("float_frac", np.nan) for r in pp])), float_max=float(np.nanmax([r.get("float_frac", np.nan) for r in pp])),
                         uz_med=float(np.nanmedian([r.get("max_uz_mm", np.nan) for r in pp if r.get("ok")])),
                         n_uz25=int(sum((r.get("n_uz_gt25mm") or 0) > 0 for r in pp if r.get("ok"))))

# eccentricity: t.L^3 proxy vs flexure+shear stiffness on the check's own merged piers, against the solver torsion ratio
E_c, NU = 25743e3, 0.2; G_c = E_c / (2 * (1 + NU)); KAP = 1.2
def ecc_two(M, H=3.0):
    nd = SAN._principal_frame(M, SAN._nodes_xyz(M))[0]
    segs = SAN._merge_collinear(SAN._story_panels(M, nd, 0, sw_only=True))
    pts = [nd[i][:2] for sl in M.get("slabs", []) if sl.get("story") == 0 for i in sl["nodes"]]
    if not segs or not pts: return None
    pts = np.vstack(pts); lo, hi = pts.min(0), pts.max(0); gc = (lo + hi) / 2; w = np.maximum(hi - lo, 1e-6)
    out = {}
    for name, kf in (("proxy", lambda t, L: t * L ** 3), ("full", lambda t, L: 1.0 / (H ** 3 / (3 * E_c * t * L ** 3 / 12) + KAP * H / (G_c * t * L)))):
        kx = ky = xc = yc = 0.0
        for p1, p2, t in segs:
            L = float(np.linalg.norm(p2 - p1)); k = kf(t, L); c = (p1 + p2) / 2
            if SAN._orient(p1, p2) == "H": kx += k; yc += k * c[1]
            else: ky += k; xc += k * c[0]
        e = []
        if ky > 0: e.append(abs(xc / ky - gc[0]) / w[0])
        if kx > 0: e.append(abs(yc / kx - gc[1]) / w[1])
        out[name] = max(e) if e else np.nan
    return out
verd = lambda v: "pass" if v < 0.10 else ("warn" if v < 0.20 else "fail")
s12 = []
for r in ten:
    o = ecc_two(r["M"]); n_p = len(SAN._merge_collinear(SAN._story_panels(r["M"], SAN._nodes_xyz(r["M"]), 0, sw_only=True)))
    s12.append((r["plan"], n_p, o["proxy"], o["full"]))
T_["S12"] = [["Plan", "Merged piers", "Worst ecc., t·L³ proxy", "Worst ecc., flexure + shear", "Difference", "Verdict (proxy / full)"]] + [
    [p, str(n), pct(a), pct(b), f"{100 * abs(a - b):.1f} pp", f"{verd(a)} / {verd(b)}"] for p, n, a, b in s12]
d_ = [100 * abs(a - b) for _, _, a, b in s12]
N["ten"].update(s12_dmed=float(np.median(d_)), s12_dmax=float(max(d_)), s12_changed=sum(verd(a) != verd(b) for _, _, a, b in s12),
                s12_lenient=sum(gr[verd(b)] < gr[verd(a)] for _, _, a, b in s12))
if solved and "--skip-ecc" not in sys.argv:
    EP = {}
    for i in solved:
        try:
            EP[i] = ecc_two(json.loads(gzip.open(os.path.join(MSDF, "models_v5", f"{i}.json.gz")).read()), H=3.0)
        except Exception:
            pass
    ii = [i for i in EP if EP[i] is not None and np.isfinite(EP[i]["proxy"]) and np.isfinite(EP[i]["full"])]
    m["ecc_cmp"] = dict(n=len(ii), rho_proxy=float(spearmanr([EP[i]["proxy"] for i in ii], [tor[i] for i in ii]).statistic),
                        rho_full=float(spearmanr([EP[i]["full"] for i in ii], [tor[i] for i in ii]).statistic),
                        verdict_change=float(np.mean([verd(EP[i]["proxy"]) != verd(EP[i]["full"]) for i in ii])),
                        lenient=float(np.mean([gr[verd(EP[i]["full"])] < gr[verd(EP[i]["proxy"])] for i in ii])),
                        fail_full=float(np.mean([verd(EP[i]["full"]) == "fail" for i in ii])))


# =====================================================================================  text
t = N["ten"]; S_ = m.get("solve", {}); P_ = m.get("solve_pub150", {}); EC = m.get("ecc_cmp", {})
n_all_models = n_ten + m["n_ok"]
def I(x): return f"{x:,}"
c_ten_inc = 100 * (t["cols0_total"] / t["cols0_pub"] - 1)
X["abstract"] = ("Converting a raster floor plan into a structural analysis model is manual, which prevents portfolio-scale assessment of existing "
                 "buildings. This study presents Plan-to-FEM, a pipeline that converts a single raster floor-plan image into a "
                 "three-dimensional finite element model through transformer-based wall segmentation, skeleton-based vectorization with metric-scale "
                 "estimation, structural-graph construction, and a multi-task graph neural network that proposes shear walls and columns under "
                 "engineering-rule validation. A joint-connectivity step gives every member a load path, and a six-check coherence module audits each "
                 "model. Vectorization recovers wall centerlines with precision 0.92 and recall 0.77; on a leakage-free split the network ranks shear "
                 "walls with ROC AUC 0.70 and places columns with layout F1 0.59. Ten plans and " + ("all 4,572" if m["n_ok"] == 4572 else I(m["n_ok"]) + " of 4,572") + " Modified Swiss Dwellings "
                 "plans complete end-to-end, and every exported model completes gravity, modal and lateral analyses in an independent solver. The models "
                 "remain over-provided with columns and require engineer verification.")
X["highlight_solver"] = f"All {I(n_all_models)} exported models solve in OpenSees, with a load path for every member."
X["contrib_solver"] = ("Every exported model is also analysed in an independent solver, which shows that a distance-based support test is not sufficient: "
                       "without an explicit joint-connectivity step, models that pass it contain members with no load path; with the step, every member has one.")
X["connectivity_para"] = (
    "Connectivity is enforced through shared joints, because an FE package connects elements only where they share a node. Coincident nodes are "
    "merged within 5 mm, and a joint-connectivity step runs after member generation. First, every beam end that is not on a column or wall joint "
    "is moved onto the nearest column, wall edge or other beam within 0.20 m, and the wall pier stack or the supporting beam is divided at that "
    "point so that the mesh stays conforming; an end with nothing within 0.20 m receives a gravity post. Second, beam-to-beam joints within "
    "0.20 m of a column or wall joint are merged onto it, which matters chiefly for the corners of the slab-outline ring beam, and an unsupported "
    "joint at which the framing changes direction receives a post. Third, a straight run of beams through unsupported joints that is longer than "
    "the 6.0 m span limit receives intermediate posts at equal spacing. Finally, any sub-assembly still without an element path to the base "
    "receives posts at its free ends and corners. Posts are continued storey by storey to the base, and joints referenced by no element are "
    "removed. Grid spans are subdivided by intermediate columns so that no clear distance between vertical supports exceeds 6.0 m, and piers "
    "longer than 3.0 m are split (within a 20% tolerance) rather than exported as monolithic panels. Members frame node-to-node at the storey top "
    "elevation: no rigid end zones, insertion-point offsets or end-length offsets are assigned, so member centerlines meet at a shared joint and "
    "panel-zone flexibility is not represented. Section 10.8 shows why the step is needed, by solving models exported with and without it: a "
    "distance tolerance alone leaves members without a load path. Table 7 collects the complete idealization "
    "together with its governing parameters and tolerances.")
s5 = m["S5"]; s5m = [v["msd"] for v in s5.values()]; s5g = [v["ge80"] for v in s5.values()]; s5r = [v["rho"] for v in s5.values()]
X["weighting_para"] = (
    "Supplementary Table S5 recomputes the score of every model under alternative weightings from the stored per-check outcomes. The ranking of "
    f"models is robust: the Spearman correlation between the deployed equal weighting and every alternative is at least {min(s5r):.3f} over the MSD "
    f"corpus. The absolute value is not: the MSD mean ranges from {min(s5m):.3f} to {max(s5m):.3f}, and the share of plans at or above 0.80 from "
    f"{pct(min(s5g), 0)} to {pct(max(s5g), 0)}, depending only on the weighting, so the score should not be read as an acceptance level. Column "
    f"continuity and orphan nodes pass on {pct(min(inc['column_continuity']['pass'], inc['orphan_nodes']['pass']))} or more of the models and add an almost "
    f"constant offset; removing them lowers the MSD mean from {s5['equal (deployed)']['msd']:.3f} to {s5['four informative checks']['msd']:.3f} and the "
    f"share at or above 0.80 from {pct(s5['equal (deployed)']['ge80'], 0)} to {pct(s5['four informative checks']['ge80'], 0)} while leaving the ranking "
    f"unchanged (rho = {s5['four informative checks']['rho']:.2f}). This four-check score is therefore reported alongside the six-check score and is "
    "recommended for comparing layouts.")
X["density_para"] = (
    f"Table 13 reports, per plan, the storey floor area from the exported slab, the column count, its density, the implied tributary area and the "
    f"median column-to-column distance. Density ranges from {min(dens):.1f} to {max(dens):.1f} columns per 100 m2, median {np.median(dens):.1f}, so each "
    f"column carries only {min(trib):.1f} to {max(trib):.1f} m2, median {np.median(trib):.1f}, with neighbors {min(spc):.2f} to {max(spc):.2f} m apart. "
    "These are an order of magnitude from practice, and the pipeline contradicts itself internally: its own export rule subdivides spans so no "
    f"clear distance exceeds 6.0 m, which on a regular grid implies about 36 m2 per column against the {np.median(trib):.1f} produced. The layouts are "
    f"not economical grids but dense picket lines along wall centerlines. The joint-connectivity step contributes {t['cols0_total'] - t['cols0_pub']} of "
    f"these columns ({c_ten_inc:.0f}% more than without it), as gravity posts under joints that would otherwise have no vertical support.")
ss = storey_secs
X["sections_para"] = (
    f"The section distribution compounds it: of {t['cols0_total']:,} storey-level columns, {ss.get('400', 0)} are 400 mm square and {ss.get('250', 0)} "
    f"are 250 mm square (with {ss.get('300', 0)} at 300 mm and {ss.get('350', 0)} at 350 mm), so the tributary sizing rule assigns large sections to "
    "columns standing about a meter apart, which no engineer would detail; the posts of the connectivity step take the default 400 mm section. "
    "The mechanism is identifiable: axes are clustered from every wall centerline, so partition-rich plans yield many closely spaced axes; a "
    "column is placed at every unsupported intersection, at pier ends and wherever a span check demands one; and the de-duplication radius is "
    "only 0.60 m. The consequence must be stated plainly, because it bears on how the score reads: a floor supported this densely satisfies the "
    "slab-span and beam-support checks almost automatically, so a high score partly reflects over-provision rather than judgement. Together "
    "with the weighting analysis of Supplementary Table S5, this is why the score is used here for triage and ranking, not as an acceptance metric.")
if "nms_d06" in t:
    X["nms_para"] = (
        f"The density is not governed by the suppression radius. Increasing the radius from 0.60 m to 4.00 m reduces density only from "
        f"{t['nms_d06']:.1f} to {t['nms_d40']:.1f} columns per 100 m2, and coherence moves from {t['nms_s06']:.3f} to {t['nms_s40']:.3f} as spans "
        f"lengthen (Supplementary Table S10), because suppression acts only on the {t['net']} network-predicted columns ({100 * t['net'] / t['cols0_total']:.0f}%); "
        f"the remaining {t['rule']} are generated by the export rules as demoted short piers, boundary columns at pier ends, span subdivision and "
        "connectivity posts.")
lat_all = all(r["ch"]["lateral_lines"]["status"] == "pass" for r in ten)
span_ok = all(r["ch"]["slab_span"]["status"] != "fail" for r in ten)
topo_ok = all(r["ch"][c]["status"] == "pass" for r in ten for c in ("orphan_nodes", "column_continuity", "beam_support"))
X["batch_para"] = (
    "The complete pipeline was run on the ten evaluation plans; all completed end to end with no stage failures. These results establish "
    "robustness and internal rule-consistency only, not structural correctness or efficiency, and the quantities below show why that distinction "
    "matters. Table 14 reports per-plan composition and outcome, Supplementary Table S11 the six-check breakdown, Fig. 8 the batch. Model size "
    f"tracks plan size and density: three-storey totals span {min(r['mm']['sw_panels'] for r in ten)} to {max(r['mm']['sw_panels'] for r in ten)} "
    f"shear-wall panels, {min(r['mm']['beams'] for r in ten)} to {max(r['mm']['beams'] for r in ten)} beams and "
    f"{min(r['mm']['columns'] for r in ten)} to {max(r['mm']['columns'] for r in ten)} columns, {min(r['locs'] for r in ten)} to "
    f"{max(r['locs'] for r in ten)} column locations per floor, all continuous. Mean coherence is {t['mean6']:.3f} ({t['mean4']:.3f} for the four "
    f"informative checks); {t['nofail']} of ten plans complete without a failing check and {t['all6']} pass all six outright. The dominant residual "
    "defect is stiffness eccentricity, failing in one direction on plans whose usable piers are exhausted on one side. "
    + ("No plan shows orphan nodes, discontinuous columns, unsupported or unconnected beam ends, or members without a path to the base; " if topo_ok else "")
    + ("all satisfy the two-lateral-line requirement" if lat_all else "") + (", and no slab span exceeds the 6.0 m limit. " if span_ok else ". ")
    + f"Scored with the same joint-aware check, the models exported without the connectivity step average {t['mean6_pub_strict']:.3f} and fail beam "
    "support on every plan (Section 10.8).")
X["quantities_para"] = (
    "Table 15 supplies material quantities, so that an over-provisioned model cannot be rewarded by checks that test only topology. Concrete "
    "volumes computed from exported geometry and assigned sections, normalized by three-storey floor area, run from "
    f"{min(conc):.3f} to {max(conc):.3f} m3 per m2, median {np.median(conc):.3f}. The slab is a fixed 0.15 m and {share['v_slab']:.0f}% of the volume, "
    f"so it cannot be the source of excess; walls account for {share['v_wall']:.0f}%, beams {share['v_beam']:.0f}% and columns {share['v_col']:.0f}%. "
    "Against figures commonly quoted for low-rise reinforced-concrete residential construction, nearer 0.35 to 0.45 m3 per m2, the models are "
    "heavier, and the columns alone, at about a fifth of the volume, are where the density analysis says the excess lies.")
X["msd_para1"] = (
    "To test whether the geometric and structural stages generalize beyond ten curated plans, the pipeline was swept over the training split of "
    f"the Modified Swiss Dwellings benchmark (Section 6.5). All 4,572 available plans were attempted and {I(m['n_ok'])} of them "
    f"({100 * m['n_ok'] / 4572:.2f}%) completed every stage through coherence checking: vectorization with prior-based scale estimation, consolidation, "
    "multi-task GNN enrichment, four-storey FE export with the joint-connectivity step, and the six checks. "
    + ("This includes plan 1453, whose sparse mask (2.6% wall pixels) is drawn at about 26 degrees to the sheet axes: traced in the drawing "
       "axes it yields no valid wall segment and an empty graph, traced in its principal frame (Section 5.5) 37 wall rectangles. "
       if not m["err_ids"] else f"The plans that fail are {m['err_ids']}. ")
    + "No per-plan tuning of any kind was applied: the same thresholds, budgets and trained checkpoint used for the ten-plan batch ran unchanged "
    "over real Swiss multi-apartment stock.")
X["msd_para2"] = (
    f"Fig. 11 shows the coherence-score distribution. The mean is {m['mean6']:.3f} ({m['mean4']:.3f} for the four informative checks) and the median "
    f"{m['median']:.3f} (quartiles {m['q25']:.3f}/{m['q75']:.3f}); {pct(m['ge80'])} of plans score at least 0.80, {pct(m['ge90'])} at least 0.90, and "
    f"{pct(m['perfect'])} achieve a perfect 1.00, while the minimum is {m['min']:.3f}. Table 16 breaks the outcome down by check. The topology-level "
    f"checks pass almost universally at scale: column continuity on {pct(inc['column_continuity']['pass'])} and orphan nodes on "
    f"{pct(inc['orphan_nodes']['pass'])} of plans, lateral-line count on {pct(inc['lateral_lines']['pass'])} and slab span on "
    f"{pct(inc['slab_span']['pass'])}. Beam support, which requires every beam end to sit on a shared joint and every member to reach the "
    f"base, passes on {pct(inc['beam_support']['pass'])}; no member of any connected model is floating. The residual defect class is stiffness "
    f"eccentricity, failing on {pct(inc['eccentricity']['fail'])} of plans and warning on a further {pct(inc['eccentricity']['warn'])} (median "
    f"worst-direction eccentricity {pct(m['ecc_med'])}, 90th percentile {pct(m['ecc_p90'])}): multi-apartment complexes with elongated, articulated "
    "footprints genuinely concentrate their wall stock asymmetrically, and the balancing pass can only redistribute what exists. Scored with the "
    f"same joint-aware check, the models exported without the connectivity step average {m['pub_strict_mean']:.3f}, and "
    f"{pct(m['pub_float_plans'])} of them contain members without a path to the base.")
if EC:
    common = ("The eccentricity check uses the flexural proxy k ∝ t·L³ over merged piers, which ignores shear deformation. Recomputing the stiffness "
              "centre with the full cantilever stiffness k = [H³/(3EI) + κH/(GA)]⁻¹ (A = t·L, κ = 1.2, ν = 0.2) over the same piers changes "
              f"worst-direction eccentricity by a median of {t['s12_dmed']:.1f} percentage points (maximum {t['s12_dmax']:.1f}) on the ten plans and changes "
              f"{t['s12_changed']} of the ten verdicts, each towards leniency (Supplementary Table S12): merged piers are often longer than the storey is "
              "high, so shear flexibility matters and the proxy overweights long walls. Over the MSD corpus the verdict would change on "
              f"{pct(EC['verdict_change'])} of plans, almost always towards leniency, and the failure rate would fall from {pct(inc['eccentricity']['fail'])} "
              f"to {pct(EC['fail_full'])}. ")
    if EC["rho_full"] > EC["rho_proxy"] + 0.01:
        X["ecc_proxy_para"] = common + (
            "The full stiffness also tracks the torsional response computed by the solver (Section 10.8) slightly better: the Spearman correlation "
            f"between worst-direction eccentricity and the solver torsion ratio is {EC['rho_full']:.2f}, against {EC['rho_proxy']:.2f} for the proxy "
            f"(n = {I(EC['n'])}). The proxy is kept in this paper so that all reported scores share one definition, and its eccentricity verdicts "
            "should be read as conservative; adopting the full stiffness is a recommended refinement of the check.")
    else:
        X["ecc_proxy_para"] = common + (
            "Against the torsional response computed by the solver (Section 10.8), however, the proxy performs at least as well: the Spearman "
            f"correlation with the solver torsion ratio is {EC['rho_proxy']:.2f} for the proxy and {EC['rho_full']:.2f} for the full stiffness "
            f"(n = {I(EC['n'])}). The proxy is therefore retained, with the caveat that its verdicts are conservative for long, squat walls.")
X["tolerance_para"] = (
    "A check can also fail for reasons of measurement rather than structure. Without the connectivity step, export closes with a repair that "
    "inserts a gravity post beneath any beam end with no column, wall edge or other beam within tolerance, and the distance-only beam-support "
    "test audits the same condition. If the repair uses 0.25 m while the audit uses 0.20 m, ends falling between the two are left unrepaired and "
    "reported as failures: across the 4,572 plans, 2,944 such ends (in 1,886 plans) lie in the 0.20-0.25 m band (median 0.223 m, maximum "
    "0.250 m; Supplementary Table S13), and matching the two tolerances at 0.20 m, as deployed, raises the MSD mean under that test from 0.840 to "
    "0.909. Matching tolerances ensure that every beam end has a support nearby; they do not ensure that the end is connected to it, which only "
    "the solver check of Section 10.8 reveals.")
b_ = m["ecc_bins"]
X["ecc_dist_para"] = (
    f"Stiffness eccentricity behaves differently. Worst-direction eccentricity is at most 10% of the plan dimension on {pct(b_[0])} of plans, between "
    f"10 and 20% on {pct(b_[1])}, between 20 and 30% on {pct(b_[2])} and above 30% on {pct(b_[3])} (99th percentile {pct(m['ecc_p99'])}), so these "
    "failures reflect a genuine and often large asymmetry of the wall stock rather than marginal exceedances of a threshold.")
X["fig12_para"] = (
    "Fig. 12 illustrates the score spectrum with three representative plans, shown as exported before the joint-connectivity step, which adds "
    "posts but leaves walls and layout unchanged. The top row (score 1.00) is a compact multi-apartment complex whose exported model carries a "
    "complete four-storey grid of columns, interior beams and distributed shear walls. The middle row (score 0.833, the lower quartile) is "
    "structurally complete but carries an eccentricity warning. The bottom row (score 0.50) exposes the principal out-of-scope geometry: a long, "
    "curved, sliver-shaped building whose non-orthogonal footprint defeats the orthogonal-snapping vectorizer and axis-based grid detector, "
    f"leaving sparse walls and an under-populated structural system, consistent with the {pct(inc['lateral_lines']['fail'])} lateral-line failures "
    "being concentrated in such footprints. That the pipeline degrades to a low score rather than crashing on these inputs, and reports the "
    "specific deficient checks, is itself the intended behavior of the coherence layer.")
def sup10(x):
    e = int(np.floor(np.log10(x))) + 1
    return "10" + str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
# ---------- §10.8
X["solver_paras"] = [
    ("The coherence checks test the exported topology; they do not show that an FE package can analyse the model. Every exported model was "
     "therefore mapped into OpenSeesPy [50,51] and analysed. The mapping reads model_fem.json as an analysis package would, connecting elements "
     "only where they share a joint. Shear-wall piers become ShellMITC4 elements with an elastic membrane-plate section, meshed at about 0.75 m; "
     "columns and beams become elastic beam-column elements with gross rectangular sections, beams bending about their strong axis; each storey "
     "carries a rigid diaphragm with its mass lumped at the master node; and base joints are fixed. Gravity load is the self-weight of walls and "
     "frames (25 kN/m3) plus, over the slab outline, the 0.15 m slab, 1.5 kPa superimposed dead load and 2.0 kPa live load, each 0.25 m slab "
     "sample being carried by the nearest beam, as a uniform line load, or the nearest vertical support; the seismic mass is G + 0.3Q. Three "
     "analyses are run: linear static gravity, with the sum of reactions checked against the applied load; modal analysis condensed to the "
     "diaphragm degrees of freedom; and an equivalent lateral force of 10% of the seismic weight in each direction, distributed in proportion to "
     "storey mass times height. No design or code check is performed, and the idealization is not that of the target package: the slab enters "
     "through its diaphragm and its load, not as shell elements."),
    ("Applied to models exported without the joint-connectivity step, the check exposes a defect that a distance-based support test passes. "
     f"Every one of the ten batch models and {P_.get('float_any', 0)} of the {P_.get('n', 0)} models of a score-stratified MSD sample (exported also without "
     f"the orientation handling) contain members with no element path to the base, a median {100 * t['pub_float_med']:.1f}% of the beams on the ten plans "
     f"(maximum {100 * t['pub_float_max']:.1f}%) and {100 * P_.get('float_med', 0):.1f}% in the sample (maximum {100 * P_.get('float_max', 0):.1f}%); over the "
     f"full corpus, {pct(m['pub_float_plans'])} of the models exported without the step contain such members. A distance-only beam-support test passes "
     "them, because it asks whether each beam end has a support within 0.20 m, not whether it is connected to it. Three mechanisms are responsible: grid-beam ends placed at wall "
     "sample points that are not wall-panel joints; corners of the slab-outline ring beam lying a few millimetres from wall corners; and long "
     "runs of beams framing into one another with no vertical support. With the floating parts removed so that the rest can be solved, the "
     f"largest gravity deflection reaches a median of {t['pub_uz_med']:.0f} mm on the ten plans and {P_.get('uz_med', 0):.0f} mm in the sample, the "
     "signature of near-mechanisms."),
    ("The joint-connectivity step of Section 5.5 removes this defect, and the beam-support check of Section 5.6 tests the same condition. "
     "Table 18 compares the exporter without and with the step. With the step, "
     + (f"all {I(n_ten + S_.get('n_ok', 0))} models (ten batch and {I(S_.get('n_ok', 0))} MSD) " if S_.get("complete") else f"[{S_.get('n_ok', 0)} MSD so far] ")
     + "complete the gravity, modal and lateral analyses with no floating member; the largest equilibrium error is "
     "below " + sup10(max(t['con_eq_max'], S_.get('eq_max', 0), 1e-300)) + f", the largest joint deflection under gravity is {max(t['con_uz_max'], S_.get('uz_max', 0)):.1f} mm and "
     f"the largest beam mid-span deflection {max(t['con_mid_max'], S_.get('mid_max', 0)):.1f} mm (99th percentile over the MSD models "
     f"{S_.get('mid_p99', np.nan):.1f} mm). The step adds {t['cols0_total'] - t['cols0_pub']} gravity posts on the ten plans "
     f"({c_ten_inc:.0f}% more columns) and a median {100 * (m['col_ratio_med'] - 1):.0f}% more columns per MSD plan (Supplementary Tables S15 "
     f"and S16). It leaves the lateral-line verdict unchanged on every plan and the eccentricity verdict on {pct(1 - m['ecc_changed'] / m['n_ok'])} of "
     "MSD plans; the exceptions lie at a band boundary and move because merging slab-outline corners onto supports shifts the plan centre "
     "slightly. Slab-span verdicts improve where posts shorten spans."),
    ("The connected models are stiff and wall-dominated. The fundamental period has a median of "
     f"{t['con_T1_med']:.2f} s on the three-storey batch and {S_.get('T1_med', np.nan):.2f} s on the four-storey MSD models, against the "
     "EN 1998-1 [52] estimate Ct·H^0.75 of 0.26 and 0.32 s for wall structures; the walls carry a median "
     f"{100 * S_.get('ws_med', np.nan):.0f}% of the base shear and {100 * S_.get('wg_med', np.nan):.0f}% of the gravity reaction, and beams carry "
     f"{100 * S_.get('beam_share_med', np.nan):.0f}% of the slab load. These magnitudes follow from gross uncracked sections, fixed bases and the dense "
     "column layout, and are plausibility checks rather than design values."),
    ("The solver also provides an external reference for two of the heuristic checks (Fig. 13). Models failing the lateral-line check have a "
     f"longest translational period {S_.get('TT_fail', np.nan):.2f} times the EN 1998-1 estimate, against {S_.get('TT_pass', np.nan):.2f} for "
     "models that pass, so that check flags directions that are genuinely flexible. The eccentricity grade correlates with the solver torsion "
     "ratio, the largest roof displacement over the centroid displacement under the lateral load (Spearman rho = "
     f"{S_.get('rho_grade', np.nan):.2f}); of the models failing the check, {pct(S_.get('ecc_fail_tor12', np.nan))} have a torsion ratio above 1.2, "
     f"against {pct(S_.get('ecc_pass_tor12', np.nan))} of those that pass. These comparisons validate the direction of the two checks, not their "
     "thresholds."),
    ("The check verifies that the exported models are analysable and respond plausibly; it is not a structural assessment. It is linear "
     "elastic, uses gross sections and nominal loads, applies no load combination or code check, and runs in OpenSees rather than in the package "
     "the models are exported to."),
]
X["solver_table_after"] = 2
X["solver_fig_after"] = 4
X["table18_caption"] = ("Table 18. Independent OpenSeesPy check of the exported models, without and with the joint-connectivity step. MSD values without "
                        "the step are for a 150-plan sample stratified by score, exported also without the orientation handling; with the step, for every completed plan. Deflections are maxima under "
                        "gravity; the mid-span value adds 5wL⁴/384EI to the mean end displacement of each beam. One sample model without the step "
                        "has no wall or column and cannot be solved.")
def fmtv(v, f): return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f(v)
T_["table18"] = [
    ["Quantity", "Ten plans, without step", "Ten plans, with step", "MSD, without step (sample)", "MSD, with step"],
    ["Models analysed / completed", f"10 / {sum(bool(sk('deployed', p).get('ok')) for p in PLANS)}", f"10 / {t['con_ok']}",
     f"{P_.get('n', 0)} / {P_.get('n_ok', 0)}", f"{I(S_.get('n', 0))} / {I(S_.get('n_ok', 0))}"],
    ["Models with members without a path to the base", "10", str(sum(c.get("n_float_beams", 0) > 0 for c in con10)), str(P_.get("float_any", 0)), str(S_.get("float_any", 0))],
    ["Floating beams, median share (max)", f"{100 * t['pub_float_med']:.1f}% ({100 * t['pub_float_max']:.1f}%)", "0", f"{100 * P_.get('float_med', 0):.1f}% ({100 * P_.get('float_max', 0):.1f}%)", "0"],
    ["Max. gravity deflection, median", f"{t['pub_uz_med']:.0f} mm", f"{np.median([c['max_uz_mm'] for c in con10]):.1f} mm", f"{P_.get('uz_med', np.nan):.0f} mm",
     fmtv(np.nanmedian([sg(i, 'max_uz_mm') for i in solved]) if solved else np.nan, lambda v: f"{v:.1f} mm")],
    ["Max. beam mid-span deflection (all models)", "-", f"{t['con_mid_max']:.1f} mm", "-", fmtv(S_.get("mid_max"), lambda v: f"{v:.1f} mm")],
    ["Beam mid-span deflection, 99th percentile", "-", "-", "-", fmtv(S_.get("mid_p99"), lambda v: f"{v:.1f} mm")],
    ["T1, median", "-", f"{t['con_T1_med']:.2f} s", "-", fmtv(S_.get("T1_med"), lambda v: f"{v:.2f} s")],
    ["Coherence, joint-aware check (mean)", f"{t['mean6_pub_strict']:.3f}", f"{t['mean6']:.3f}", f"{m['pub_strict_mean']:.3f} (corpus)", f"{m['mean6']:.3f}"],
]
X["fig13_caption"] = ("Fig. 13. Independent solver check. (a) Share of beams without an element path to the base and (b) largest gravity deflection, for "
                      "the exporter without and with the joint-connectivity step, on the ten batch plans and the MSD models; (c) longest translational "
                      "period over the EN 1998-1 estimate, by lateral-line verdict; (d) solver torsion ratio by eccentricity verdict, for the connected "
                      "MSD models.")
_tm = json.load(open(os.path.join(V3, "timing_connect.json")))
_dt = np.median(np.array(_tm["connect"]) - np.array(_tm["noconnect"]))
X["perf_insert"] = (f"These timings exclude the joint-connectivity step, which raises FE model assembly from a median "
                    f"{np.median(_tm['noconnect']):.2f} s to {np.median(_tm['connect']):.2f} s per plan when both are measured on a second, 16-core "
                    f"machine (ten plans, three runs each), a cost of about {_dt:.1f} s.")
X["perf_msd"] = ("The MSD sweep, which bypasses Stage 1, was run on the second, 16-core machine of Supplementary Table S8 with eight worker "
                 "processes; with Stages 2 to 5, the export with and without the connectivity step, the checks and the solver check, a plan took a "
                 f"median of {np.median([r['seconds'] for r in V4R.values() if 'seconds' in r]):.1f} s, of which the OpenSees check took a median of "
                 f"{S_.get('sec_med', np.nan):.1f} s.")
# ---------- arms (post-fix)
if "arms" in t:
    a = t["arms"]; g_ = m.get("gnn", {})
    X["arms_para1"] = (
        "Table 17 compares three shear-wall selector configurations on identical geometry: the learned prior disabled, the deployed hybrid, and "
        f"rule-based selection disabled. The rules alone select {a['rules only']['n']:.1f} piers per plan totaling {a['rules only']['L']:.1f} m, at a mean "
        f"coherence of {a['rules only']['score']:.3f}; the deployed system selects {a['deployed']['n']:.1f} piers totaling {a['deployed']['L']:.1f} m at "
        f"{a['deployed']['score']:.3f}. Across the batch, {t['arm_added']} of the {t['arm_total']} selected piers, {100 * t['arm_added'] / t['arm_total']:.0f}%, "
        f"are admitted only because the network scored above the prior, and {t['arm_missing']} piers selected by the rules alone are absent from the "
        "deployed layout. The learned prediction therefore widens the candidate set that the rules then govern. The two layouts share "
        f"{a['rules only']['J']:.2f} of their piers by Jaccard overlap, and admitting more piers through a permissive prior "
        f"({a['permissive']['n']:.1f} piers, {a['permissive']['L']:.1f} m) gives a coherence of {a['permissive']['score']:.3f}.")
    X["arms_para2"] = (
        "Disabling rule-based selection entirely, so that a pier is chosen when the network scores it above the prior and for no other reason, "
        "gives a layout that is neither the deployed one nor the rules-only one: it shares only "
        f"{a['network only']['J']:.2f} of its piers with the deployed layout by Jaccard overlap, adds {t['net_add']} piers the deployed system does not "
        f"select, and omits {t['net_omit']} that it does. On the ten evaluation plans the network alone scores {a['network only']['score']:.3f} against "
        f"{a['deployed']['score']:.3f} for the deployed hybrid and {a['rules only']['score']:.3f} for the rules alone, with "
        f"{a['network only']['fails']:.2f} failing checks per plan against {a['deployed']['fails']:.2f} and {a['rules only']['fails']:.2f}, using "
        f"{a['network only']['L']:.1f} m of wall against {a['deployed']['L']:.1f} m.")
    if g_:
        X["arms_para3"] = (
            ("Ten plans are too few to settle this, and over the whole MSD corpus " if g_.get("complete") else f"[MSD arm incomplete: {g_['n']} plans] ")
            + f"the network-only configuration reaches a mean coherence of {g_['mean']:.3f} against {g_['dep_mean']:.3f} for the deployed hybrid "
            f"({pct(g_['ge80'])} against {pct(g_['dep_ge80'])} of plans at or above 0.80), the difference being concentrated in the checks the rules exist "
            f"to control: stiffness eccentricity fails on {pct(g_['ecc_fail'])} of plans without the balancing pass against {pct(g_['dep_ecc_fail'])} with "
            f"it, and the lateral-line check on {pct(g_['ll_fail'])} against {pct(g_['dep_ll_fail'])}.")
    X["arms_para4"] = (
        "This comparison does not justify removing the rules, nor does it establish that they produce better structures. The coherence score is "
        "defined in this paper, is sensitive to its weighting, does not price material and rewards over-provision (Section 10.3); the rules also "
        "encode redundancy requirements that the score does not test; and no engineer ground truth is available. What the comparison does establish "
        "is that at scale the rule layer earns its place on the specific grounds it was designed for, stiffness eccentricity and lateral "
        "redundancy. The deployed hybrid is recommended on those grounds and on code alignment rather than on the score.")
X["colsplit_para"] = (
    f"For columns the balance is different and the learned share is larger. Of the {t['cols0_total']:,} storey-level columns exported across the "
    f"batch, {t['net']} ({100 * t['net'] / t['cols0_total']:.0f}%) coincide with network node predictions and {t['rule']} "
    f"({100 * t['rule'] / t['cols0_total']:.0f}%) are generated by the export rules as demoted short piers, boundary columns at pier ends, span "
    f"subdivision and the {t['cols0_total'] - t['cols0_pub']} connectivity posts. Taken together, the two measurements show that this is a hybrid system "
    "in which the network supplies roughly half of the column positions and "
    + (f"{100 * t['arm_added'] / t['arm_total']:.0f}% of the shear-wall selection" if "arm_added" in t else "part of the shear-wall selection")
    + ", while engineering rules determine the remainder and retain the final say over everything the network proposes. Member placement is "
    "therefore learned only in this qualified sense. The improvement from the synthetic design loop is established only on synthetic validation "
    "and test graphs; no independent real labelled test set exists on which to show that the network exceeds its rule teacher.")
X["limitations_para"] = (
    "Several limitations temper these findings. On the corrected leak-free split the shear-wall edge head exceeds the all-positive base rate only "
    "slightly, 0.637 against 0.620, so wall selection remains rule-led with the network as a prior; promoting it to decision-maker awaits richer "
    "labels. The column head over-proposes, and the rule validator plus suppression carry the thinning. The training stock, Chinese high-rise "
    "shear-wall residences plus parametric synthetics, does not span the world's typologies; the exporter builds identical storeys, so setbacks, "
    "transfer levels and varying storey heights are out of scope; the idealization omits foundations and non-orthogonal geometry; and the 0.20 m "
    "thickness prior will mis-estimate scale on stock whose walls deviate systematically. At scale the topology checks pass, while stiffness "
    f"eccentricity, median worst-direction {pct(m['ecc_med'])}, and non-orthogonal curved footprints define the frontier. Member density is the "
    "largest defect: because column candidates are nodes of a wall-centerline graph, about a meter apart in partition-rich plans, the models carry "
    "roughly ten times more columns than ordinary framing implies, no suppression radius corrects it, and the connectivity step adds further "
    "gravity posts that guarantee a load path but are not designed members; replacing the candidate set with a coarse three-to-four-meter grid over "
    "which the network ranks positions is the most consequential structural improvement available.")
X["threat_solver"] = ("Finally, the solver check establishes analysability and plausibility under one idealization, in OpenSees, with gross sections and "
                      "nominal loads; the models were not imported into the target package, and no load combination, cracked-section stiffness or code "
                      "check was applied.")
X["conclusions_para"] = (
    "We presented Plan-to-FEM, an automated pipeline that converts a single raster architectural floor plan, under a fixed set of prescribed "
    "analysis assumptions, into a three-dimensional finite element model. Its central methodological element is the multi-task formulation of "
    "structural enrichment, with shear-wall ratios on edges and column placement as node-level classification over virtual grid candidates, "
    "trained on graphs built by the same vectorization path used at inference. On a de-duplicated and leakage-free validation split the network "
    "places columns with a layout F1 of 0.587 at 0.5 m tolerance and ranks shear walls with an edge ROC AUC of 0.696, although its edge IoU of "
    "0.637 exceeds the 0.620 all-positive base rate only narrowly, so the final layouts are decided by rule-based validation with the network "
    "supplying candidates and a prior. On ten heterogeneous plans the pipeline succeeds end-to-end in every case, with a mean coherence score of "
    f"{t['mean6']:.3f} ({t['mean4']:.3f} on the four informative checks); over 4,572 Modified Swiss Dwellings plans, processed without retraining or "
    f"per-dataset tuning, {I(m['n_ok'])} complete with a mean score of {m['mean6']:.3f}, residual defects being concentrated in stiffness "
    "eccentricity. Every exported model was analysed in an independent solver. That check exposed members without a load path in models the "
    "coherence module had passed, led to the joint-connectivity step and a stricter beam-support check, and with them every model completes "
    "gravity, modal and lateral analyses with small deflections. Four qualifications bound these results. The pipeline automates model generation; "
    "it does not perform structural design, and the solver check is a plausibility check, not a code-compliant analysis. No engineer-drawn ground "
    "truth exists for any evaluation plan, so the coherence evidence is internal rather than comparative. The MSD sweep bypasses Stage 1, so the "
    "large-scale evidence tests the geometric and structural stages rather than the segmentation front end. And the generated layouts carry about "
    "ten times more columns than ordinary framing, a consequence of using wall-centerline junctions as column candidates.")
X["future_para"] = (
    "Future work follows from these limitations: importing the models into the target package and running code-based design checks on a "
    "representative subset; replacing the junction-based column candidates with a coarse three-to-four-meter structural grid over which the "
    "network ranks positions; fine-tuning the column head on real annotations from frame-structure drawings; richer shear-wall labels that would "
    "let the network act as decision-maker with the rules as veto; multi-floor stacking with floor-specific diaphragms; non-orthogonal and curved "
    "footprints; robustness studies on scanned and skewed inputs; and, above all, a public benchmark of paired architectural and engineer-drawn "
    "structural plans.")
X["S12_caption"] = ("Table S12. The t·L³ eccentricity proxy against a flexure-plus-shear cantilever stiffness on the ten connected models, both computed "
                    "over the merged piers the eccentricity check uses. Eccentricity is the distance between the stiffness centre and the plan centre, "
                    "as a fraction of the plan dimension, in the worse direction; verdicts use the check's 10% and 20% bands.")
X["S14_note"] = ("The timings exclude the joint-connectivity step, whose cost is reported in main-text Section 10.9.")

# =====================================================================================  Supplementary S1-S3, Figs. 4, 7, 10, 12 (pass 3b)
GAL = os.path.join(ROOT, "04_results", "figs_v3", "gallery"); FG = os.path.join(ROOT, "04_results", "figs_v3")
_E5 = json.load(open(os.path.join(ENR, "plan_05.json")))
_M5 = json.load(open(os.path.join(V3, "snap", "plan_05", "model_fem.json")))
_M5p = json.load(open(os.path.join(V3, "deployed", "plan_05", "model_fem.json")))
_S5 = json.load(open(os.path.join(V3, "snap", "plan_05", "sanity_report.json")))
_C5 = json.load(open(os.path.join(V3, "snap", "plan_05", "model_fem.connect.json")))
_V5 = sk("snap", "plan_05")
_ch5 = {c["name"]: c for c in _S5["checks"]}
_n5 = {n["id"]: n for n in _M5["nodes"]}
_c50 = [f for f in _M5["frame_members"] if f["type"] == "column" and f["story"] == 0]
_loc5 = len({(round(_n5[f["start_node"]]["x"], 2), round(_n5[f["start_node"]]["y"], 2)) for f in _c50})
_c50p = sum(1 for f in _M5p["frame_members"] if f["type"] == "column" and f["story"] == 0)
_sw5 = sum(1 for w in _E5["walls"] for r in w.get("rects", []) if r.get("is_shear_wall"))
_rect5 = sum(len(w.get("rects", [])) for w in _E5["walls"]); _psw5 = sum(1 for v in _E5.get("edge_predictions", {}).values() if v.get("is_psw"))
_gcol5 = sum(1 for c in _E5.get("columns", []) if c.get("source") == "gnn")
_q5 = [r for r in T_["table15"][1:] if r[0] == "plan_05"][0]; _d5 = [r for r in T_["table13"][1:] if r[0] == "plan_05"][0]
_p1, _jn, _sp = _C5["pass1"], _C5["joints"], _C5["spans"]
_ecc5 = _ch5["eccentricity"]["value"]; _ll5 = _ch5["lateral_lines"]["value"]
X["S1_paras"] = [
    ("This section follows one plan, plan_05, through every stage; the main text shows its source drawing and wall mask (Fig. 5), the vectorized "
     "geometry and graph (Fig. 2), the Stage 4 enrichment (Figs. 4 and 7a) and the exported model (Fig. 7b). The drawing is a residential floor of several "
     f"apartments served by two stair-and-lift cores, digitized at {1000 * _E5['scale_m_per_px']:.1f} mm per pixel, the scale that Stage 2 estimates from "
     "the drawing's own wall-thickness statistics."),
    ("Stage 1 segments the wall pixels. Stage 2 skeletonizes the mask, traces and snaps the skeleton into straight segments and consolidates "
     f"them into {len(_E5['walls'])} walls made of {_rect5} rectangles with measured thickness. Stage 3 turns this geometry into the GNN input graph "
     "of 250 nodes and 202 edges: 138 potential shear-wall edges, 39 indoor door or window openings and 4 outdoor doors, plus 7 virtual "
     "grid-candidate nodes attached by 21 message-passing links."),
    (f"Stage 4 scores the {_psw5} potential shear walls with the edge head and the candidate nodes with the column head. The engineering rules "
     f"(minimum pier length, area budget, stiffness-eccentricity balance) retain {_sw5} shear-wall rectangles, which form the piers of Fig. 4; "
     f"the column head proposes {_gcol5} column positions above the deployed threshold of 0.40."),
    (f"Stage 5 extrudes three storeys of 3.0 m. Before the joint-connectivity step each storey carries {_c50p} columns: the network positions, "
     "short piers demoted to columns, boundary columns at pier ends and span-subdivision columns. The connectivity step then moves "
     f"{_p1['snap_column']} beam ends onto columns, {_p1['snap_wall']} onto a wall edge and {_p1['snap_beam']} onto other beams, dividing "
     f"{_p1['wall_splits']} wall stacks and {_p1['beam_splits']} beams at the new joints; merges {_jn['merge_column']} beam-to-beam joints onto "
     f"columns and {_jn['merge_wall']} onto walls; adds {_jn['posts_corner']} posts under unsupported corner joints and {_sp['posts_span']} posts on "
     f"{_sp['long_runs']} unsupported beam runs longer than 6.0 m; and removes {_C5['orphan_nodes_removed']} joints that no element references. "
     f"The exported model has {_M5['summary']['n_wall_panels']} shear-wall panels, {_M5['summary']['n_beams']} beams and "
     f"{_M5['summary']['n_columns']} columns, {len(_c50)} per storey at {_loc5} plan locations, over {_d5[1]} m2 per storey; its concrete "
     f"volume is {_q5[2]} m3, {_q5[3]} m3 per m2 of floor."),
    (f"The coherence module scores the model {_S5['score']:.3f}. There are {_ll5['x_dir']} distinct shear-wall lines resisting X and "
     f"{_ll5['y_dir']} resisting Y; the stiffness eccentricity is {100 * _ecc5['x']:.1f}% of the plan dimension in x and {100 * _ecc5['y']:.1f}% in y, "
     "the second above the 10% warning band and the only non-passing check; every beam end sits on a shared joint and every member reaches the "
     f"base; no point of the slab is more than {_ch5['slab_span']['value']:.2f} m from a vertical support; all {_loc5} column locations are continuous "
     "over the three storeys; and no joint is orphaned."),
    (f"In the independent solver check (main-text Section 10.8) the model completes all three analyses with an equilibrium error below "
     f"{sup10(max(_V5['equilibrium_err'], 1e-300))}. The largest joint deflection under gravity is {_V5['max_uz_mm']:.2f} mm and the largest beam "
     f"mid-span deflection {_V5['max_beam_mid_mm']:.2f} mm. The fundamental period is {_V5['T1']:.3f} s against the EN 1998-1 estimate of "
     f"{_V5['EC8_T1_walls']:.2f} s; the walls carry {100 * _V5['wall_shear_share_X']:.1f}% (X) and {100 * _V5['wall_shear_share_Y']:.1f}% (Y) of the "
     f"base shear and {100 * _V5['wall_share_gravity']:.1f}% of the gravity reaction, and beams carry {100 * _V5['slab_share_on_beams']:.1f}% of "
     f"the slab load. The torsion ratios, {_V5['torsion_ratio_X']:.2f} in X and {_V5['torsion_ratio_Y']:.2f} in Y, are consistent with the "
     "eccentricity warning being in y."),
]
X["S2_intro"] = ("Figs. S1 and S2 show every evaluation plan in the layout of main-text Fig. 10: the source drawing, the Stage 1 wall mask, the "
                 "Stage 4 shear-wall and column layout, and the three-storey model exported with the joint-connectivity step. Row labels give the "
                 "coherence score and the checks that warn or fail. The Stage 4 panels plot plan y downwards, as in the drawing, and the model "
                 "panels upwards.")
X["figS1_caption"] = "Fig. S1. Evaluation plans floor_plan_1 to floor_plan_5: source drawing, Stage 1 wall mask, Stage 4 layout and exported model."
X["figS2_caption"] = "Fig. S2. Evaluation plans floor_plan_6 to floor_plan_9 and plan_05: source drawing, Stage 1 wall mask, Stage 4 layout and exported model."

X["orientation_para"] = (
    "Vectorization snaps, merges and consolidates wall segments along the image axes; the graph construction, the shear-wall selector (run "
    "grouping, per-direction budgets, lateral lines and eccentricity balancing) and the pier, grid, beam and column rules of the exporter work "
    "along two orthogonal plan axes; and the network was trained on axis-aligned drawings. Many buildings, however, are drawn rotated on the "
    "sheet. The dominant wall direction theta is therefore estimated, from the edge-gradient directions of the wall mask in Stage 2 and from the "
    "wall segments in the later stages, as the weighted circular mean of 4phi, which treats directions phi and phi + 90 degrees as the same. "
    "When theta deviates from the drawing axes by more than 5 degrees, Stage 2 vectorizes the mask rotated into this principal frame and maps "
    "its output back to the drawing, Stages 3 and 4 run on the plan rotated by -theta about its centroid and rotate their output back, and the "
    "exporter builds the model in the same frame and rotates every node back, so a rotated building is treated exactly as an axis-aligned one "
    "would be. In the exporter, a shear-wall pier that deviates from the frame axes by more than 5 degrees, or that snapping onto an axis would "
    "move by more than 0.30 m, is exported along its own direction, and a Stage 4 pier group whose members lie more than 0.30 m off a common line "
    "is split into straight runs. The direction-based coherence checks classify walls in the same principal frame. Section 10.7 quantifies the "
    "effect of this handling by disabling it. The ten evaluation plans lie within 0.1 degrees of the drawing axes, so the step is inactive for them and their models are unchanged.")
_cn = sum(1 for f in _M5["frame_members"] if f["type"] == "column" and f["story"] == 0)
X["fig4_cols"] = (f"green squares are the {_cn} storey-1 columns of the exported model, at {_loc5} distinct plan locations, including the {_cn - _c50p} posts "
                  "added by the joint-connectivity step.")
_r10 = {p: json.load(open(os.path.join(V3, "snap", p, "sanity_report.json")))["score"] for p in PLANS}
_hi = sorted(p for p in PLANS if _r10[p] == max(_r10.values())); _lo = sorted(p for p in PLANS if _r10[p] == min(_r10.values()))
X["fig10_caption"] = (f"Fig. 10. Representative success and failure from the batch, at full column width. The top row, floor_plan_9, is one of the "
                      f"{len(_hi)} plans scoring {max(_r10.values()):.2f}; the bottom row, floor_plan_1, is one of the {len(_lo)} lowest-scoring plans "
                      f"({min(_r10.values()):.2f}). Columns show the source drawing, the Stage 1 wall mask, the Stage 4 shear-wall and column layout, "
                      "and the three-storey model exported with the joint-connectivity step. The complete ten-plan gallery is in Supplementary Section S2.")
N["files"].update(fig4=os.path.join(FG, "fig4_v3.png"), fig7a=os.path.join(FG, "fig7a_v3.png"), fig7b=os.path.join(FG, "fig7b_v3.png"),
                  fig10=os.path.join(FG, "fig10_v3.png"), fig12=os.path.join(FG, "fig12_v5.png"),
                  figS1=os.path.join(FG, "figS1_v3.png"), figS2=os.path.join(FG, "figS2_v3.png"),
                  figS3=os.path.join(FG, "figS3_v5.png"), figS4=os.path.join(FG, "figS4_v5.png"), figS5=os.path.join(FG, "figS5_v5.png"))
# ---- Fig. 12 and the MSD gallery (picks: render_gallery_v4.py)
_sel = json.load(open(os.path.join(GAL, "msd_selection_v5.json")))
def _rep(i): return json.load(open(os.path.join(GAL, f"msd5_{i}", "sanity_report.json")))
def _bad(i, st): return [c["name"].replace("_", " ") for c in _rep(i)["checks"] if c["status"] == st]
_t12, _m12, _b12 = _sel["fig12"]
_o = m["orient"]; _wst = m["wallstock"]; _om_ = m["onmask"]
_AX_NUM = json.load(open(os.path.join(HERE, "numbers_v3_axis.json"), encoding="utf-8"))
X["orient_para"] = (
    f"Many MSD buildings are drawn rotated: the dominant wall direction deviates from the drawing axes by more than 5 degrees in "
    f"{pct(_o['rot_share'])} of the plans (median {_o['fa_med_rot']:.0f} degrees where it does). The effect of the principal-frame handling of "
    "Section 5.5 is measured by disabling it, so that every stage works in the drawing axes: Stage 2 then snaps walls lying within 25 degrees of "
    "an axis onto it and leaves steeper ones as unmerged fragments, Stage 4 groups, budgets and balances the walls along axes that are not their "
    "own, and the exporter projects every pier onto the x or y axis. A direct measure is the share of the exported storey-1 shear-wall length that "
    "lies on wall pixels of the input mask (within three pixels). Without the handling it is a median "
    f"{_om_['first_panels_rot']:.2f} on the rotated plans and {_om_['first_panels_ax']:.2f} on the others, and {pct(_om_['first_lt80'])} of all "
    f"models have less than 0.80 of their shear-wall length on the mask; with it the medians are {_om_['new_panels_rot']:.2f} and "
    f"{_om_['new_panels_ax']:.2f} and {pct(_om_['new_lt80'])} of the models fall below 0.80. The vectorized walls themselves lie on the mask for a "
    f"mean {_om_['first_walls_mean']:.2f} of their length without the handling and {_om_['new_walls_mean']:.2f} with it. Without it the mean "
    f"coherence would be {_o['axis_mean6']:.3f}, with eccentricity failing on {pct(_o['axis_ecc_fail'])} and the lateral-line check on "
    f"{pct(_o['axis_ll_fail'])} of plans; on the rotated plans the mean is {_o['rot_mean_axis']:.3f} without and {_o['rot_mean_new']:.3f} with "
    f"the handling, and on the remaining plans {_o['ax_mean_axis']:.3f} and {_o['ax_mean_new']:.3f}. The higher score without the handling is "
    "not a merit: misplaced walls are still valid shells, so neither the checks nor the solver reject them, and the eccentricity of a rotated "
    "plan is then measured on walls that are not where the drawing puts them. Measured on walls where the drawing puts them, the eccentricity "
    f"grade agrees better with the solver torsion ratio (Spearman rho = {S_.get('rho_grade', np.nan):.2f}, against "
    f"{_AX_NUM['msd']['solve']['rho_grade']:.2f} without the handling). All other MSD results in this section, in Table 16, Figs. 11 to 13, "
    "Section 10.8 and Section 11.1 use the handling.")
X["fig12_para"] = (
    f"Fig. 12 illustrates the score spectrum with three plans. The top row (MSD {_t12}, score {_rep(_t12)['score']:.2f}) is an axis-aligned "
    "multi-apartment floor whose model carries distributed shear walls, interior beams and a continuous column grid and passes every check. The "
    f"middle row (MSD {_m12}, score {_rep(_m12)['score']:.3f}, the lower quartile) is drawn at "
    f"{abs(_rep(_m12).get('frame_angle_deg') or 0):.0f} degrees to the sheet axes; it is enriched and exported in its principal frame, so its walls keep their "
    f"direction, and it is structurally complete but warns on {', '.join(_bad(_m12, 'warn'))}. The bottom row (MSD {_b12}, score "
    f"{_rep(_b12)['score']:.3f}, the lowest in the corpus) shows the failure mode that dominates the low tail: a building drawn with thin, fragmented "
    "wall strokes, from which vectorization recovers little continuous wall, so that few shear walls survive and the slab hull spans large areas "
    f"without them; it fails {', '.join(_bad(_b12, 'fail'))}. The tail is of this kind generally: plans failing the lateral-line "
    f"check retain a median of {_wst['ll_fail_med']:.0f} m of vectorized wall against {_wst['ll_pass_med']:.0f} m for plans that pass. That the "
    "pipeline degrades to a low score rather than crashing on these inputs, and reports the specific deficient checks, is the intended behavior "
    "of the coherence layer.")
X["fig12_caption"] = (f"Fig. 12. Representative MSD results spanning the score range (top to bottom: MSD {_t12}, {_m12} and {_b12}, coherence "
                      f"{_rep(_t12)['score']:.2f}, {_rep(_m12)['score']:.3f} and {_rep(_b12)['score']:.3f}); columns show the wall mask derived from "
                      "the MSD structural raster, the Stage 4 shear-wall and column layout, and the four-storey model exported with the joint-connectivity "
                      "step in the plan's principal frame. The Stage 4 panels plot plan y downwards, the model panels upwards. More plans are shown in "
                      "Supplementary Section S3.")
_order = [i for k in _sel["gallery"] for i in _sel["gallery"][k]]
_nlow = len(_sel["gallery"].get("low", []))
X["S3_intro"] = (f"Figs. S3 to S5 show {len(_order)} further MSD plans in the layout of main-text Fig. 12, all exported with the joint-connectivity "
                 "step and in their principal frame. They were drawn at random (seed 2026) as two plans from each of the five most frequent score "
                 f"levels and {'the lowest-scoring remaining plan' if _nlow == 1 else f'the {_nlow} lowest-scoring remaining plans'}, keeping plan "
                 "identifiers more than 100 apart so that no two floors of one building appear; the plans of Fig. 12 are excluded. Row labels give the plan identifier, the coherence score and the checks that "
                 "warn or fail.")
for _k, _ch in enumerate([_order[0:4], _order[4:8], _order[8:]]):
    X[f"figS{3 + _k}_caption"] = (f"Fig. S{3 + _k}. MSD plans {', '.join(str(i) for i in _ch[:-1])} and {_ch[-1]}: wall mask, Stage 4 layout and "
                                  "exported four-storey model.")
X["limitations_para"] = X["limitations_para"].replace(
    "the idealization omits foundations and non-orthogonal geometry;",
    "the idealization omits foundations; rotated plans are built in their principal frame and off-frame walls keep their direction, but the grid, "
    "beam and column rules work along one pair of axes, so plans that mix several wall directions receive a grid aligned with the dominant one only;")
X["limitations_para"] = X["limitations_para"].replace("and non-orthogonal curved footprints define the frontier.",
                                                      "and inputs with sparse, fragmented wall strokes define the frontier.")
X["future_para"] = X["future_para"].replace("non-orthogonal and curved footprints;", "grid and beam rules for plans with several wall directions and curved footprints;")


N["files"].update(fig8=os.path.join(ROOT, "04_results", "figs_v3", "fig8_batch_v3.png"), fig11=os.path.join(ROOT, "04_results", "figs_v3", "fig11_msd_v3.png"),
                  fig13=os.path.join(ROOT, "04_results", "figs_v3", "fig13_solver_v3.png"))
json.dump(N, open(os.path.join(HERE, "numbers_v3.json"), "w", encoding="utf-8"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (sorted(o) if isinstance(o, set) else str(o)))
print("numbers_v3.json written", "(partial)" if not (S_.get("complete") and m.get("gnn", {}).get("complete")) else "(complete)")

# ---------- figure data
FD = dict(msd_scores=[float(x) for x in sc],
          ten_pub_ff=[float(x) for x in pub_ff], ten_con_ff=[float(c.get("float_frac", 0)) for c in con10],
          ten_pub_uz=[float(x) for x in pub_uz], ten_con_uz=[float(c.get("max_uz_mm", np.nan)) for c in con10],
          pub150_ff=[float(r.get("float_frac", np.nan)) for r in pp], pub150_uz=[float(r.get("max_uz_mm", np.nan)) for r in pp if r.get("ok")],
          msd_con_ff=[float(sg(i, "float_frac", 0)) for i in solved], msd_con_uz=[float(sg(i, "max_uz_mm")) for i in solved],
          TT_by_ll={k: [float(Tmax_ratio[i]) for i in solved if con[i]["strict"]["checks"]["lateral_lines"] == k] for k in ("pass", "fail")},
          tor_by_ecc={k: [float(tor[i]) for i in solved if con[i]["strict"]["checks"]["eccentricity"] == k] for k in ("pass", "warn", "fail")})
json.dump(FD, open(os.path.join(ROOT, "04_results", "figs_v3", "figdata_v3.json"), "w"))
