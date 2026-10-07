"""Full MSD sweep v5 (orientation fix in Stages 2-5).  Per plan: Stage 2 and consolidation (subprocess) once; Stage 4 in process,
in the principal frame, with the deployed and with the network-only selector.  Deployed: corrected exporter with the joint-connectivity
step and without it (--no-connect), current check, metrics, wall-geometry fidelity against the Stage-4 walls (also of the stored
first-run model against its own Stage-4 output), OpenSees solve.  Network-only: corrected exporter and check."""
import os, sys, json, gzip, shutil, subprocess, time, importlib.util, io, contextlib, logging, math
import numpy as np
REPO = r"C:\Users\chris\Documents\Plan2FEM_revision\03_experiments\run\Plan-to-FEM"; SI = r"C:\Users\chris\Documents\Plan2FEM_revision\we\data\msd\modified-swiss-dwellings-v2\train\struct_in"; SOLVE = r"C:\Users\chris\Documents\Plan2FEM_revision\03_experiments\solve"; OUT = r"C:\Users\chris\Documents\Plan2FEM_revision\03_experiments\msd_full"; RUN = r"C:\Users\chris\Documents\Plan2FEM_revision\03_experiments\run"
PY = sys.executable
logging.disable(logging.CRITICAL)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
sys.path.insert(0, os.path.join(REPO, "etabs")); sys.path.insert(0, SOLVE); sys.path.insert(0, RUN)
X = load("export", os.path.join(REPO, "etabs", "export.py"))
SN = load("sanity", os.path.join(REPO, "etabs", "sanity.py"))
MX = load("metrics", os.path.join(RUN, "metrics.py"))
O = load("opensees_from_fem", os.path.join(SOLVE, "opensees_from_fem.py"))
S4 = load("stage_4_inproc", os.path.join(REPO, "stages", "stage_4", "stage_4.py"))
for f in ("visualize", "visualize_predictions", "visualize_gnn_graph"):
    if hasattr(S4, f): setattr(S4, f, lambda *a, **k: None)
SEL_DEPLOYED = S4.ShearWallSelector.select
_src = open(os.path.join(RUN, "scratch", "stage4_gnnonly.py"), encoding="utf-8").read()
_src = _src[_src.index("def select("):_src.index("stage_4.ShearWallSelector.select")]
_ns = {"np": np, "engineering_thickness": S4.engineering_thickness}
exec(_src, _ns)
SEL_GNNONLY = _ns["select"]
RES = os.path.join(OUT, "res_v5"); RESG = os.path.join(OUT, "arm_gnnonly", "res_v5"); MOD = os.path.join(OUT, "models_v5"); ENR5 = os.path.join(OUT, "enriched_v5")
def call_main(mod, argv):
    old = sys.argv; sys.argv = ["x"] + argv
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return mod.main()
    finally:
        sys.argv = old
def sub(cmd):
    p = subprocess.run([PY] + cmd, cwd=REPO, capture_output=True, text=True, timeout=600)
    if p.returncode: raise RuntimeError(cmd[0] + ": " + (p.stderr or p.stdout).strip()[-300:])
def summ(S):
    return dict(score=S["score"], n_fail=S["n_fail"], n_warn=S["n_warn"], frame_angle_deg=S.get("frame_angle_deg"),
                checks={c["name"]: c["status"] for c in S["checks"]},
                values={c["name"]: c.get("value") for c in S["checks"]})
def _dps(p, a, b):
    ab = b - a; dd = float(ab @ ab)
    if dd < 1e-12: return float(np.linalg.norm(p - a))
    t = min(max(float((p - a) @ ab) / dd, 0.0), 1.0); return float(np.linalg.norm(p - (a + t * ab)))
def fidelity(E, M, min_len=1.2):
    """Exported storey-1 shear-wall panels against the Stage-4 selected shear walls (plan coordinates)."""
    pts = {p["id"]: np.array([p["x"], p["y"]], float) for w in E["walls"] for p in w.get("points", [])}
    S = [(pts[r["p1_id"]], pts[r["p2_id"]]) for w in E["walls"] if not w.get("rejected") for r in w.get("rects", [])
         if r.get("is_shear_wall") and r["p1_id"] in pts and r["p2_id"] in pts]
    N = {n["id"]: np.array([n["x"], n["y"]], float) for n in M["nodes"]}
    P = [(N[w["nodes"][0]], N[w["nodes"][1]]) for w in M["wall_panels"] if w.get("story") == 0 and w.get("is_shear_wall")]
    if not S or not P: return dict(n_panels=len(P), n_s4=len(S))
    offs, angs = [], []
    for a, b in P:
        dists = [max(_dps(a, s0, s1), _dps(b, s0, s1), _dps((a + b) / 2, s0, s1)) for s0, s1 in S]
        k = int(np.argmin(dists)); offs.append(dists[k])
        s0, s1 = S[k]; u = (b - a) / max(np.linalg.norm(b - a), 1e-9); v = (s1 - s0) / max(np.linalg.norm(s1 - s0), 1e-9)
        angs.append(math.degrees(math.acos(min(abs(float(u @ v)), 1.0))))
    Lp = sum(float(np.linalg.norm(b - a)) for a, b in P)
    return dict(n_panels=len(P), n_s4=len(S), off_max=max(offs), off_p95=float(np.percentile(offs, 95)),
                off_mean=float(np.mean(offs)), ang_max=max(angs), frac_off_gt_030=float(np.mean(np.array(offs) > 0.30)),
                panel_len=Lp, s4_len=sum(float(np.linalg.norm(s1 - s0)) for s0, s1 in S))
def export_check(fe, W, tag, extra=(), metrics=True):
    m = W(f"m_{tag}.json"); s = W(f"s_{tag}.json")
    call_main(X, ["-i", fe, "-o", m, "--stories", "4", "--floor-height", "3.0"] + list(extra))
    call_main(SN, ["-i", m, "-o", s]); d = {"strict": summ(json.load(open(s)))}
    if metrics:
        mm = MX.model_metrics(m, s); d["metrics"] = {k: v for k, v in mm.items() if not isinstance(v, (dict, list))}
    cj = os.path.splitext(m)[0] + ".connect.json"
    if os.path.exists(cj): d["connect"] = json.load(open(cj))
    return m, d
def one(i):
    rp = os.path.join(RES, f"{i}.json"); rg = os.path.join(RESG, f"{i}.json")
    if os.path.exists(rp) and os.path.exists(rg): return
    w = os.path.join(OUT, "_w5", f"{os.getpid()}"); shutil.rmtree(w, ignore_errors=True); os.makedirs(w)
    W = lambda f: os.path.join(w, f)
    r = dict(id=i); g = dict(id=i); t0 = time.time()
    try:
        sub(["tools/msd_to_mask.py", "-i", os.path.join(SI, f"{i}.npy"), "-o", W("mask.png")])
        sub(["stages/stage_2/stage_2.py", "-i", W("mask.png"), "-o", W("fo.json")])
        sub(["stages/stage_2/consolidate.py", "-i", W("fo.json"), "-o", W("fc.json")])
        r["frame2"] = json.load(open(W("fo.json"))).get("principal_frame")
        # deployed selector
        S4.ShearWallSelector.select = SEL_DEPLOYED
        if call_main(S4, ["-i", W("fc.json"), "-o", W("fe.json"), "--sw-threshold", "0.10"]) == 2: raise RuntimeError("Stage 4: empty graph")
        shutil.copy(W("fe.json"), os.path.join(ENR5, f"{i}.json"))
        E = json.load(open(W("fe.json")))
        r["frame"] = E.get("principal_frame")
        m, d = export_check(W("fe.json"), W, "con")
        d["fidelity"] = fidelity(E, json.load(open(m)))
        try:
            with gzip.open(os.path.join(OUT, "models", f"{i}.json.gz"), "rt") as f: M0 = json.load(f)
            d["fidelity_axis"] = fidelity(json.load(open(os.path.join(OUT, "enriched", f"{i}.json"))), M0)
        except Exception as e:
            d["fidelity_axis"] = {"err": repr(e)[-200:]}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                so = O.analyse(m, connect="as_exported")
            so.pop("worst_uz", None); d["solve"] = so
        except Exception as e:
            d["solve"] = {"ok": False, "err": repr(e)[-300:]}
        with open(m, "rb") as fi, gzip.open(os.path.join(MOD, f"{i}.json.gz"), "wb") as fo: shutil.copyfileobj(fi, fo)
        r["connected"] = d
        _m, r["noconnect"] = export_check(W("fe.json"), W, "nc", extra=["--no-connect"])
        r["seconds"] = round(time.time() - t0, 1)
        # network-only selector
        t1 = time.time()
        S4.ShearWallSelector.select = SEL_GNNONLY
        call_main(S4, ["-i", W("fc.json"), "-o", W("fg.json"), "--sw-threshold", "0.10"])
        _m, g["connected"] = export_check(W("fg.json"), W, "gnn")
        g["seconds"] = round(time.time() - t1, 1)
    except Exception as e:
        r.setdefault("error", repr(e)[-400:]); g.setdefault("error", repr(e)[-400:])
    finally:
        S4.ShearWallSelector.select = SEL_DEPLOYED
    for path, obj in ((rp, r), (rg, g)):
        json.dump(obj, open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
if __name__ == "__main__":
    for d_ in (RES, RESG, MOD, ENR5, os.path.join(OUT, "_w5")): os.makedirs(d_, exist_ok=True)
    ids = [int(x) for x in open(sys.argv[1]).read().split()]
    n = 0; t0 = time.time()
    for i in ids:
        one(i); n += 1
        if n % 25 == 0: print(n, round(time.time() - t0), flush=True)
    print("done", n, round(time.time() - t0), flush=True)
