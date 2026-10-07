"""Batch OpenSees check (opensees_from_fem.analyse, v3) over a list of model files (.json or .json.gz).
Usage: python solve_pass.py <list.txt> <out.jsonl> [connect_mode]"""
import sys, os, json, gzip, io, contextlib, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opensees_check as O
lst, outp = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else "as_exported"
done = set()
if os.path.exists(outp):
    for l in open(outp):
        try: done.add(json.loads(l)["key"])
        except Exception: pass
tmpd = tempfile.mkdtemp()
with open(outp, "a") as fo:
    for line in open(lst):
        key, path = line.rstrip("\n").split("\t")
        if key in done: continue
        p = path
        if path.endswith(".gz"):
            p = os.path.join(tmpd, "m.json"); open(p, "wb").write(gzip.open(path).read())
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                r = O.analyse(p, connect=mode)
        except Exception as e:
            r = {"ok": False, "err": repr(e)[-300:]}
        r.pop("model", None); r["key"] = key
        fo.write(json.dumps(r, default=float) + "\n"); fo.flush()
