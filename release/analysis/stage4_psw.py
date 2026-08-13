# -*- coding: utf-8 -*-
"""Run Stage 4 exactly as production does, but with the PSW thickness threshold
taken from the PSW_T environment variable instead of the hard-coded 0.06.

Production code is not modified: the module-level graph builder is replaced
before main() runs, with the same call it makes internally.
"""
import os, sys, json, collections

ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_4"))
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_3"))

TH = float(os.environ.get("PSW_T", "0.06"))
STATS = os.environ.get("PSW_STATS", "")

import stage_4
import stage_3

def patched(floor, proximity_tol_m=0.10, openings=None):
    G = stage_3.build_graph(
        floor,
        openings=openings,
        variant=stage_3.GraphVariant.EDGE_PSW_DW,
        proximity_tol_m=proximity_tol_m,
        auto_detect_gaps=True,
        classify_mode="fixed",
        psw_min_thickness=TH,
    )
    if STATS:
        c = collections.Counter()
        try:
            for _u, _v, d in G.edges(data=True):
                c[str(d.get("edge_type_name", d.get("edge_type", "?")))] += 1
            json.dump({"threshold": TH, "n_nodes": G.number_of_nodes(),
                       "n_edges": G.number_of_edges(),
                       "by_class": dict(c)}, open(STATS, "w"), indent=2)
        except Exception as e:
            json.dump({"threshold": TH, "error": str(e)}, open(STATS, "w"))
    return G

stage_4.build_graph_from_stage2 = patched
stage_4.main()
