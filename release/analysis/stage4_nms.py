# -*- coding: utf-8 -*-
"""Run Stage 4 as production does, but with the column NMS radius taken from
the NMS_R environment variable instead of the hard-coded 0.60 m."""
import os, sys
ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_4"))
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_3"))
sys.path.insert(0, os.path.join(ROOT, "models"))

R = float(os.environ.get("NMS_R", "0.60"))

import stage_4
_orig = stage_4.GEP.nms_points

def patched(xy, probs, *, min_dist=0.60):
    return _orig(xy, probs, min_dist=R)

stage_4.GEP.nms_points = patched
stage_4.main()
