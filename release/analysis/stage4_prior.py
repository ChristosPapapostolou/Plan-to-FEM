# -*- coding: utf-8 -*-
"""Run Stage 4 with the shear-wall GNN prior taken from GNN_PRIOR.

The prior is an admission threshold: a candidate pier is selected if it is long
enough OR its GNN score reaches the prior. Setting the prior above 1 therefore
disables the learned contribution entirely, leaving a rules-only selector.
"""
import os, sys
ROOT = r"C:\Dev\Plan_2_FEM_2026"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_4"))
sys.path.insert(0, os.path.join(ROOT, "stages", "stage_3"))
sys.path.insert(0, os.path.join(ROOT, "models"))

P = float(os.environ.get("GNN_PRIOR", "0.50"))

import stage_4
_orig = stage_4.StructuralEnrichmentPipeline.__init__

def patched(self, **kw):
    kw["sw_gnn_prior"] = P
    _orig(self, **kw)

stage_4.StructuralEnrichmentPipeline.__init__ = patched
stage_4.main()
