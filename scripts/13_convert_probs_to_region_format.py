"""
13_convert_probs_to_region_format.py

Converts a label-based model's saved 5-channel softmax .npz probabilities
(bg, NETC, SNFH, ET, RC) into 4-channel overlapping region-probability format
(WT, TC, ET, RC), matching the format used for ensembling with region-based
model outputs. WT = NETC+SNFH+ET, TC = NETC+ET, ET = ET, RC = RC.

Edit NPZ_DIR / OUT_DIR for each model you want to convert (e.g. run once for
Primus label-based, once for ResEncL label-based, etc.) before running
15_ensemble_final.py.
"""
import numpy as np
from pathlib import Path

NPZ_DIR = Path("/workspace/predictions_primus_validation")
OUT_DIR = Path("/workspace/ensemble_inputs/primus_labelbased")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for npz_file in sorted(NPZ_DIR.glob("*.npz")):
    case_id = npz_file.stem
    data = np.load(npz_file)
    prob = data["probabilities"]  # shape: (5, X, Y, Z) -- bg, NETC, SNFH, ET, RC

    wt = prob[1] + prob[2] + prob[3]
    tc = prob[1] + prob[3]
    et = prob[3]
    rc = prob[4]

    region_probs = np.stack([wt, tc, et, rc])
    np.save(OUT_DIR / f"{case_id}.npy", region_probs)

print(f"Converted {len(list(NPZ_DIR.glob('*.npz')))} cases -> {OUT_DIR}")
