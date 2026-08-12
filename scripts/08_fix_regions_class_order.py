"""
08_fix_regions_class_order.py

Corrects the regions_class_order field in an existing Dataset002 dataset.json.
Region-based nnU-Net reconstructs a discrete label map from overlapping region
predictions by processing regions in a declared order, with each region
overwriting prior assignments where they overlap; this assumes the declared
order matches the regions' anatomical containment hierarchy, outermost to
innermost. Naive ascending order [1,2,3,4] is WRONG for this label scheme,
since label 1 (NETC) is innermost while label 2 (SNFH) is outermost -- the
reverse of what ascending order assumes. This collapses the whole-tumor
region into label 1 instead of label 2, inflating NETC with misclassified
SNFH voxels and collapsing the tumor-core (TC) score (see paper Section 2.5).

This script requires no retraining -- it only patches dataset.json so that
subsequent inference/reconstruction uses the corrected order. If predictions
were already generated under the buggy order, use
09_reconstruct_regions_from_probs.py on the saved .npz probabilities instead
of re-running inference.
"""
import json
from pathlib import Path

ds_json_path = Path("/workspace/nnUNet_raw/Dataset002_BraTSMETS2026Regions/dataset.json")

with open(ds_json_path) as f:
    ds = json.load(f)

old_order = ds.get("regions_class_order")
ds["regions_class_order"] = [2, 1, 3, 4]

with open(ds_json_path, "w") as f:
    json.dump(ds, f, indent=4)

print(f"regions_class_order updated: {old_order} -> [2, 1, 3, 4]")
print("NOTE: also patch the copy of dataset.json inside each trainer's results")
print("      folder (nnUNet_results/.../dataset.json) -- fixing only the")
print("      nnUNet_raw copy does not propagate to already-trained models'")
print("      inference-time reconstruction.")
