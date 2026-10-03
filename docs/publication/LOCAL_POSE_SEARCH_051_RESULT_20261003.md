# 051 fixed local pose expansion: some additional capture, poor selection

The frozen 019 pose model supplied its actual top-eight hypotheses. Each received 31 fixed full-frame variants (original plus local rotation, translation, in-plane scale and shear moves), giving **248 truth-blind candidate planes per section**. Frozen 050 batch 3,000 scored finite-thickness atlas renders; no model was trained, no image inspected, and no reference coordinates entered proposal generation or selection. The read-only receipt audit passed source/checkpoint/config/output hashes and 241 raw rows: 177 eligible synthetic DEV sections from eight synthetic deformation identities and 64 weak-reference real DEV sections from six donors.

| Equal identity/donor mean | Original top one | Best original eight | 051 selected | Best expanded 248 (oracle) |
| --- | ---: | ---: | ---: | ---: |
| Synthetic rigid tissue error | 2.499 mm | 1.001 mm | **2.374 mm** | 0.724 mm |
| Weak-real five-point disagreement | 0.584 mm | 0.499 mm | 0.577 mm | 0.367 mm |

On synthetic DEV, an expanded candidate was within 0.5 mm on 61/177 sections, but the score selected such a candidate on only 29/177. The expanded oracle improves by 0.278 mm, just below the predeclared 0.3-mm capture requirement; selected error improves only 0.125 mm, far short of the 0.4-mm gain and ≤2.0-mm absolute requirements. The worst weak-real donor regression was +0.143 mm, within the 0.2-mm safety bound. The overall necessary gate **failed**.

The failure is not lack of compute or candidate count alone. Axis-wise changes usually still leave coupled errors, and 050's near-exact-plane recognition did not reliably select better imperfect planes. Do not ship a 248-render search: its accuracy gain is too small for its runtime. The next substantive development should improve global cross-orientation, cross-location image/atlas retrieval with hard anatomical lookalikes and physical correspondence supervision, then test truth-blind full-plane capture. Another scalar score or more local perturbations would repeat a falsified approach. No electrode uncertainty is calibrated; final-test animals and the DeepSlice benchmark remain untouched.

Frozen raw results: `I:/AnatomyTracker/runs/local_pose_search_051_diagnostic`.
