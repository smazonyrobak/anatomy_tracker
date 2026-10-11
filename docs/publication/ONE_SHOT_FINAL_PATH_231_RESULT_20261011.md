# 231: fitted final output helps modestly, does not rescue arbitrary-plane pose

The [frozen no-update protocol](ONE_SHOT_FINAL_PATH_231_PROTOCOL_20261011.md) completed on the exact 64 fresh framed-v7 TRAIN-inner sections from the 226 readout. All input hashes and physical-section IDs matched. An independent post-exit comparison against the original 226 rows matched **all 128** direct-prior position, 160-branch oracle and antipodal-normal values exactly (maximum absolute difference 0.0), despite later optional/default-off additions to the model source. Both checkpoints were strictly loaded with the same global plane matcher, and the unchanged `infer_one_shot` path selected among two free and six anchor branches using rendered atlas fit, then returned its one final map. No truth entered inference.

| Checkpoint; median on identical surviving-tissue sites | Direct prior | Fitted final | Truth-best of all 160 direct branches |
| --- | ---: | ---: | ---: |
| 221 parent, rigid CCF error | 3.839 mm | 3.833 mm | 1.663 mm |
| 226 step 1200, rigid CCF error | 3.810 mm | 3.578 mm | 1.560 mm |
| 221 parent, antipodal normal error | 50.88° | 50.30° | — |
| 226 step 1200, antipodal normal error | 47.43° | 42.40° | — |
| 221 parent, local tissue-to-CCF map error | 3.987 mm | 3.871 mm | — |
| 226 step 1200, local tissue-to-CCF map error | 3.847 mm | 3.546 mm | — |

For 226, fitted selection changed the direct branch in 19/64 cases; 38/64 had lower rigid error, but the median **case-paired** rigid gain was only 0.014 mm. The difference between the two separate group medians is 0.232 mm, below the preregistered 0.30-mm rescue threshold. Normal error improved rather than regressed, and local-map median improved, but the conjunctive rescue gate **fails**. Only 16/64 final rigid results were <2 mm (12/64 direct); the 221 parent had 10/64 final versus 11/64 direct. Fitter and more exposure do not turn this into a site-level model.

These numbers are synthetic and TRAIN-inner: the cases are independently drawn physical-plane *simulations* from a small virtual base family, not physically cut unseen animals. The ~47° figure was the direct-prior normal, not the final output; it was neither a radians/sign bug nor the final system's exact angle. The final 42.4° is still genuinely poor. Normal angle also omits AP offset, in-plane roll, raster reflection and local mapping, hence the separate millimetre endpoints. The 221/226 fitter/refiner/map heads were frozen during those pose-exposure runs, so this audit measures the existing feedback machinery rather than newly learned feedback.

The successful output is `I:/AnatomyTracker/runs/one_shot_final_path_231_002`. A first attempt in `_001` wrote rows and receipt but exited with a final `print` formatting exception; it is superseded and not used for the result. The independent post-exit audit matched `_002`'s config, rows and summary hashes to its completion receipt and matched the script/protocol hashes; row count is 128, with no training, reserved DEV/CAL/FINAL access, calibration, GUI change or public benchmark.

**Decision:** do not deploy or scale this pose recipe unchanged. The gap between blind choice (3.58 mm fitted) and truth-best candidate (1.56 mm) remains dominated by selection/appearance evidence. The coordinate-compatible KimLab LSFM source from [230](KIMLAB_LSFM_SOURCE_PREFLIGHT_230_RESULT_20261011.md) is eligible for a small matched source-contrast test, not automatic large-scale training. A targeted [primary-source search](https://figshare.com/articles/dataset/DeepSlice_Source_Data_and_Ground_Truth_datasets/22802411) found no confirmed downloadable, product-rights-clear multi-animal cohort of **physically steep-oblique** sections with independent 3-D pose truth; near-coronal human-annotated sections and virtual oblique reslices cannot establish the desired all-angle claim. Acquisition of that validation source remains a separate necessary gate.
