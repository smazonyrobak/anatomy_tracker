# 263: extend only the spatial-candidate field-of-view grid

The frozen 262 geometry assay missed its 70% reachability gate by one case:
158/227 were within the proposed local-correction bounds. All 64 fixed normal
anchors provided a normal within 15 degrees, and only seven predicted centres
missed 3 mm; 65 cases missed the physical-span bound. Among 27 sections with
less than 10% surviving tissue, the median predicted minor span was 2.73 mm
against 16.13 mm true. Thus test one predeclared framing repair before training
an image–atlas head: add a third **inference-available** common FOV scale of 3.0
to the 262 scales 1 and 1.8. The centre/normal grid and all correction bounds
remain unchanged. Do not use true pose, mask or visible fraction to choose a
scale at inference; the candidate scorer must later consider all three.

Use the exact 260 checkpoint and frozen 248 DEV panel as 262, with no new
section or appearance. Score the same simultaneous 15-degree normal, 3-mm
centre and 0.7-per-axis log-span limits. Advance to a bounded spatial
image–atlas pilot only if correctability reaches 80% overall, at least 7/8
synthetic plans each reach 70%, and the absolute case gain over 262 is at least
10 percentage points. Report low-tissue coverage separately; even a pass does
not establish that its atlas evidence can select the correct candidate.
No acquired, calibration or final animals, GUI, probability claims or public
benchmark are involved.
