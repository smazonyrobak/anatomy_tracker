# 264: prevent an image-regressor span collapse from deleting all atlas hypotheses

The frozen 263 multiplier-only repair raised geometric coverage from 158 to
164/227, still below its predeclared gate. The 262 failed-span rows show why:
all 65 have a predicted minor FOV below 40% of truth, including 47 below 20%; a
3× multiplier cannot rescue a near-zero prediction. The proposed spatial
atlas-candidate grid should therefore include an **absolute anatomical FOV
prior**, independent of 260's span. Choose one 16×24-mm in-plane span pair
from the known mouse atlas box dimensions, with both raster-axis assignments
available via roll/reflection bins. Do not use true section FOV or visible
tissue at inference. The original 260-derived FOV candidates may remain in a
later model, but this preflight tests the absolute pair alone.

Use the same fixed 64 antipodal normals and predicted-centre offsets {-2,0,+2}
mm as 262–263; the same bounded correction limits are ≤15° unsigned normal,
≤3 mm centre, and ≤0.7 log ratio for each sorted physical span. Run on the
exact frozen 227 eligible DEV sections and 260 checkpoint, without creating
new images. Require ≥90% simultaneous correctability overall, at least 7/8
synthetic plans ≥85%, and at least 50% in sections with <10% visible tissue
before training a spatial atlas updater. This is still an optimistic
truth-measured geometry bound, not evidence that a blind model can recognize
the correct plane or that the fixed FOV transfers to acquired slides.
