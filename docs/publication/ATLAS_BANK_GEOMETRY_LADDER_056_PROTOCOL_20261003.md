# 056: locate where a true anatomical patch loses its atlas-bank match

053 exposed a 21-point loss when an oracle-warped atlas render was replaced by the rigid true plane. 055 then trained the query encoder against every frozen atlas key and improved unknown-plane top-16 recall only from 15.42% to 17.89%; top-one remained about 2%. Before another descriptor or pose-network run, measure the *same query* against a sequence of increasingly deployable atlas patches. This is a frozen numerical diagnosis, not image inspection or model training.

Use the scratch-trained 025 checkpoint and exactly the 185 eligible synthetic DEV sections/eight deformation identities, 32 fixed GT-valid pixels, and both required image parities resolved to the atlas bank's antipodal hemisphere. For each point calculate cosine similarity to:

1. A: atlas patch rendered on the exact synthetic warped CCF surface (the original oracle control).
2. B: rigid patch centered on that point's true CCF coordinate, retaining its exact observed pixel basis and section PSF. This removes local warp while avoiding a centre-offset artifact.
3. C: rigid patch at that same true CCF coordinate, but with the closest discretized bank frame, fixed 67 µm/pixel scale and fixed ±50 µm PSF.
4. D: the frozen actual bank descriptor at the nearest bank-grid position and that same frame.

Compute the query-to-key cosine for each stage, nearest-grid displacement, orientation mismatch and the rank of D among the full atlas bank. Also measure the best of four nearby frames as a diagnostic bound, not an inference label. Equal-weight synthetic-identity means and raw/black/imperfect-brush strata must be retained. A→B isolates local deformation geometry, B→C combines frame/scale/PSF discretization, and C→D isolates bank-grid position quantization; these differences can interact and are not guaranteed to add independently. No synthetic truth may be used to select an inference pose or make a probability claim. Do not touch real final-test animals or public benchmarks.
