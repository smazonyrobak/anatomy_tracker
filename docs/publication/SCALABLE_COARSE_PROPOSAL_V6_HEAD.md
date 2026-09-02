# Scalable coarse proposal v6 head

This standalone engineering head is freshly initialized and has no prior
weights, features, embeddings, pseudolabels, or segmentation dependency. It
does not yet change any runner, inference, refinement, or uncertainty contract.

The head consumes the complete declared catalogue. Fixed spatial pooling bins
retain coarse histology layout instead of reducing the feature map to a global
bag of local features. Plane geometry uses smooth polynomial quotient
invariants: `n outer n`, the closest-plane vector `d n`, and the roll frame
`(v, u outer n)`. These are exactly unchanged by the equivalent frame encoding
`(u,n) -> (-u,-n)` and have no largest-component sign seam.

Each learned component is normalized over all catalogue cells before normalized
mixture weights are applied. The returned full-catalogue probabilities,
component distributions, mixture weights, and entropies are raw and explicitly
uncalibrated. Exact finite rendering, top-M selection, training integration,
posterior semantics, calibration, and versioned run namespaces remain separate
future integration work.
