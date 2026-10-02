# Frozen-geometry ranker probe 020: negative result

The prespecified probe trained only the existing fitted candidate scorer for 6,000 batches on 12,000 newly sampled arbitrary-plane synthetic sections. Its parent was the best 019 development checkpoint at batch 18,000. The independent verifier passed all source, checkpoint, draw, evaluation and result hashes; confirmed all training draws came from TRAIN synthetic base subjects rather than DEV identities; and confirmed every non-scorer model tensor was bitwise unchanged. No real weak labels, public benchmark, calibration animals or final-test animals trained the scorer.

| Scorer-only batch | Identity-equal synthetic selected mapping error | Exact-pose mapping error | Real weak-label five-point error |
| --- | ---: | ---: | ---: |
| 0 (019 parent) | **2.591 mm** | 0.129 mm | 0.595 mm |
| 2,000 | 2.773 mm | 0.129 mm | 0.608 mm |
| 4,000 | 2.773 mm | 0.129 mm | 0.584 mm |
| 6,000 | 2.696 mm | 0.129 mm | 0.581 mm |

The 0.25 mm improvement gate failed at every trained checkpoint. Frozen candidate geometry and exact-pose mapping were indeed unchanged. On the synthetic training stream, the scorer's first-500 versus last-500 mean selected mapping error rose from 2.077 to 2.145 mm even as its objective fell slightly from 1.274 to 1.250; stochastic training cases differ across those windows, so these are not paired measurements. The fixed DEV comparison is decisive for this probe: stronger isolated optimization of the same scalar ranker did not improve selection.

Retain 019 batch 18,000 as the best current development checkpoint. Do not deploy it as an excellent replacement or benchmark it against DeepSlice yet. The next evidence-led test should add a different source of anatomical location information, not another continuation of the current scorer. A dense tissue-to-atlas correspondence head, with a weighted plane fit and explicit fit residual, is motivated by 2D/3D correspondence registration such as [Markova et al.](https://arxiv.org/abs/2205.03439) and multimodal registration features such as [CoMIR](https://papers.nips.cc/paper/2020/hash/d6428eecbe0f7dff83fc607c5044b2b9-Abstract.html). This is a hypothesis to test on the frozen DEV data, not a literature-backed guarantee of better mouse histology performance.
