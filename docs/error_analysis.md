# Error analysis: baseline ResNet-18

3 of 6,000 test images were misclassified, all labelled "no crack" but predicted "crack". Each was inspected manually (227×227 px patches) alongside its Grad-CAM heatmap.

| File | Model confidence | Visual inspection |
|---|---|---|
| (filename 1) | 0.953 | Visible crack line. Likely label error. |
| (filename 2) | 0.575 | Localised crack mark at the patch corner; cannot tell whether it continues beyond the patch. Ambiguous. |
| (filename 3) | 0.991 | Visible crack line. Likely label error. |

## Conclusion
Two of the three errors appear to be label errors (cracks present in images labelled "no crack"), and the third is an ambiguous boundary case where a crack is cut off by the patch edge. The Grad-CAM heatmaps in all three cases concentrate on the crack-like feature, so the model is attending to the right evidence.

## Implications
- The reported 99.95% test accuracy slightly understates performance on correctly labelled data (at most 1 genuine error in 6,000 if the two label errors were corrected).
- Label noise may also exist in the training and validation sets; only misclassified test images were reviewed.
- Patch-boundary cases suggest a real-world pipeline should classify overlapping patches or whole images, not isolated crops.
- Generalisation to other cameras, surfaces and lighting remains untested (see the SDNET2018 issue).