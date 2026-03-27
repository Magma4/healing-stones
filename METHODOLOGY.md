# Methodology

Healing Stones is built as a full reconstruction pipeline rather than a collection of isolated experiments. The overall flow is: load and preprocess the fragments, identify likely break surfaces, extract local geometric descriptors, match fragment pairs, refine those matches with ICP, and finally assemble the accepted placements into a global reconstruction. The implementation is intentionally practical. I wanted something that could run on the available dataset and produce inspectable intermediate outputs, even if it is not yet the final word on archaeological fragment reassembly.

The first major design decision was to separate likely break surfaces from original carved or exterior surfaces before matching. This step matters because most of the interesting evidence for adjacency lives on the fracture regions, not on the intact faces. The challenge is that the project does not start with a manually annotated training set. Because of that, the classifier is semi-supervised: it begins from geometric heuristics and pseudo-labels, then trains a Random Forest to stabilize the decision boundary. I chose that route because it is simple, interpretable, and workable with limited labels. A fully supervised deep model would probably be stronger in the long run, but it would also need annotation work that was outside the scope of this iteration.

The second major decision was to make FPFH adaptive instead of using a single global descriptor scale. This became important once I realized the fragments vary dramatically in physical size. Some are small chips on the order of a few centimeters, while others are large slabs spanning hundreds of millimeters. In practice that means the largest fragments are easily more than 20 times the scale of the smallest ones. A voxel size that is meaningful for a small chip is effectively nonsense for a large stele fragment. The final implementation therefore estimates descriptor scale from each fragment’s own break-surface extent and uses that scale in both feature extraction and matching. This was one of the biggest turning points in the project, because it finally made the large fragments participate in the pairwise match graph instead of staying disconnected.

The matching stage uses FPFH descriptors together with multi-scale RANSAC. I added the multi-scale part because even with adaptive descriptors, cross-scale fragment pairs still behaved inconsistently. Some pairs only became visible when the correspondence threshold was relaxed, while others degraded if that threshold was too loose. Trying multiple scales for large-fragment pairs turned out to be a good compromise between robustness and cost. It is more expensive than a single pass, but the number of important large-large comparisons is still small enough to make it worthwhile.

After coarse alignment, each candidate pair is refined with ICP. Standard ICP alone was not sufficient here because the break surfaces are not clean puzzle edges: they have erosion, missing material, and imperfect overlap. That is why the current implementation uses point-to-plane ICP with a Tukey robust kernel. The reason is simple: when there are gap regions or outliers, I want those mismatched points to have reduced influence instead of dominating the least-squares objective. In practice, this made the refinement step much more tolerant of weathered and incomplete fracture surfaces. Increasing the iteration budget at the finest scale also helped some large-fragment pairs settle into better alignments.

The final reconstruction is assembled greedily from the refined match graph. I kept this stage intentionally simple but added post-ICP quality gates so that weak bridges do not control the entire assembly. Seed selection is biased toward physically larger fragments, and disconnected components can be assembled separately before a merge attempt. This is not as principled as a full pose-graph optimization, but it is easy to inspect and it was good enough to reach a full 16-fragment placement in the latest run.

## Key References

- Rusu et al. — *Fast Point Feature Histograms (FPFH) for 3D Registration*
- Besl and McKay — *A Method for Registration of 3-D Shapes*
- Zhou, Park, and Koltun — *Open3D: A Modern Library for 3D Data Processing*
- Fischler and Bolles — *Random Sample Consensus*
- PointNet reference direction: Qi et al. — *PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation*

## Lessons Learned

The first thing that did not work was treating every fragment as if it lived at the same geometric scale. Early versions of the pipeline used descriptor parameters that were effectively tuned to the smaller fragments, which meant the large fragments often produced zero useful matches. That failure mode was not obvious at first because the small-fragment cluster looked internally plausible, but it was hiding the fact that the larger archaeological pieces were being ignored.

Another issue was trusting coarse RANSAC scores too much. Before the robust ICP and post-refinement gates were tightened, the pipeline would happily accept visually convincing but geometrically weak matches. That created crossed-slab assemblies and other locally plausible but globally wrong reconstructions. Tightening the seed logic and making ICP more robust improved that a lot.

I also learned that evaluation metrics can be misleading if they are not calibrated to the actual geometry. The current gap estimate is still useful as a warning sign, but it should not be interpreted too literally. In a future version, I would replace it with a better global consistency measure tied more directly to fracture-surface support and pose-graph residuals.
