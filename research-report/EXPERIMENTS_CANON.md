# Canonical Experiment List (Specific Conditions + Exact Measurements)

This is the authoritative list of experiments that matter for the current report draft.

## Global Conditions (held fixed unless explicitly noted)
- Dataset: `src/hmm_process/artifacts/mess3_mixed_dataset.jsonl`
- Number of sequences used in eval pipelines: `MAX_SEQS=6000`
- Token positions per sequence used for supervision: model `seq_len` (currently 12)
- Split: `train_frac=0.8`, `seed=0`
- Features: raw residual stream vectors (no PCA)
- Model checkpoints compared:
  - trained: `artifacts/tiny_transformer.pt`
  - control: `artifacts/tiny_transformer_random_init.pt`
- Residual stream locations:
  - `final_ln` (pre-unembedding)
  - `layer_0_after_mlp` (after first layer MLP)

---

## E1. Residual -> Simplex, Information-Geometry Log Target (ALR, No PCA)
- Script: `src/compare_alr_kl_nopca.py`
- Conditions:
  - Features: raw residuals (no PCA)
  - Regression target parameterization:
    - 3-way simplex targets (`p0/p1/p2/block`): ALR 2D
    - joint 9-way target: ALR 8D
  - Prediction to simplex: inverse-ALR then probabilities
- Measures (validation):
  - `per_process_kl_val` for process `0,1,2`
  - `joint_8_simplex_kl_val` (9-way block representation)
  - `block_3_simplex_kl_val`
- Outputs:
  - `artifacts/residual_simplex/comparison_trained_vs_control_nopca_alr_kl.json`
  - `artifacts/residual_simplex/comparison_trained_vs_control_nopca_alr_kl_layer1.json`

## E2. Residual -> Simplex, Direct Probability Regression (No PCA, No Log)
- Script: `src/compare_prob_kl_nopca.py`
- Conditions:
  - Features: raw residuals (no PCA)
  - Regression target parameterization: direct probabilities
  - Prediction to simplex: clamp+renormalize projection
- Measures (validation):
  - `per_process_kl_val` for process `0,1,2`
  - `joint_8_simplex_kl_val`
  - `block_3_simplex_kl_val`
- Outputs:
  - `artifacts/residual_simplex/comparison_trained_vs_control_nopca_prob_kl.json`
  - `artifacts/residual_simplex/comparison_trained_vs_control_nopca_prob_kl_layer1.json`

---

## E3. Orthogonality Test in Residual Stream (ALR Maps)
- Script: `src/rowspace_orthogonality_alr_nopca.py`
- Conditions:
  - Fit 4 linear maps from residuals -> ALR coordinates (`p0`, `p1`, `p2`, `block`)
  - Features: raw residuals (no PCA)
  - Done for both checkpoints and both residual locations (`final_ln`, `layer1`)
- Measures:
  - Row-space rank per map
  - Pairwise principal-angle statistics within each checkpoint:
    - `principal_angles_deg`, `mean_angle_deg`, `min_angle_deg`, `max_angle_deg`, `avg_cos2`
  - Cross-checkpoint same-map subspace comparisons:
    - `trained_vs_control_p0`, `..._p1`, `..._p2`, `..._block`
- Output:
  - `artifacts/residual_simplex/rowspace_orthogonality_alr_nopca.json`

## E4. Ground-Truth Coordinate Redundancy Test (No Residuals): ALR->ALR
- Script: `src/alr_to_alr_regression.py`
- Conditions:
  - Source coordinates from ground-truth trajectories only (no transformer residuals)
  - Linear map from ALR of one geometry to ALR of another:
    - sources/targets in `{p0, p1, p2, block}`
  - Convert predicted ALR back to probs before KL
- Measures:
  - `kl_val` per ordered pair (`source_to_target`)
  - `alr_mse_val` per ordered pair
- Output:
  - `artifacts/residual_simplex/alr_to_alr_kl.json`

## E5. Ground-Truth Coordinate Redundancy Test (No Residuals): Prob->Prob
- Script: `src/prob_to_prob_regression.py`
- Conditions:
  - Source/target in direct probability coordinates only
  - Linear map + simplex projection
  - Same ordered pair grid `{p0,p1,p2,block} -> {p0,p1,p2,block}`
- Measures:
  - `kl_val` per ordered pair
  - `prob_mse_val` per ordered pair
- Output:
  - `artifacts/residual_simplex/prob_to_prob_kl.json`

---

## E6. Visual Diagnostics (for qualitative support)
- Ground-truth simplex point clouds (per process + overlay + summary):
  - Script: `src/hmm_process/visualize_mess3_mixture.py`
  - Output dir: `src/hmm_process/artifacts/mess3_mixture_viz`
- Ground-truth ALR coordinate plots:
  - Script: `src/visualize_ground_truth_alr.py`
  - Output dir: `artifacts/residual_simplex/alr_ground_truth`

---

## Primary Claims These Experiments Support
- C1: Factor-level and block-level belief geometries are linearly decodable from residual stream better than random-init control.
  - Supported by E1/E2.
- C2: ALR (information-geometry) is the only canonical log-target parameterization included here.
  - Supported by E1 vs E2.
- C3: The three factor maps are mutually aligned, while block map is much more separated.
  - Supported by E3.
- C4: Low-KL linear transforms exist between factor geometries themselves in ground-truth coordinate space (Mess3-specific redundancy), but not from block to factor.
  - Supported by E4/E5.
