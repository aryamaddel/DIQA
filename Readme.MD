### Quick summary of decisions

* **Main goals:** usefulness across users + speed.
* **Router:** simple, interpretable, fast classifier using **handcrafted features**.
* **IQA set (initial):** BRISQUE, NIQE, PIQE (fast traditional) + MANIQA, HyperIQA (representative deep NR-IQA).
* **Dataset:** **Train on KONIQ-10k**; **test (cross-evaluate)** on one other realistic dataset (e.g., LIVE Challenge / SPAQ — use whichever is closest in style to KONIQ for cross-testing).
* **Evaluation metrics:** SROCC, PLCC, RMSE for quality prediction; Router accuracy; latency/inference-time comparison; compute savings vs. always-running heavy models.
* **Baseline:** best single IQA method on the validation set (so you can show gains vs. “always use X”).

---

### Concrete pipeline (what you will implement)

1. **Preprocess image** (resize / standardize).
2. **Extract handcrafted features** (fast, detailed below).
3. **Router (classifier)** takes features → predicts one IQA method.
4. **Run chosen IQA** → produce score → **map/normalize** to MOS scale.
5. **Output**: `{MOS_estimate, confidence, selected_model, time_taken}` and **log** for retraining.

---

### Recommended handcrafted feature set (minimal but strong)

Compute these fast features — they capture brightness, contrast, sharpness, noise, color, texture, and basic compression/blockiness cues. Use `scikit-image` + simple NumPy ops.

1. **Global stats (6)**

   * mean(RGB luminance), std(luminance), skewness, kurtosis, entropy, median.

2. **Colorfulness & saturation (2)**

   * colorfulness metric (Hasler & Süsstrunk style), mean saturation.

3. **Sharpness / Blur (2)**

   * variance of Laplacian, Tenengrad (mean gradient magnitude).

4. **Edge density / texture (2)**

   * Canny edge pixel ratio, local binary pattern (LBP) histogram summary (e.g., 8 bins).

5. **Noise / smoothness (1)**

   * estimate noise std via median absolute deviation on high-pass residual.

6. **Blockiness / compression indicator (1)**

   * block artifact energy: compare variance inside 8×8 blocks vs. neighbors.

7. **Contrast & dynamic range (2)**

   * RMS contrast, percentile-based contrast (90th–10th).

8. **Frequency energy bands (3)**

   * mean power in low / mid / high freq bands from a small DCT or FFT patch sample.

**Total ≈ 19–20 features** — small, fast, and interpretable. This keeps training stable and inference tiny.

---

### Router model choice (recommended)

* **Random Forest classifier** (100 trees, max depth 12)

  * Why: robust, fast inference, gives class probabilities (confidence), interpretable feature importances, simple to train with little hyper-tuning.
* If you want slightly smaller memory & faster predict, use **LightGBM** or **XGBoost** with shallow trees.
* Alternative: a tiny MLP (1 hidden layer, 64 units) if you prefer pure NN — but RF is my pick.

**Router output:** predicted best-scorer label + probability vector → use probability as *confidence*.

---

### How to create training labels for router

For each KONIQ-10k image:

1. Compute outputs from every candidate IQA algorithm.
2. Convert each algorithm’s output to the MOS range (simple linear fit on a small validation split).
3. Compute absolute error vs. ground-truth MOS for each method.
4. Label image with the method that has **minimum absolute error** (tie-break by lower runtime or prefer traditional if equal).
   This yields supervised labels for router training.

---

### Confidence & fallback

* If router probability for top class < 0.5, **fallback policy**: run top-2 scorers and **combine** (simple average weighted by router probs).
* This keeps things safe and improves robustness without much complexity.

---

### MOS mapping (simple)

* For each IQA method, compute a **linear regression** (OLS) from the method’s raw score to MOS on a validation split. Save slope & intercept. At inference, apply that mapping. This is simple and effective.

---

### Evaluation plan (experiments to run)

1. **Router accuracy:** % images where router picked the single best method.
2. **Quality metrics:** SROCC, PLCC, RMSE of final MOS estimate — compare:

   * Router system (single chosen method)
   * Router w/ top-2 fusion
   * Each candidate IQA individually (identify best single baseline)
3. **Latency & compute:** average inference time (feature extraction + router + selected IQA) vs. running **only** heavy model (MANIQA or HyperIQA) on all images.
4. **Cross-dataset generalization:** train router on KONIQ-10k; test MOS performance and router behavior on another dataset (LIVE Challenge or SPAQ).
5. **Ablation:** remove one feature group (e.g., no frequency features) to show effect.

---

### Baselines to compare against

* **Best single IQA** from validation (this is the main baseline).
* **Always traditional fast method** (e.g., BRISQUE) — shows gains for edge cases where deep models shine.
* **Always deep model (MANIQA/HyperIQA)** — shows compute vs. accuracy tradeoff.

---

### Logging & retraining

* Log: `{features, selected_model, predicted_score, runtime, image_id}`.
* Periodically (or when logged dataset grows) re-evaluate router with a small retraining step. Keep it simple: retrain RF with incremental dataset.

---

### Practical speed targets (guidelines)

* **Feature extraction + router:** aim for **< 50–100 ms** per image on a modern CPU (depending on image size).
* **Traditional scorers (BRISQUE/PIQE):** tens to a few hundred ms.
* **Deep scorers (MANIQA/HyperIQA):** hundreds of ms to seconds on CPU — so the router should avoid them for most images if not needed.
  (Exact timings will depend on implementation & hardware; these targets help frame the goals.)

---

### Implementation stack & libraries

* Python, `numpy`, `scikit-image`, `opencv`, `scikit-learn` (RandomForest), `lightgbm` (optional), `pyiqa` or `piq` for IQA implementations, `pandas` for logs.
* Use batch processing for initial label generation (run all scorers once per image offline).

---

### Paper / report angles you can highlight

* **Efficiency gains**: slicing heavy-model usage with nearly same (or better) accuracy.
* **Interpretability**: feature importance explains routing decisions.
* **Practicality**: easy-to-integrate pipeline for developers; offline or near-real-time capable.
* **Robustness**: cross-dataset testing shows generalization.

---

### Minimal MVP checklist 

* [ ] Get KONIQ-10k MOS and images.
* [ ] Implement feature extractor (start with the 19 features).
* [ ] Implement BRISQUE, NIQE, PIQE wrappers and run on a subset.
* [ ] Pick one deep IQA (MANIQA or HyperIQA) and run on a subset.
* [ ] Label images by lowest absolute error and train RandomForest.
* [ ] Implement inference pipeline + MOS mapping + logging.
* [ ] Run evaluation vs. baselines and generate table/plots.

