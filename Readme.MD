# **Project Title:**

**Adaptive and Efficient Framework for No-Reference Image Quality Assessment Using Handcrafted Feature-Driven XGBoost Routing**

---

## **1. Introduction**

Assessing the perceptual quality of an image without a reference (i.e., *no-reference image quality assessment* or NR-IQA) is a fundamental problem in digital image processing. Traditional NR-IQA algorithms such as **BRISQUE**, **NIQE**, and **PIQE** offer fast quality estimation using handcrafted statistical features but often fail under complex, unseen distortions. In contrast, deep learning-based models such as **HyperIQA** and **MANIQA** deliver high correlation with human subjective perception but are computationally expensive and impractical for real-time or large-scale applications.

The challenge arises because no single NR-IQA method performs consistently well across all types of images or distortions. Selecting the best IQA algorithm manually for each use case is time-consuming and requires expert knowledge. Therefore, this project proposes an **adaptive IQA routing framework** that automatically selects the most appropriate NR-IQA method for a given image using a lightweight, data-driven decision mechanism. The system aims to combine the efficiency of traditional methods with the adaptability of modern learning-based models, achieving high accuracy and speed while remaining interpretable and computationally feasible.

---

## **2. Objective**

The main objective of this project is to design and implement a **fast, intelligent, and generalizable IQA routing pipeline** that:

1. Analyzes an input image using a set of low-cost handcrafted features.
2. Predicts which NR-IQA algorithm (traditional or deep) will yield the most accurate perceptual quality score.
3. Runs only the selected IQA algorithm to compute the final quality score, normalized to the **Mean Opinion Score (MOS)** scale.
4. Provides a confidence value and logs data for potential retraining and performance monitoring.

This approach directly addresses the problem of *algorithm selection* in NR-IQA and enhances both computational efficiency and generalizability.

---

## **3. Methodology**

### **3.1 Overview of the Proposed Pipeline**

**Flow:**
`Input Image → Handcrafted Feature Extractor → XGBoost Router → Selected IQA Model → MOS Mapping → Output (Score, Confidence, Selected Model)`

The system operates in two main stages:

1. **Router (Decision Gate):**
   A lightweight XGBoost classifier analyzes handcrafted image features and predicts which IQA algorithm is likely to perform best.

2. **Quality Assessment:**
   The selected IQA algorithm is executed to obtain a raw quality score, which is then mapped to the MOS scale using pre-computed regression parameters.

---

### **3.2 Datasets**

The primary dataset used is **KONIQ-10k**, containing 10,073 images with authentic distortions and associated **human-rated MOS values**.
To assess generalization, cross-dataset testing can be conducted using datasets such as:

* **LIVE Challenge Dataset**
* **SPAQ Dataset**

These datasets ensure diversity in image content and distortion types.

---

### **3.3 Handcrafted Feature Extraction**

To maintain computational simplicity and interpretability, the router uses a compact set (~19) of handcrafted features that efficiently characterize global image quality indicators:

| Category                      | Example Features                                         | Count |
| ----------------------------- | -------------------------------------------------------- | ----- |
| **Global statistics**         | mean luminance, std, skewness, kurtosis, entropy, median | 6     |
| **Colorfulness & saturation** | colorfulness index, mean saturation                      | 2     |
| **Sharpness / blur**          | variance of Laplacian, Tenengrad gradient magnitude      | 2     |
| **Edge density / texture**    | Canny edge ratio, LBP histogram summary                  | 2     |
| **Noise / smoothness**        | residual noise standard deviation                        | 1     |
| **Blockiness / compression**  | block variance difference metric                         | 1     |
| **Contrast / dynamic range**  | RMS contrast, percentile contrast                        | 2     |
| **Frequency energy**          | mean DCT/FFT energy in low, mid, high bands              | 3     |

All features are normalized and standardized before training.
These descriptors collectively capture color, texture, contrast, sharpness, and distortion characteristics, providing enough discriminative power for routing without heavy computation.

---

### **3.4 IQA Model Pool**

The router chooses from five representative NR-IQA algorithms, covering both traditional and learning-based approaches:

| Type                                  | Algorithm    | Description                                                    |
| ------------------------------------- | ------------ | -------------------------------------------------------------- |
| **Traditional (fast)**                | **BRISQUE**  | Natural scene statistics-based model                           |
|                                       | **NIQE**     | Unsupervised quality estimator based on distribution deviation |
|                                       | **PIQE**     | Perception-based blockwise distortion measure                  |
| **Deep Learning (accurate but slow)** | **HyperIQA** | CNN with hyper-network modeling visual sensitivity             |
|                                       | **MANIQA**   | Transformer-based model for authentic distortions              |

This combination ensures a wide quality-accuracy trade-off spectrum.

---

### **3.5 Label Generation for Router Training**

For each image in the KONIQ-10k dataset:

1. All IQA algorithms are executed to obtain raw scores.
2. Each score is linearly mapped to the MOS range using regression on a small validation subset.
3. The absolute error between the predicted MOS and the ground-truth MOS is computed for each IQA method.
4. The IQA method with the **lowest absolute error** becomes the “best performer” label for that image.

This forms a supervised dataset where:

* Input = handcrafted features
* Target = best IQA method label (class index 0–4)

---

### **3.6 Router Model: XGBoost Classifier**

The router is implemented using **XGBoost** due to its efficiency, robustness, and built-in feature importance analysis.

**Configuration:**

```python
from xgboost import XGBClassifier
router = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=5,
    n_jobs=-1,
    random_state=42
)
```

* The model outputs class probabilities for confidence estimation.
* Router accuracy and feature importance are evaluated using accuracy, F1-score, and gain-based importance plots.
* Early stopping and cross-validation ensure robustness.

---

### **3.7 Inference and Deterministic Routing**

During inference:

1. Features are extracted from the input image.
2. XGBoost predicts the best IQA method and provides confidence scores.
3. The top-1 predicted method is **always selected** (deterministic routing).
4. Only the selected IQA method runs, ensuring maximum efficiency.
5. The final output includes:

   ```
   {
       "MOS_estimate": value,
       "Selected_Model": name,
       "Confidence": value,
       "Inference_Time": time_ms,
       "Router_Probabilities": {...}
   }
   ```

**Why Deterministic?**
- The router is trained on thousands of images and already learns the optimal selection
- Running multiple methods when confidence is low adds computational cost without guaranteed accuracy improvement
- Simpler code with single execution path (no branching logic)
- Trusts the router's training to make the best decision every time

This design maximizes both speed and simplicity while maintaining accuracy.

---

### **3.8 MOS Mapping**

Each IQA model’s output is linearly transformed to the MOS scale:
[
MOS_{pred} = a_i \times Score_i + b_i
]
where (a_i) and (b_i) are regression coefficients estimated from the validation split.
This standardization ensures fair comparison and unified scoring.

---

### **3.9 Evaluation Metrics**

Performance is evaluated using:

* **SROCC** – Spearman’s rank correlation between predicted and ground-truth MOS.
* **PLCC** – Pearson correlation to measure linearity.
* **RMSE** – Root Mean Square Error for accuracy.
* **Router Accuracy** – percentage of correct IQA model predictions.
* **Inference Time** – mean latency per image compared to running all IQA models.

Baselines:

1. Always using the best single IQA method (validation-based).
2. Always using the fastest traditional IQA (e.g., BRISQUE).
3. Always using the slowest deep IQA (e.g., MANIQA).

---

### **3.10 Logging and Retraining**

Each inference logs:

```
{image_id, features, selected_model, predicted_MOS, confidence, runtime}
```

This can later be used for incremental retraining or domain adaptation on new data, improving the router over time.

---

## **4. Expected Outcomes**

1. **Accuracy:** The adaptive router should outperform any single fixed IQA model in terms of SROCC/PLCC on diverse datasets.
2. **Speed:** The average inference time should be significantly lower than deep IQA-only systems, as heavy models are used selectively.
3. **Interpretability:** Feature importance plots will reveal which image characteristics influence algorithm selection.
4. **Scalability:** The system can be extended to new IQA methods or domains (e.g., medical, satellite, or low-light imaging) by retraining on corresponding data.

---

## **5. Tools and Technologies**

* **Programming Language:** Python
* **Libraries:** `xgboost`, `numpy`, `pandas`, `scikit-image`, `opencv-python`, `scikit-learn`, `pyiqa`, `matplotlib`, `tqdm`
* **Hardware:** CPU preferred; optional GPU for deep IQA evaluation
* **Dataset:** KONIQ-10k (training), LIVE Challenge / SPAQ (testing)

---

## **6. Project Deliverables**

1. Source code for:

   * Handcrafted feature extractor
   * Router training (XGBoost)
   * Inference pipeline
   * Evaluation scripts
2. Trained router model (`router_xgb.json`)
3. Results & analysis report (correlation tables, runtime comparison, feature importance plots)
4. Research paper/report documenting methodology and findings.

---

## **7. Potential Research Contributions**

* A **new adaptive framework** for NR-IQA algorithm selection.
* Demonstrated **trade-off optimization** between accuracy and computational cost.
* Insights into how **image features correlate** with algorithmic performance.
* Foundation for **dynamic IQA systems** applicable in streaming, mobile, or large-scale image evaluation.

---

## **8. Future Enhancements**

* Incorporate **real-time learning** to adapt router weights on newly collected MOS data.
* Explore **hybrid fusion strategies** (weighted blending of top methods).
* Extend to **video quality assessment (NR-VQA)** using temporal features.
* Integrate **explainability tools** (e.g., SHAP values) for fine-grained feature interpretation.

---

## **9. Conclusion**

This project presents a **computationally efficient and adaptive NR-IQA framework** that bridges traditional feature-based and deep learning-based quality assessment methods through intelligent routing. By leveraging handcrafted image statistics and an XGBoost classifier, the system dynamically selects the most effective IQA algorithm per image, optimizing both accuracy and runtime. The proposed approach eliminates the need for manual IQA selection, provides interpretability through feature importance, and lays the groundwork for scalable, real-world deployment in imaging applications where speed and reliability are critical.
