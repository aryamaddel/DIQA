# Software Requirements Specification (SRS)  
## Hybrid IQA Regressor: Classical + Deep Fusion  

***

## 1. Introduction  
### 1.1 Purpose  
This SRS defines the requirements for a lightweight image quality assessment (IQA) system that fuses a pretrained deep-IQA model score with classical no-reference metrics (BRISQUE, NIQE, PIQE) via a small regressor, to predict perceptual quality.  

### 1.2 Scope  
The system computes four scalar inputs per image—deep model score, BRISQUE, NIQE, PIQE—and outputs a final quality score correlated with human Mean Opinion Scores (MOS). It supports offline batch evaluation and real-time inference on resource-constrained devices.  

### 1.3 Definitions, Acronyms, and Abbreviations  
-  IQA: Image Quality Assessment  
-  MOS: Mean Opinion Score  
-  BRISQUE: Blind/Referenceless Image Spatial Quality Evaluator  
-  NIQE: Naturalness Image Quality Evaluator  
-  PIQE: Perception based Image Quality Evaluator  
-  CNN: Convolutional Neural Network  
-  MLP: Multi-Layer Perceptron  
-  SROCC/PCC: Spearman/Pearson correlation coefficient  

### 1.4 References  
LIVE Dataset Specification  
TID2013 Dataset Specification  
KADID-10k Dataset Specification  

***

## 2. Overall Description  
### 2.1 Product Perspective  
A modular command-line and library-based tool. Four existing IQA modules feed into a regression module.  

### 2.2 Product Functions  
1. **Image Ingestion:** Load images from disk or camera.  
2. **Classical Metrics Extraction:** Compute BRISQUE, NIQE, PIQE.  
3. **Deep Model Scoring:** Run a pretrained lightweight CNN to produce a deep IQA score.  
4. **Regressor Fusion:** Input the four scores into an MLP or linear regressor to predict final quality.  
5. **Reporting:** Output CSV/JSON with per-image scores and correlation against ground truth.  

### 2.3 User Characteristics  
-  Researchers in IQA and computer vision  
-  Engineers deploying quality monitoring in media streaming, photography, or embedded vision  

### 2.4 Constraints  
-  Total inference per image ≤100 ms on a CPU-only ARM device.  
-  Memory footprint ≤200 MB.  

### 2.5 Assumptions and Dependencies  
-  Pretrained deep IQA model and classical metric libraries (e.g., OpenCV) are available.  
-  Ground-truth MOS available for offline evaluation.  

***

## 3. Specific Requirements  
### 3.1 Functional Requirements  

| ID   | Requirement                                                                                     |
|------|-------------------------------------------------------------------------------------------------|
| FR1  | The system shall accept images in JPEG, PNG, BMP formats.                                       |
| FR2  | The system shall compute BRISQUE, NIQE, and PIQE scores per image.                              |
| FR3  | The system shall load a pretrained CNN model and produce a deep-IQA score per image.           |
| FR4  | The system shall normalize all four input scores to  before regression.                   |
| FR5  | The system shall support at least two regressor types: linear regression and MLP with one hidden layer. |
| FR6  | The system shall train the regressor on MOS-labeled datasets (LIVE, TID2013, KADID-10k).        |
| FR7  | The system shall output final quality scores in CSV or JSON with image identifiers.            |
| FR8  | The system shall report Pearson and Spearman correlation between predicted scores and MOS.     |
| FR9  | The system shall provide a command-line interface supporting “train”, “evaluate”, and “infer” modes. |

### 3.2 Non-Functional Requirements  

| ID    | Requirement                                                                          |
|-------|--------------------------------------------------------------------------------------|
| NFR1  | Average inference time per image shall be ≤100 ms on ARM Cortex-A53 CPU.            |
| NFR2  | Peak memory usage shall be ≤200 MB during inference.                                |
| NFR3  | Training module shall utilize GPU if available, CPU otherwise.                      |
| NFR4  | System shall achieve Pearson correlation ≥0.90 and Spearman correlation ≥0.88 on validation sets. |
| NFR5  | Codebase shall follow PEP8 (for Python) or equivalent style guidelines.             |
| NFR6  | System shall include unit tests covering ≥90% of code.                              |
| NFR7  | All external dependencies shall be specified with version pins in a requirements file.|

***

## 4. System Architecture  
### 4.1 High-Level Components  
1. **Data Loader:** Handles image I/O and ground-truth labels.  
2. **Metric Extractors:** BRISQUE, NIQE, PIQE modules.  
3. **Deep IQA Model:** Pretrained CNN inference component.  
4. **Regressor Module:** Implements linear and MLP regressors.  
5. **Evaluation & Reporting:** Correlation calculators and result exporters.  
6. **CLI Interface:** Orchestrates pipeline stages.  

### 4.2 Data Flow  
Image → Metric Extractors + Deep Model → Normalizer → Regressor → Output → Evaluation  

***

## 5. Data Requirements  
- **Training Data:** LIVE, TID2013, KADID-10k datasets with images and MOS labels.  
- **Model Files:**  
  - pretrained_cnn.pth (Torch model)  
  - scaler.pkl (MinMaxScaler for normalization)  
  - regressor_linear.pkl, regressor_mlp.pkl  
- **Output Files:** CSV/JSON with columns: image_id, brisque, niqe, piqe, deep_score, fused_score  

***

## 6. Use Cases  
### 6.1 Train Regressor  
**Actors:** Researcher  
**Preconditions:** Datasets and pretrained model available  
**Flow:**  
1. `hyb_iqa train --data_dir data/ --model_dir models/ --regressor linear`  
2. System computes metrics, deep scores, trains regressor, saves model files.  

### 6.2 Evaluate Performance  
**Actors:** Researcher  
**Flow:**  
1. `hyb_iqa evaluate --data_dir data/ --model_dir models/ --output report.csv`  
2. System loads regressor, predicts scores, computes correlations, writes report.  

### 6.3 Infer on New Images  
**Actors:** Engineer  
**Flow:**  
1. `hyb_iqa infer --images imgs/ --model_dir models/ --output results.json`  
2. System outputs per-image fused quality scores.  

***

## 7. Validation and Verification  
- **Unit Tests:** For each module (IQA metrics, deep model loader, regressor).  
- **Integration Tests:** End-to-end training and inference on a small subset.  
- **Performance Tests:** Measure inference time and memory on target hardware.  
- **Accuracy Tests:** Verify correlations meet targets on held-out datasets.  

***

## 8. Appendices  
### 8.1 Glossary  
(Defines technical terms)  
### 8.2 External Interfaces  
(API signatures for Python library and CLI)  
### 8.3 Versioning  
Track major/minor SRS revisions.  

---  

*End of SRS*
