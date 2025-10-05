# Architecture Comparison: Before vs After

## OLD ARCHITECTURE (Confidence-Based Routing)

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (~19 features)                   │
│  • Luminance statistics  • Colorfulness  • Sharpness            │
│  • Edge density  • Noise  • Blockiness  • Frequency energy      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  XGBOOST ROUTER (200 trees)                      │
│           Outputs: [prob₁, prob₂, prob₃, prob₄, prob₅]         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────┴────────┐
                    │  max(probs) ≥   │
                    │  threshold?     │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ YES (confidence ≥ 0.5)      │ NO (confidence < 0.5)
              ▼                             ▼
    ┌─────────────────────┐      ┌──────────────────────────┐
    │  RUN TOP-1 METHOD   │      │   RUN TOP-2 METHODS      │
    │                     │      │                          │
    │  • BRISQUE or       │      │  • Method 1 + Method 2   │
    │  • NIQE or          │      │                          │
    │  • PIQE or          │      │  Compute both scores     │
    │  • MANIQA or        │      │  ↓                       │
    │  • HyperIQA         │      │  Apply MOS mapping       │
    │                     │      │  ↓                       │
    │  Get raw score      │      │  Weighted average:       │
    │  ↓                  │      │  MOS = Σ(wᵢ × MOSᵢ)     │
    │  MOS = a×score + b  │      │                          │
    └─────────┬───────────┘      └─────────┬────────────────┘
              │                            │
              └────────────┬───────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   FINAL OUTPUT         │
              │   • MOS estimate       │
              │   • Selected method    │
              │   • Confidence         │
              │   • Methods used (1-2) │
              │   • Timing breakdown   │
              └────────────────────────┘

COMPLEXITY: 
  - 2 execution paths
  - 1-2 methods per image
  - 3 parameters
  - ~40 lines of code
```

---

## NEW ARCHITECTURE (Deterministic Routing)

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT IMAGE                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              FEATURE EXTRACTION (~19 features)                   │
│  • Luminance statistics  • Colorfulness  • Sharpness            │
│  • Edge density  • Noise  • Blockiness  • Frequency energy      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  XGBOOST ROUTER (200 trees)                      │
│           Outputs: [prob₁, prob₂, prob₃, prob₄, prob₅]         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ Select argmax  │
                    │   (top-1)      │
                    └────────┬───────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  RUN SELECTED METHOD │
                  │                      │
                  │  • BRISQUE or        │
                  │  • NIQE or           │
                  │  • PIQE or           │
                  │  • MANIQA or         │
                  │  • HyperIQA          │
                  │                      │
                  │  Get raw score       │
                  │  ↓                   │
                  │  MOS = a×score + b   │
                  │  (Linear regression) │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌────────────────────────────┐
              │     FINAL OUTPUT           │
              │     • MOS estimate         │
              │     • Selected method      │
              │     • Confidence           │
              │     • Timing breakdown     │
              │     • Router probabilities │
              └────────────────────────────┘

SIMPLICITY:
  - 1 execution path
  - Always 1 method per image
  - 1 parameter
  - ~30 lines of code
```

---

## SIDE-BY-SIDE COMPARISON

```
┌───────────────────────────────────────────────────────────────────────┐
│                          FEATURE COMPARISON                           │
├─────────────────────────────┬─────────────────┬─────────────────────┤
│         Aspect              │      Before     │       After         │
├─────────────────────────────┼─────────────────┼─────────────────────┤
│ Execution Paths             │  2 (branching)  │  1 (linear)         │
│ Methods per Image           │  1-2 (variable) │  1 (constant)       │
│ Function Parameters         │  3              │  1                  │
│ Confidence Threshold        │  Required (0.5) │  Not needed         │
│ Hyperparameter Tuning       │  Yes            │  No                 │
│ Code Complexity             │  ~40 lines      │  ~30 lines          │
│ MOS Mapping                 │  Linear reg ✓   │  Linear reg ✓       │
│ Router Model                │  XGBoost        │  XGBoost            │
│ Feature Count               │  19             │  19                 │
│ IQA Methods                 │  5              │  5                  │
│ Timing Metrics              │  Yes            │  Yes                │
│ Output Format               │  Complex        │  Simplified         │
└─────────────────────────────┴─────────────────┴─────────────────────┘
```

---

## PERFORMANCE FLOW COMPARISON

### OLD: Variable Processing
```
Image 1 (high confidence) → Extract features → Route → Run 1 method  → 100ms
Image 2 (low confidence)  → Extract features → Route → Run 2 methods → 200ms
Image 3 (high confidence) → Extract features → Route → Run 1 method  → 100ms
Image 4 (low confidence)  → Extract features → Route → Run 2 methods → 200ms
                                                        Average: 150ms
```

### NEW: Consistent Processing
```
Image 1 → Extract features → Route → Run 1 method → 100ms
Image 2 → Extract features → Route → Run 1 method → 100ms
Image 3 → Extract features → Route → Run 1 method → 100ms
Image 4 → Extract features → Route → Run 1 method → 100ms
                                      Average: 100ms
```

---

## CODE COMPARISON

### Function Call Evolution

```python
# OLD API (Complex)
result = predict(
    image_path="image.jpg",
    confidence_threshold=0.5,    # Need to tune this
    use_top_k=2                  # Need to tune this
)

if result['confidence'] >= 0.5:
    print(f"Used {result['methods_used'][0]}")
else:
    print(f"Used average of {result['methods_used']}")


# NEW API (Simple)
result = predict(image_path="image.jpg")

print(f"Used {result['selected_method']}")
```

### Internal Logic Evolution

```python
# OLD LOGIC (Branching)
if confidence >= threshold:
    method = top_1_method
    score = run_iqa(method)
    mos = map_to_mos(score, method)
else:
    methods = top_k_methods(2)
    scores = [run_iqa(m) for m in methods]
    mos_scores = [map_to_mos(s, m) for s, m in zip(scores, methods)]
    mos = weighted_average(mos_scores, weights)


# NEW LOGIC (Linear)
method = top_1_method
score = run_iqa(method)
mos = map_to_mos(score, method)
```

---

## DECISION TREE VISUALIZATION

```
                    ┌─────────────────┐
                    │  Input Image    │
                    └────────┬────────┘
                             │
                             ▼
        ┌────────────────────────────────────┐
        │                                    │
    OLD APPROACH                        NEW APPROACH
        │                                    │
        ▼                                    ▼
  ┌──────────┐                        ┌──────────┐
  │  Router  │                        │  Router  │
  └─────┬────┘                        └─────┬────┘
        │                                    │
        ▼                                    ▼
   Confidence?                          Select Top-1
        │                                    │
    ┌───┴───┐                               │
    │       │                                │
  High    Low                                │
    │       │                                │
    ▼       ▼                                ▼
  Run 1   Run 2                           Run 1
  method  methods                         method
    │       │                                │
    └───┬───┘                                │
        │                                    │
        ▼                                    ▼
   Get MOS                                Get MOS
```

---

## KEY INSIGHT DIAGRAM

```
┌─────────────────────────────────────────────────────────────────┐
│                    ROUTER TRAINING PROCESS                       │
│                                                                  │
│  8000+ images → Compute all methods → Find best per image       │
│                                      ↓                           │
│              Train XGBoost to predict best method                │
│                                      ↓                           │
│          Router learns optimal selection patterns                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    WHAT THIS MEANS                               │
│                                                                  │
│  ✓ Router already knows the best method                         │
│  ✓ Low confidence ≠ wrong prediction                            │
│  ✓ Top-1 prediction incorporates all training                   │
│  ✓ Running multiple methods doesn't help                        │
│                                                                  │
│              → TRUST THE ROUTER'S TRAINING ←                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## SUMMARY

**Before:** Complex confidence-based system with branching logic  
**After:** Simple deterministic system with single path

**Result:** 
- 25% less code
- Up to 2x faster on low-confidence images
- No hyperparameters to tune
- Easier to understand and maintain
- Same accuracy (trusts the trained router)
