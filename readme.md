# AHT–BCAF: Extended Research Pipeline, Architecture Variants, and Benchmarking Protocol

This document extends the original AHT–BCAF proposal with (1) a fully specified end-to-end
processing pipeline, (2) a family of architecture variants for ablation and comparison,
(3) an expanded baseline suite, and (4) a rigorous, statistically grounded benchmarking
strategy suitable for a Q1 submission (IEEE Access / PLOS ONE).

---

## 1. End-to-End Processing Pipeline

The pipeline has eight stages. Each stage below lists **inputs, operations, outputs, and
the specific design decisions that need to be justified in an ablation.**

### Stage 0 — Data Acquisition & Governance
- **Sources:** ChestX-ray14 (NIH), Guangzhou Women & Children's Medical Center pediatric
  set (Kermany), RSNA Pneumonia Detection Challenge (has bounding boxes — useful for
  explainability grounding), CheXpert (for domain-shift testing), PadChest (external
  validation), and the prospective local hospital collection.
- **Governance:** IRB/ethics approval, de-identification (DICOM tag scrubbing, pixel-level
  burned-in-text removal via OCR + inpainting), patient-level (not image-level) train/val/test
  partitioning to prevent leakage from multiple images of the same patient.
- **Output:** A versioned, de-identified image manifest with patient ID, view position
  (PA/AP/lateral — restrict to PA/AP for consistency), acquisition device, and label
  provenance (radiology report NLP label vs. radiologist-confirmed).

### Stage 1 — Preprocessing & Quality Control
- **Geometric:** Resize to 224×224 (and 320×320 variant for a high-resolution ablation)
  with aspect-preserving padding rather than distortion.
- **Photometric:** CLAHE (contrast-limited adaptive histogram equalization) for lung-field
  contrast normalization; per-dataset intensity histogram matching to reduce inter-scanner
  domain shift.
- **Quality filters:** Automatic rejection of rotated/mislabeled views, lateral views (unless
  a lateral-aware variant is being tested), and images below a Laplacian-variance sharpness
  threshold.
- **View/lung segmentation (optional but recommended):** A lightweight U-Net lung-field
  segmenter constrains CAHT's entropy map to anatomically valid regions, preventing the
  tokenizer from allocating fine tokens to collimator borders or text artifacts.
- **Output:** Cleaned, normalized 224×224 (or 320×320) single-channel tensors with an
  associated lung-field mask.

### Stage 2 — Content-Adaptive Hierarchical Tokenization (CAHT)
- **Input:** Normalized image + lung mask.
- **Operation:** Lightweight Tokenizer-CNN computes a local entropy/gradient saliency map;
  regions are quantized into a token-size map (8×8 fine / 16×16 mid / 32×32 coarse) subject
  to a **token budget constraint** (target ~100 tokens vs. 196 fixed-patch baseline).
- **Design decisions to ablate:** entropy vs. gradient vs. learned saliency; 2-tier vs.
  3-tier granularity; hard token-size assignment vs. soft/differentiable assignment
  (Gumbel-softmax) for end-to-end trainability of the tokenizer itself.
- **Output:** A variable-granularity token sequence with positional encodings that encode
  both location and token scale.

### Stage 3 — Dual-Stream Encoding
- **Local stream (CNN):** Processes the full-resolution image independently of tokenization
  (preserves fine local texture that hierarchical tokenization might smooth over).
- **Global stream (ViT):** Consumes CAHT tokens through a compact transformer.
- **Design decisions to ablate:** backbone choice for the local stream (see Section 2),
  depth/width of the ViT stream, weight sharing vs. independent streams, and whether the
  CNN stream should also receive a low-res global summary (dual-resolution CNN).

### Stage 4 — Bidirectional Cross-Attention Fusion with Channel-Gating (BCA-CG)
- CNN→ViT and ViT→CNN cross-attention as specified in the base proposal, followed by
  learnable channel-wise gating.
- **Design decisions to ablate:** unidirectional vs. bidirectional attention, scalar vs.
  channel-wise gate, number of fusion rounds (single-pass vs. iterative refinement across
  2–3 rounds), and gate conditioning (image-only vs. image + metadata such as age/view).

### Stage 5 — Classification & Loss
- LayerNorm → GELU → Linear head; Focal Loss (γ=2) for class imbalance; label smoothing 0.1.
- **Design decisions to ablate:** Focal Loss vs. class-balanced cross-entropy vs. LDAM loss;
  single binary head vs. multi-label head (co-occurring findings, if using ChestX-ray14's
  14-class labels for a multi-pathology extension).

### Stage 6 — Explainability Layer
- Grad-CAM from the CNN stream, attention-rollout from the ViT stream, and a fused
  saliency map derived from the BCA-CG gate weights (novel: gate-weighted overlay showing
  *which stream* drove each region's decision).
- **Quantitative grounding:** where bounding-box annotations exist (RSNA subset), compute
  IoU / pointing-game accuracy between saliency and ground-truth lesion boxes — this
  converts a qualitative XAI figure into a quantitative, reviewer-defensible metric.

### Stage 7 — Evaluation, Benchmarking & Clinical Validation
- Internal (public-dataset) benchmarking against baselines and ablations (Sections 2–4),
  followed by prospective radiologist-in-the-loop validation on local hospital data, with
  findings feeding back into architecture refinement (dashed loop in the diagram above).

---

## 2. Architecture Variants (Model Family for Ablation)

Beyond the base AHT–BCAF model, define an explicit **variant family** so every claimed
component's contribution is independently verifiable.

| Variant | What changes vs. base | Purpose |
|---|---|---|
| **AHT-BCAF-Base** | As proposed (MobileNetV3-S + 4-layer ViT + bidirectional channel-gated fusion) | Primary model |
| **AHT-BCAF-FixedPatch** | CAHT replaced with standard fixed 16×16 patching | Isolates CAHT's contribution |
| **AHT-BCAF-Uni** | Bidirectional cross-attention → unidirectional (CNN→ViT only) | Isolates bidirectionality's contribution |
| **AHT-BCAF-Concat** | Cross-attention fusion → naive feature concatenation | Isolates cross-attention's contribution vs. prior-art fusion |
| **AHT-BCAF-ScalarGate** | Channel-wise gate → single scalar gate (AAG-style) | Isolates per-channel gating's contribution |
| **AHT-BCAF-Iterative** | Single-pass fusion → 2–3 rounds of iterative cross-attention refinement | Tests whether repeated querying improves fusion |
| **AHT-BCAF-Deep** | ViT stream 4→6 layers, embed dim 256→384 | Upper-bound accuracy under a relaxed (but still <15M param) budget |
| **AHT-BCAF-MultiScale** | Adds a third mid-resolution (16×16) token tier alongside 8×8/32×32 | Tests finer-grained hierarchical tokenization |
| **AHT-BCAF-Backbone-{X}** | Local stream swapped: EfficientNet-Lite0, ShuffleNetV2-1.0x, GhostNet-1.0 | Backbone sensitivity analysis |
| **AHT-BCAF-SSL** | Local + global streams pretrained via masked-image-modeling (MAE-style) on ~100K unlabeled CXRs before fine-tuning | Tests whether domain-specific self-supervised pretraining substitutes for ImageNet pretraining |
| **AHT-BCAF-KD** | Same architecture, trained with knowledge distillation from a large teacher (ConvNeXt-Base or Swin-B hybrid) | Tests whether distillation further improves the lightweight model without changing parameter count |
| **AHT-BCAF-Quant** | Post-training INT8 quantization + ONNX export of AHT-BCAF-Base | Establishes true edge-deployment latency/accuracy trade-off |

**Ablation logic:** each row above changes exactly one component relative to Base, so a
performance delta can be attributed unambiguously (standard controlled-ablation design).
Report all variants with the same metric suite (Section 4) and the same statistical tests.

---

## 3. Expanded Baseline Suite

Organize baselines into five tiers so reviewers can see where AHT-BCAF sits relative to
the full spectrum of prior art, not just hand-picked hybrids.

| Tier | Models | Why included |
|---|---|---|
| **Pure CNN** | ResNet-50, DenseNet-121 (CheXNet reproduction), EfficientNet-B0, MobileNetV3-Small, VGG-16 | Classical and lightweight CNN references |
| **Pure ViT** | ViT-Ti/16, DeiT-Tiny, Swin-Tiny | Transformer-only references at comparable/larger scale |
| **Lightweight Hybrid** | MobileViT-XS, EfficientFormer-L1, CoAtNet-0 | Efficient hybrids — the closest competitive class to AHT-BCAF |
| **Heavyweight Hybrid** | ResNet-50 + ViT-B16 sequential (Slimi et al.-style), Swin + CNN sequential hybrid | Current published performance ceiling (98%+ accuracy band) |
| **Human/Reference** | Individual radiologist reads (inter-rater), majority-vote radiologist panel | Establishes the clinically meaningful upper bound and the kappa baseline |

**Fairness of comparison:** all baselines must be retrained (not copied from the original
papers) on the *same* train/val/test splits, with the *same* augmentation budget and epoch
count as AHT-BCAF, and reported with the *same* compute profiling — otherwise accuracy
comparisons across papers are not valid due to differing preprocessing/splits.

---

## 4. Benchmarking Strategy

### 4.1 Datasets & Splits
- **Development:** ChestX-ray14 and Guangzhou pediatric set, patient-wise 70/10/20
  train/val/test split, 5-fold cross-validation on the training partition for
  hyperparameter selection.
- **Cross-dataset generalization:** train on ChestX-ray14, test zero-shot on Guangzhou
  (and vice versa), and on RSNA/CheXpert/PadChest, to quantify domain shift explicitly
  rather than only reporting in-distribution numbers.
- **Prospective clinical validation:** local hospital collection, held out entirely from
  architecture development, used only in the final validation phase.

### 4.2 Metrics
| Category | Metrics |
|---|---|
| Discrimination | Accuracy, Precision, Recall/Sensitivity, Specificity, F1, AUROC, AUPRC |
| Calibration | Expected Calibration Error (ECE), Brier score, reliability diagrams |
| Agreement | Cohen's κ (model vs. radiologist; radiologist vs. radiologist) |
| Clinical utility | Decision Curve Analysis (net benefit across threshold probabilities) |
| Explainability | Pointing-game accuracy / IoU vs. RSNA bounding boxes; radiologist 1–5 Likert usefulness score |
| Efficiency | Parameter count, FLOPs, peak VRAM (training and inference), CPU/edge-device latency (Jetson Nano / Raspberry Pi 5), throughput (images/sec), training wall-clock time to convergence |
| Robustness | Accuracy under Gaussian noise, JPEG compression artifacts, ±15° rotation, brightness/contrast perturbation, and simulated scanner-domain shift (histogram remapping) |
| Fairness | Subgroup performance by age band, sex, pediatric vs. adult, and image-quality tier |

### 4.3 Statistical Testing
- **Paired classifier comparison:** McNemar's test on paired predictions between AHT-BCAF
  and each baseline.
- **AUC comparison:** DeLong's test for statistically comparing ROC-AUC between correlated
  models on the same test set.
- **Confidence intervals:** 1,000-iteration bootstrap resampling of the test set to report
  95% CIs on all headline metrics, not point estimates alone.
- **Multiple-comparison correction:** Holm–Bonferroni adjustment when comparing AHT-BCAF
  against the full baseline tier (Section 3) simultaneously, to control family-wise error rate.
- **Cross-validation variance:** report mean ± SD across the 5 folds for every ablation
  variant, not a single-seed number — single-seed comparisons are a common reviewer
  rejection reason in this literature.

### 4.4 Computational Benchmarking Protocol
- Fixed hardware profile (RTX 2060, 6 GB) for all training-side comparisons; a separate
  edge-inference profile (Jetson Nano, Raspberry Pi 5, and a mid-range Android SoC) for
  deployment-side comparisons.
- Report VRAM via `nvidia-smi` peak sampling during training, not theoretical estimates.
- Report FLOPs via a standard profiler (e.g., `fvcore` or `ptflops`) at the actual inference
  resolution used (224×224 and 320×320).

### 4.5 Explainability Validation Protocol
1. Generate Grad-CAM, attention-rollout, and the novel gate-weighted fused overlay for
   every test-set positive case.
2. For the RSNA subset (has ground-truth boxes), compute pointing-game accuracy and IoU.
3. For the local hospital set, run the radiologist-in-the-loop session: 2–3 board-certified
   radiologists score explanation usefulness (1–5 Likert) and independently label the same
   held-out set (for inter-rater κ).
4. Perform failure-mode analysis: bucket false negatives by lesion type (lobar, interstitial,
   subtle/early) and correlate with saliency-map quality to identify whether failures are a
   modeling problem or an explainability problem.

---

## 5. Experimental Design Matrix (Summary)

| Axis | Levels tested |
|---|---|
| Tokenization | Fixed 16×16, CAHT 2-tier, CAHT 3-tier (multi-scale) |
| Fusion direction | Unidirectional, bidirectional |
| Fusion mechanism | Concatenation, scalar gate, channel-wise gate, iterative (2–3 rounds) |
| Local backbone | MobileNetV3-S, EfficientNet-Lite0, ShuffleNetV2, GhostNet |
| Pretraining | ImageNet-only, domain-specific SSL (MAE-style) |
| Compression | Full precision, INT8 post-training quantization |
| Resolution | 224×224, 320×320 |
| Dataset | In-distribution, cross-dataset, prospective clinical |

This yields a tractable but thorough ablation grid; not every cell needs to be run — prioritize
the rows in Section 2's table first (single-factor ablations), then run 2–3 combined
"best-of" configurations informed by the single-factor results, rather than a full factorial
sweep.

---

## 6. Implementation Stack (Suggested)

- **Framework:** PyTorch 2.x + `torch.cuda.amp` (mixed precision), gradient checkpointing on
  ViT layers, gradient accumulation for effective batch size 32.
- **Profiling:** `fvcore`/`ptflops` for FLOPs, `nvidia-smi --query-gpu` polling for VRAM,
  `torch.utils.benchmark` for latency.
- **Explainability:** `pytorch-grad-cam` for Grad-CAM, custom attention-rollout
  implementation for the ViT stream.
- **Statistics:** `scipy.stats` (McNemar via `statsmodels`), `scikit-learn` for
  bootstrap CIs and DeLong's test (or the `pROC`/`fastDeLong` reference implementation ported
  to Python).
- **Experiment tracking:** Weights & Biases or MLflow, logging all variants in Section 2 as
  distinct runs with fixed seeds (report 3-seed variance for the top 3 configurations).
- **Deployment/edge export:** ONNX + TensorRT (Jetson) / TFLite (Android) for the
  `AHT-BCAF-Quant` variant.
