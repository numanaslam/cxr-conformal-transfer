# Cross-dataset transfer of conformal abstention thresholds in chest radiograph classification

This README is also provided with the article as Supplemental Article S2.

# 1. Title

**Cross-dataset transfer of conformal abstention thresholds in chest radiograph classification: code and
results.**

| | |
|---|---|
| Article | Cross-dataset transfer of conformal abstention thresholds in chest radiograph classification (PeerJ Computer Science, Research Article) |
| Code and results | GitHub: https://github.com/numanaslam/cxr-conformal-transfer; archived at Zenodo, https://doi.org/10.5281/zenodo.23219340 |
| Codebook | `CODEBOOK.md` in the repository (Supplemental Article S3) |
| License | MIT (code and result files); the datasets keep their own terms |

# 2. Description

The code tests whether conformal abstention thresholds calibrated on one chest radiograph dataset keep
their stated error targets on another dataset, and compares them with thresholds recalibrated on the
deployment dataset. One frozen predictor is used: a BiomedCLIP ViT-B/16 image encoder with nine linear,
Platt-calibrated heads (eight findings and *Normal*), trained on NIH ChestX-ray14 only. Two rules are
evaluated separately:

- **G1 (false deferral, target 0.05):** split conformal prediction on the valve statistic
  s(x) = A(x)(1 - max_k p_k(x)), with A(x) = 1 - p_Normal(x).
- **G2 (confident error, target 0.10):** conformal risk control on the margin |A(x) - tau|; a study is
  accepted when the margin is at least lambda.

Each rule is calibrated at each of three datasets and evaluated at all three (three local settings and
six transfer directions). The repository contains the code, machine-readable result files (JSON) and
console logs of the reported runs, from which every number in the article and in its Supplemental
Information (Article S1) is taken. It contains no images. The scripts are archived exactly as they were
run, so the console logs correspond to this code.

# 3. Dataset Information

The images are not redistributed. All three datasets are third-party, publicly available resources.

| Dataset | Source (URL) | Access terms |
|---|---|---|
| NIH ChestX-ray14 (112,120 frontal images, 30,805 patients) | https://nihcc.app.box.com/v/ChestXray-NIHCC | released by the NIH Clinical Center |
| VinDr-CXR (15,000 labeled training studies) | https://physionet.org/content/vindr-cxr/1.0.0/ ; 512-pixel copy used: https://www.kaggle.com/datasets/awsaf49/vinbigdata-512-image-dataset | PhysioNet credentialed data use agreement |
| Indiana University collection (3,603 coded studies kept) | https://openi.nlm.nih.gov/ ; copy used: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university | CC BY-NC-ND |

Dataset identifiers and descriptor articles:

- **NIH ChestX-ray14:** Wang et al. (2017), https://doi.org/10.1109/CVPR.2017.369.
- **VinDr-CXR:** PhysioNet dataset DOI https://doi.org/10.13026/3akn-b287 (version 1.0.0); Nguyen et al.
  (2022), https://doi.org/10.1038/s41597-022-01498-w.
- **Indiana University collection:** Demner-Fushman et al. (2016), https://doi.org/10.1093/jamia/ocv080.

**Labels.** The vocabulary is the eight findings shared by the three label spaces (Atelectasis,
Cardiomegaly, Consolidation, Pleural Effusion, Infiltration, Mass, Nodule, Pleural Thickening) and
*Normal*. Each builder script writes a per-study label file `labels.csv` in which every finding and
*Normal* is a 0/1 column; `CODEBOOK.md` (Supplemental Article S3) defines every numerically coded field
in these files and in the result files.

**Result files.** Result files are JSON; console logs are plain text.

- Pairs are written `c->d`: the threshold is calibrated at dataset `c` and evaluated at dataset `d`
  (`nih`, `vindr`, `iuxray`).
- `s35_multisite*.json` has a `primary` block (per-site metrics, the nine pairs with rates and bootstrap
  intervals, the domain-classifier AUROC) and a `resplits` block (per-seed values and summaries).
- `s36_resplits.json` has the blocks `primary` (each pair under the symmetric rule, the uncorrected
  margin, the fixed confidence threshold and the one-sided rule at 0.10, 0.05 and 0.02), `redraws`
  (200 repartitions; keys such as `g1|c->d`, `acc|c->d` and `cer|c->d`), `resplits`, `sample_size`
  and `shift`.

# 4. Code Information

| Path | Content |
|---|---|
| `ovcbmr/` | Python package: splits, data loading, frozen encoders, concept heads and Platt scaling, valve statistic, conformal routines (`ood/conformal.py`), metrics (`eval/metrics.py`) |
| `scripts/` | one script per stage (`s00` to `s40`), dataset builders (`build_manifest_nih.py`, `build_manifest_vindr.py`, `build_iuxray.py`) and figure scripts |
| `configs/` | configuration files for each dataset and encoder (`nih`, `vindr`, `iuxray`, `iuxray_nihtax`, and `*_clip`) |
| `tests/test_smoke.py` | checks of the split, the valve and the conformal routines; needs no data |
| `experiments/base_run/` | base pipeline run (all NIH images; Article S1) |
| `experiments/three_dataset/` | one image per patient at NIH; three-dataset transfer matrix |
| `experiments/resplits/` | 20 re-splits and 200 repartitions |
| `experiments/final/` | CLIP encoder, further checks, simulation, network head |
| `RUN_FROM_SCRATCH.md` | step-by-step instructions with downloads and expected counts |
| `CODEBOOK.md` | codes of all categorical fields recorded as numbers (Supplemental Article S3) |

Where each reported result comes from:

**Main article**

| Item | Script | Result file |
|---|---|---|
| Table 2 (the three datasets) | `build_iuxray.py`, `count_manifest.py`, `s35` | `experiments/three_dataset/s35_nihtax_console.txt` |
| Table 3 (which studies fit each quantity) | `s31`, `s35`, `s36` | the code (`_fit_site` in `scripts/s36_resplits.py`) |
| Table 4, primary split | `s35` | `experiments/three_dataset/s35_multisite_nihtax.json` |
| Table 4, re-split means and repartitions above target | `s36` | `experiments/resplits/s36_resplits.json` |
| Fig. 2 | `make_fig_multisite.py` | the two files above |
| Table 5, CLIP encoder | `s35`, `s36` with `*_clip.yaml` | `experiments/final/clip_console.txt` |
| Table 5, one-hidden-layer network | `s39` | `experiments/final/s39_console.txt` |
| Table 5, full Indiana labels | `s35` (full set) | `experiments/three_dataset/s35_multisite.json` |
| Table 6, calibration size, uncorrected margin, fixed threshold, workload | `s36` | `experiments/resplits/s36_resplits.json` |
| Table 6, conditional-error rule and simpler G1 scores; prevalence matching | `s37` | `experiments/final/s37_console.txt` |
| Table 6, importance weighting | `s27` | `experiments/base_run/results/s27_weighted_sweep.json` |
| Predictor performance and calibration | `s09`, `s19`, `s26` | `experiments/base_run/results/` |
| Checks of Propositions 1 and 2 | `s38`, `s40` | `experiments/final/s38_validity_simulation.json`; `s40` prints its result |

**Supplemental Information (Article S1)**

| Item | Script | Result files |
|---|---|---|
| Section S1, results on all NIH images | `s31`, `s33` | `experiments/base_run/results/` |
| One image per patient at NIH | `s31`, `s33`, `s34` | `experiments/three_dataset/` |
| Calibration protocols, per-concept results, selective rule, concept layer | `s09`, `s10`, `s19`, `s20`, `s23`, `s25`, `s30`, `s32` | `experiments/base_run/` |
| Valve statistic, novelty detection, importance weighting | `s05`, `s24`, `s27` | `experiments/base_run/` |
| Simulation (Table S10), G1 with simpler scores (Table S11), calibration size (Table S12) | `s38`, `s37`, `s36` | `experiments/final/`, `experiments/resplits/` |
| Second encoder (Table S13), network head (Table S14), measured shift (Table S15) | CLIP configurations, `s39`, `s35`, `s36` | `experiments/final/`, `experiments/three_dataset/`, `experiments/resplits/` |
| Full results (Tables S16 and S17), label definition (Table S18), workload (Table S19) | `s35`, `s36` | `experiments/three_dataset/`, `experiments/resplits/` |

Notes on the code:

- The package name `ovcbmr` and some script names (`s01_prepare_cxrlt.py`, the `cxrlt` configuration)
  come from an earlier project plan and are kept because the pipeline calls them.
- Some console banners print reference values from an earlier data partition next to the computed
  values. The computed values are the ones reported.
- Stages `s03`, `s04`, `s06` and `s08` evaluate a zero-shot variant that the article does not report.
- If no margin satisfies the conformal risk control condition, lambda is set to infinity (every study is
  deferred). No reported run reached this case.

# 5. Usage Instructions

Commands are run from the repository root. They are written for Windows PowerShell; on Linux or macOS,
replace `^` with `\`. A driver script (`scripts/run_all.py`) runs the base pipeline for NIH
ChestX-ray14 and VinDr-CXR; the later stages are single commands, listed in the order to run them.
`RUN_FROM_SCRATCH.md` gives the downloads and the expected counts at each step.

1. **Install the requirements** (Section 6) and download the three datasets (Section 3).

2. **Smoke test** (no data, no GPU):

    ```
    python tests/test_smoke.py
    ```

3. **Load the datasets and run the base pipeline** (NIH ChestX-ray14 and VinDr-CXR: label files,
   embeddings, heads, and the stages reported in Article S1). `--check` verifies that the data are in
   place before anything is computed:

    ```
    python scripts/run_all.py --check
    python scripts/run_all.py
    ```

4. **Indiana University collection** (matched labels; omit `--nih-taxonomy-only` and change `--out` for
   the full set), then embeddings with stages `s01` and `s02` using `configs/iuxray_nihtax.yaml` and
   `configs/iuxray.yaml`:

    ```
    python scripts/build_iuxray.py --reports <dir>/indiana_reports.csv ^
        --projections <dir>/indiana_projections.csv ^
        --images <dir>/images/images_normalized --img-out <dir>/images_256 ^
        --nih-taxonomy-only --out data/raw/iuxray_nihtax/labels.csv
    ```

5. **Transfer matrix on the primary split** (repeat with `iuxray=configs/iuxray.yaml` for the full
   Indiana labels):

    ```
    python -m scripts.s35_multisite --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --seeds 0 1 2 3 4
    ```

6. **Re-splits, repartitions, comparators and workload; further checks; network head:**

    ```
    python -m scripts.s36_resplits --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --seeds 20 --redraws 200
    python -m scripts.s37_conditional_and_g1 --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --redraws 200
    python -m scripts.s39_nonlinear_g2 --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --seeds 20 --redraws 200
    ```

7. **Second encoder:** rerun stages `s01`, `s02`, `s09`, `s35` and `s36` with the `configs/*_clip.yaml`
   files.

8. **Checks of the conformal rules as implemented** (no data; about 70 seconds and 3 minutes on a CPU):

    ```
    python scripts/s38_validity_simulation.py --reps 4000
    python scripts/s40_exact_crc_check.py
    ```

9. **Figures:**

    ```
    python scripts/make_fig_multisite.py ^
        --json experiments/three_dataset/s35_multisite_nihtax.json ^
        --resplits experiments/resplits/s36_resplits.json ^
        --out fig_transfer.png --print --height-in 4.3
    python -m scripts.s13_figures --config configs/vindr.yaml
    ```

    The first command draws Fig. 2 and the second draws Fig. S1. Fig. 1 is drawn by
    `scripts/make_fig_method_2row.py` from four NIH ChestX-ray14 test images, which are not included and
    are exported from the NIH data with `scripts/export_fig_images.py`.

**Reading the results without rerunning.** Every reported number can be read from the JSON files with any
JSON reader, for example in Python:

```
import json
r = json.load(open("experiments/resplits/s36_resplits.json"))
print(r["redraws"].keys())
```

# 6. Requirements

| Item | Detail |
|---|---|
| Operating system | Windows (GPU machine); the CPU-only checks also run on macOS and Linux |
| Hardware | One NVIDIA GeForce RTX 5070 (12 GB), used only to compute the embeddings (about eight minutes per dataset); all later stages run on the CPU |
| Language | Python 3.10 |
| Python libraries | PyTorch 2.4.1 and torchvision 0.19.1, open_clip_torch 2.26.1, transformers 4.44.2, huggingface_hub 0.25.1, timm 1.0.9, Pillow 10.4.0, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, pandas 2.2.3, PyYAML 6.0.2, tqdm 4.66.5, Matplotlib 3.9.2; pinned in `requirements.txt` and `environment.yml` |
| Model weights | BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, Hugging Face hub); CLIP ViT-B/16 (OpenAI release via open_clip); DenseNet-121 comparator (`densenet121-res224-nih`, TorchXRayVision) |

Install PyTorch first, matched to the local CUDA version, then the remaining packages:

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

# 7. Methodology

**Data processing.**

- NIH ChestX-ray14 labels are mapped to the eight findings and *Normal* (`No Finding`); Effusion is
  mapped to Pleural Effusion (`build_manifest_nih.py`).
- VinDr-CXR box-level annotations are aggregated per study; a finding is positive when any of the three
  readers annotated it, Nodule/Mass sets both Nodule and Mass, and *No finding* sets *Normal*
  (`build_manifest_vindr.py`).
- `build_iuxray.py` keeps one frontal image per coded Indiana report (3,603 studies; reports coded
  "No Indexing" and reports without a readable frontal image are dropped), maps the MeSH headings to the
  vocabulary, sets *Normal* only when "normal" is the sole heading, and converts images to 8-bit grayscale
  with the shorter side at 256 pixels. With `--nih-taxonomy-only` it builds the primary matched set
  (2,261 studies) by leaving out the 1,342 studies that are not normal and carry no NIH-taxonomy finding.
- Images are passed through the encoder's own preprocessing (resizing of the shorter side to 224 pixels,
  a central 224 by 224 crop and channel normalization). No augmentation is used. Each image is embedded
  once; all later computation uses the cached 512-dimensional, l2-normalized embeddings.
- Section S7 of Article S1 gives the label mappings.

**Splits.** These are fixed by a hash of the seed and the patient or study identifier and are not stored
as files. NIH is split by patient (70/10/10/10, seed 42), with one image per patient in the calibration
and test sets; a correct rebuild prints 78,607 / 10,711 / 10,939 / 11,863 images. VinDr-CXR and Indiana
have no patient identifiers and are split by study.

**Modeling.** No hyperparameter search was performed. The settings below are the same at every site and
in every experiment.

| Component | Setting |
|---|---|
| Encoder | BiomedCLIP ViT-B/16 image tower, frozen; 512-dimensional, l2-normalized embedding. Second encoder: CLIP ViT-B/16, frozen |
| Concept heads | one logistic regression per concept (scikit-learn, C = 1.0, class-weight balanced, max 1,000 iterations), fit on the NIH training split |
| Calibration of heads | Platt scaling (one-dimensional logistic regression) on the NIH validation split |
| One-hidden-layer network (robustness check) | scikit-learn MLPClassifier, 256 hidden units, early stopping on a 10% validation fraction, max 200 iterations; outputs Platt-scaled |
| G2 decision threshold tau | value maximizing balanced accuracy over a grid of 201 points on the calibration-site tuning data |
| G1 threshold q-hat | k-th smallest valve statistic among in-vocabulary abnormal calibration studies, k = ceil((n + 1)(1 - 0.05)) |
| G2 margin lambda | first of 201 quantiles of the calibration margins (levels 0, 0.005, ..., 1) satisfying (n R(lambda) + 1)/(n + 1) <= 0.10; infinity if none |
| Domain classifier (shift measure) | logistic regression (C = 1.0) on frozen embeddings, trained on one half and scored on the other |
| Seeds | primary split seed 42; re-splits seeds 0 to 19; 200 repartitions; 1,000 bootstrap resamples, clustered by patient at NIH |

**Calibration protocols.**

- **Clean protocol (primary).** At NIH, the Platt scalars and the decision threshold tau are fit on the
  validation split, and the conformal thresholds on the calibration split. At VinDr-CXR and Indiana, tau
  and lambda are fit on two disjoint halves of the calibration set, and the G1 threshold on the whole of
  it. Thus tau is always fit on studies disjoint from those that fit lambda, as the G2 guarantee
  requires.
- **Coupled protocol (sensitivity only).** All of these quantities are fit on the same calibration split.
  This does not meet the independence condition of the G2 guarantee. It is the `paper` block of
  `s31_clean_protocol.json`, and the supplement marks the analyses that use it.

**Evaluation metrics.**

| Metric | Definition |
|---|---|
| Concept AUROC, ECE | area under the ROC curve and expected calibration error (10 equal-width bins) of each concept head |
| False-deferral rate (G1) | share of in-vocabulary abnormal studies that are deferred |
| Confident-error rate (G2) | share of all studies that are both accepted and misclassified (joint rate) |
| Acceptance | share of studies accepted by the G2 rule |
| Error among accepted | confident-error rate divided by acceptance (conditional error) |
| Above target | share of re-splits or repartitions whose rate exceeds the target |
| Workload | studies cleared as normal, flagged and deferred per 1,000 studies |

A transferred threshold is judged to violate its target when its mean over re-splits exceeds the target,
because the guarantees are marginal and a single rate above the target does not contradict them.

# 8. Citations

Datasets:

- Wang X, et al. 2017. ChestX-ray8: hospital-scale chest X-ray database and benchmarks on
  weakly-supervised classification and localization of common thorax diseases. Proceedings of the IEEE
  Conference on Computer Vision and Pattern Recognition, 2097-2106. DOI: 10.1109/CVPR.2017.369.
- Nguyen HQ, et al. 2022. VinDr-CXR: an open dataset of chest X-rays with radiologist's annotations.
  Scientific Data 9:429. DOI: 10.1038/s41597-022-01498-w. Dataset: DOI 10.13026/3akn-b287.
- Demner-Fushman D, et al. 2016. Preparing a collection of radiology examinations for distribution and
  retrieval. Journal of the American Medical Informatics Association 23:304-310.
  DOI: 10.1093/jamia/ocv080.

Software:

- Harris CR, et al. 2020. Array programming with NumPy. Nature 585:357-362.
  DOI: 10.1038/s41586-020-2649-2.
- Hunter JD. 2007. Matplotlib: a 2D graphics environment. Computing in Science & Engineering 9:90-95.
  DOI: 10.1109/MCSE.2007.55.
- Ilharco G, et al. 2021. OpenCLIP. Software, Zenodo. DOI: 10.5281/zenodo.5143773.
- Paszke A, et al. 2019. PyTorch: an imperative style, high-performance deep learning library. Advances
  in Neural Information Processing Systems 32:8024-8035.
- Pedregosa F, et al. 2011. Scikit-learn: machine learning in Python. Journal of Machine Learning
  Research 12:2825-2830.
- Virtanen P, et al. 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature
  Methods 17:261-272. DOI: 10.1038/s41592-019-0686-2.

Models:

- Zhang S, et al. 2025. A multimodal biomedical foundation model trained from fifteen million image-text
  pairs (BiomedCLIP). NEJM AI 2. DOI: 10.1056/AIoa2400640.
- Radford A, et al. 2021. Learning transferable visual models from natural language supervision (CLIP).
  Proceedings of the 38th International Conference on Machine Learning, PMLR 139:8748-8763.
- Cohen JP, et al. 2022. TorchXRayVision: a library of chest X-ray datasets and models. Proceedings of
  the 5th International Conference on Medical Imaging with Deep Learning, PMLR 172:231-249.

To cite this code, cite the article and the Zenodo archive above; `CITATION.cff` in the repository gives
the citation metadata.

# 9. License and Contribution Guidelines

- **License.** The code and result files are released under the MIT License (`LICENSE`). The datasets
  are not redistributed and keep their own terms: NIH ChestX-ray14 as released by the NIH Clinical
  Center, VinDr-CXR under the PhysioNet credentialed data use agreement, and the Indiana collection under
  CC BY-NC-ND.
- **Contributions.** Questions, bug reports and suggested changes are welcome as GitHub issues or pull
  requests at https://github.com/numanaslam/cxr-conformal-transfer. The archived release is kept exactly
  as used for the article; changes go into later releases with their own version numbers. A pull request
  should describe the change, keep the result files of the reported runs unchanged, and pass
  `python tests/test_smoke.py`.
