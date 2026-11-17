# Cross-dataset transfer of conformal abstention thresholds in chest radiograph classification

Code and results for the article by Numan Aslam, Ghulam Mustafa and Zia Ul Rehman, submitted to
*PeerJ Computer Science*. This README is also provided with the article as Supplemental Article S2.

## Article information

| | |
|---|---|
| Article | Cross-dataset transfer of conformal abstention thresholds in chest radiograph classification |
| Journal | PeerJ Computer Science (Research Article) |
| Authors | Numan Aslam (1), Ghulam Mustafa (1), Zia Ul Rehman (2) |
| Affiliations | (1) Department of Computer Sciences, University of Central Punjab, Lahore, Pakistan; (2) Department of Computer Science, Government College University Lahore, Lahore, Pakistan |
| Corresponding author | Numan Aslam, l1f20phdc0002@ucp.edu.pk |
| Code and results | GitHub: https://github.com/numanaslam/cxr-conformal-transfer; archived at Zenodo, https://doi.org/10.5281/zenodo.23219340 |
| License | MIT (code and result files); the datasets keep their own terms |

This README describes the code, the result files and the third-party data needed to reproduce every
number in the article and in its Supplemental Information (Article S1). The archive contains code,
machine-readable result files (JSON) and console logs. It contains no images.

## How this repository meets the PeerJ Computer Science requirements

| PeerJ requirement | Where it is met |
|---|---|
| All code needed to reproduce the results available at submission, in a repository with a Zenodo DOI | Code and results (table above); the archive is an exact copy of the code used, see Archive contents |
| Raw data in a machine-readable format | Result files are JSON (see Result files); third-party images are obtained from their sources |
| Third-party datasets named with their source (DOI or URL) | Third-party datasets |
| Data preprocessing described | Third-party datasets, Preprocessing |
| Computing infrastructure (operating system, hardware, software) | Computing infrastructure |
| Evaluation methods, assessment metrics and model selection | Model fitting and selection; Evaluation metrics |
| Reproduction script | How to reproduce the results |
| Third-party code cited, with a DOI where available | Third-party software and references |

## Purpose of the code

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
six transfer directions).

## Third-party datasets

The images are not redistributed. All three datasets are third-party resources.

| Dataset | Source (URL) | Access terms |
|---|---|---|
| NIH ChestX-ray14 (112,120 images) | https://nihcc.app.box.com/v/ChestXray-NIHCC | released by the NIH Clinical Center |
| VinDr-CXR (15,000 labeled training studies) | https://physionet.org/content/vindr-cxr/1.0.0/ ; 512-pixel copy used: https://www.kaggle.com/datasets/awsaf49/vinbigdata-512-image-dataset | PhysioNet credentialed data use agreement |
| Indiana University collection | https://openi.nlm.nih.gov/ ; copy used: https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university | CC BY-NC-ND |

Dataset identifiers and descriptor articles:

- **NIH ChestX-ray14:** Wang et al. (2017), https://doi.org/10.1109/CVPR.2017.369.
- **VinDr-CXR:** PhysioNet dataset DOI https://doi.org/10.13026/3akn-b287 (version 1.0.0); Nguyen et al.
  (2022), https://doi.org/10.1038/s41597-022-01498-w.
- **Indiana University collection:** Demner-Fushman et al. (2016), https://doi.org/10.1093/jamia/ocv080.

**Preprocessing.**

- Frontal images are resized to 224 pixels and embedded once with the frozen encoder; all later
  computation uses the cached 512-dimensional embeddings.
- NIH labels are mapped to eight findings and *Normal*. A VinDr-CXR finding is positive when any of its
  three readers annotated it.
- `scripts/build_iuxray.py` keeps one frontal image per coded Indiana report (3,603 studies). With
  `--nih-taxonomy-only` it builds the primary matched set (2,261 studies) by leaving out the 1,342
  studies that are not normal and carry no NIH-taxonomy finding.
- The label mappings are given in Section S7 of Article S1.

**Splits.** These are fixed by a hash of the seed and the patient or study identifier and are not stored
as files.

- NIH is split by patient (70/10/10/10, seed 42), with one image per patient in the calibration and test
  sets. A correct rebuild prints 78,607 / 10,711 / 10,939 / 11,863 images.
- VinDr-CXR and Indiana have no patient identifiers and are split by study.

## Computing infrastructure

| Item | Detail |
|---|---|
| Operating system | Windows (GPU machine); the CPU-only checks also run on macOS and Linux |
| Hardware | One NVIDIA GeForce RTX 5070 (12 GB), used only to compute the embeddings (about eight minutes per dataset); all later stages run on the CPU |
| Language | Python 3.10 |
| Main packages | open_clip_torch 2.26.1, NumPy 1.26.4, SciPy 1.14.1, scikit-learn 1.5.2, Matplotlib 3.9.2, PyTorch with CUDA 12.8 support; all versions pinned in `requirements.txt` |
| Model weights | BiomedCLIP (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`, Hugging Face hub); CLIP ViT-B/16 (OpenAI release via open_clip); DenseNet-121 comparator (`densenet121-res224-nih`, TorchXRayVision) |

## Model fitting and selection

No hyperparameter search was performed. The settings below are the same at every site and in every
experiment.

| Component | Setting |
|---|---|
| Encoder | BiomedCLIP ViT-B/16 image tower, frozen; images resized to 224 pixels; 512-dimensional, l2-normalized embedding. Second encoder: CLIP ViT-B/16, frozen |
| Concept heads | one logistic regression per concept (scikit-learn, C = 1.0, class-weight balanced, max 1,000 iterations), fit on the NIH training split |
| Calibration of heads | Platt scaling (one-dimensional logistic regression) on the NIH validation split |
| One-hidden-layer network (robustness check) | scikit-learn MLPClassifier, 256 hidden units, early stopping on a 10% validation fraction, max 200 iterations; outputs Platt-scaled |
| G2 decision threshold tau | value maximizing balanced accuracy over a grid of 201 points on the calibration-site tuning data |
| G1 threshold q-hat | k-th smallest valve statistic among in-vocabulary abnormal calibration studies, k = ceil((n + 1)(1 - 0.05)) |
| G2 margin lambda | first of 201 quantiles of the calibration margins (levels 0, 0.005, ..., 1) satisfying (n R(lambda) + 1)/(n + 1) <= 0.10; infinity if none |
| Domain classifier (shift measure) | logistic regression (C = 1.0) on frozen embeddings, trained on one half and scored on the other |
| Seeds | primary split seed 42; re-splits seeds 0 to 19; 200 repartitions; 1,000 bootstrap resamples, clustered by patient at NIH |

## Evaluation metrics

| Metric | Definition |
|---|---|
| Concept AUROC, ECE | area under the ROC curve and expected calibration error of each concept head |
| False-deferral rate (G1) | share of in-vocabulary abnormal studies that are deferred |
| Confident-error rate (G2) | share of all studies that are both accepted and misclassified (joint rate) |
| Acceptance | share of studies accepted by the G2 rule |
| Error among accepted | confident-error rate divided by acceptance (conditional error) |
| Above target | share of re-splits or repartitions whose rate exceeds the target |
| Workload | studies cleared as normal, flagged and deferred per 1,000 studies |

A transferred threshold is judged to violate its target when its mean over re-splits exceeds the target,
because the guarantees are marginal and a single rate above the target does not contradict them.

## Repository structure

| Path | Content |
|---|---|
| `ovcbmr/` | Python package: splits, data loading, frozen encoders, concept heads and Platt scaling, valve statistic, conformal routines (`ood/conformal.py`), metrics |
| `scripts/` | one script per stage (`s00` to `s40`), dataset builders and figure scripts |
| `configs/` | configuration files for each dataset and encoder (`nih`, `vindr`, `iuxray`, `iuxray_nihtax`, and `*_clip`) |
| `tests/test_smoke.py` | checks of the split, the valve and the conformal routines; needs no data |
| `experiments/base_run/` | base pipeline run (all NIH images; Article S1) |
| `experiments/three_dataset/` | one image per patient at NIH; three-dataset transfer matrix |
| `experiments/resplits/` | 20 re-splits and 200 repartitions |
| `experiments/final/` | CLIP encoder, further checks, simulation, network head |
| `RUN_FROM_SCRATCH.md` | step-by-step instructions with downloads and expected counts |

## Archive contents

The archive at the DOI above is an exact copy of the code used for the article: the scripts are archived
as they were run, and the console logs were produced by this code. It contains no images and no
manuscript files.

## Result files

Result files are machine-readable JSON; console logs are plain text.

- Pairs are written `c->d`: the threshold is calibrated at dataset `c` and evaluated at dataset `d`
  (`nih`, `vindr`, `iuxray`).
- `s35_multisite*.json` has a `primary` block (per-site metrics, the nine pairs with rates and bootstrap
  intervals, the domain-classifier AUROC) and a `resplits` block (per-seed values and summaries).
- `s36_resplits.json` has the blocks `primary` (each pair under the symmetric rule, the uncorrected
  margin, the fixed confidence threshold and the one-sided rule at 0.10, 0.05 and 0.02), `redraws`
  (200 repartitions; keys such as `g1|c->d`, `acc|c->d` and `cer|c->d`), `resplits`, `sample_size`
  and `shift`.

## How to reproduce the results

A driver script (`scripts/run_all.py`) runs the base pipeline for NIH ChestX-ray14 and VinDr-CXR. The
stages added for the third dataset, the repeated experiments, the second encoder and the checks are
single commands, listed below in the order to run them. Commands are run from the repository root. They are written for Windows PowerShell; on Linux or macOS,
replace `^` with `\`.

1. **Smoke test** (no data, no GPU):

    ```
    python tests/test_smoke.py
    ```

2. **Base pipeline** (NIH ChestX-ray14 and VinDr-CXR: data preparation, embeddings, heads, and the
   stages reported in Article S1):

    ```
    python scripts/run_all.py --check
    python scripts/run_all.py
    ```

3. **Indiana University collection** (matched labels; omit `--nih-taxonomy-only` and change `--out` for
   the full set), then embeddings with stages `s01` and `s02` using `configs/iuxray_nihtax.yaml` and
   `configs/iuxray.yaml`:

    ```
    python scripts/build_iuxray.py --reports <dir>/indiana_reports.csv ^
        --projections <dir>/indiana_projections.csv ^
        --images <dir>/images/images_normalized --img-out <dir>/images_256 ^
        --nih-taxonomy-only --out data/raw/iuxray_nihtax/labels.csv
    ```

4. **Transfer matrix on the primary split** (repeat with `iuxray=configs/iuxray.yaml` for the full
   Indiana labels):

    ```
    python -m scripts.s35_multisite --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --seeds 0 1 2 3 4
    ```

5. **Re-splits, repartitions, comparators and workload; further checks; network head:**

    ```
    python -m scripts.s36_resplits --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --seeds 20 --redraws 200
    python -m scripts.s37_conditional_and_g1 --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --redraws 200
    python -m scripts.s39_nonlinear_g2 --config configs/vindr.yaml ^
        --sites vindr=configs/vindr.yaml iuxray=configs/iuxray_nihtax.yaml --seeds 20 --redraws 200
    ```

6. **Second encoder:** rerun stages `s01`, `s02`, `s09`, `s35` and `s36` with the `configs/*_clip.yaml`
   files.

7. **Checks of the conformal rules as implemented** (no data; about 70 seconds and 3 minutes on a CPU):

    ```
    python scripts/s38_validity_simulation.py --reps 4000
    python scripts/s40_exact_crc_check.py
    ```

8. **Figures:**

    ```
    python scripts/make_fig_multisite.py ^
        --json experiments/three_dataset/s35_multisite_nihtax.json ^
        --resplits experiments/resplits/s36_resplits.json ^
        --out fig_transfer.png --print --height-in 4.3
    python -m scripts.s13_figures --config configs/vindr.yaml
    ```

    - The first command draws Fig. 2 and the second draws Fig. S1.
    - Fig. 1 is drawn by `scripts/make_fig_method_2row.py` from four NIH ChestX-ray14 test images. These
      are not included in the archive and are exported from the NIH data with
      `scripts/export_fig_images.py`.

## Where each reported result comes from

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

## Calibration protocols

- **Clean protocol (primary).**
  - At NIH, the Platt scalars and the decision threshold tau are fit on the validation split, and the
    conformal thresholds on the calibration split.
  - At VinDr-CXR and Indiana, tau and lambda are fit on two disjoint halves of the calibration set, and
    the G1 threshold on the whole of it.
  - Thus tau is always fit on studies disjoint from those that fit lambda, as the G2 guarantee
    requires.
- **Coupled protocol (sensitivity only).** All of these quantities are fit on the same calibration
  split.
  - This does not meet the independence condition of the G2 guarantee.
  - It is the `paper` block of `s31_clean_protocol.json`, and the supplement marks the analyses that use
    it.

## Notes

- The scripts are archived exactly as run, so the console logs correspond to this code.
- The package name `ovcbmr` and some script names (`s01_prepare_cxrlt.py`, the `cxrlt` configuration)
  come from an earlier project plan and are kept because the pipeline calls them.
- Some console banners print reference values from an earlier data partition next to the computed values.
  The computed values are the ones reported.
- Stages `s03`, `s04`, `s06` and `s08` evaluate a zero-shot variant that the article does not report.
- If no margin satisfies the conformal risk control condition, lambda is set to infinity (every study is
  deferred). No reported run reached this case.

## Third-party software and references

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
