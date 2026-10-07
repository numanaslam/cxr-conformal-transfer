# Code and results for "Cross-Site Transfer of Conformal Abstention Guarantees in Interpretable Chest X-ray Concept Models"

Numan Aslam, Ghulam Mustafa, Adnan N. Qureshi, Zia Ul Rehman, Asghar Ali Shah

This archive holds the Python code that produces every number reported in the article, together
with the result files and the console logs of the run from which those numbers were taken. It
contains no images.

## What the study does

One frozen predictor is moved between hospitals, and two conformal abstention guarantees are
checked before and after the move. The predictor is a frozen BiomedCLIP image encoder with nine
linear, Platt-calibrated concept heads (eight findings and *Normal*), trained on NIH
ChestX-ray14. Two rules decide when it defers a study.

- **G1, false deferral.** A split-conformal threshold on the valve statistic
  `s(x) = A(x) * (1 - max_k a_k(x))`, where `A(x) = 1 - p(Normal)`. It bounds how often an
  abnormal study with an in-vocabulary finding is deferred (target 0.05).
- **G2, confident error.** A conformal risk control threshold on the margin `|A(x) - tau|`. A
  study is accepted when the margin is at least `lambda`. It bounds the fraction of studies that
  are both accepted and wrong (target 0.10). A one-sided variant bounds accepted missed
  abnormalities only.

Both rules are calibrated at NIH ChestX-ray14 and at VinDr-CXR, and each is evaluated on-site
and after transfer in both directions, on the primary split (seed 42) and on five re-splits.

## Contents

```
ovcbmr/                        importable package
  io/splits.py                 patient-level hash splits and the leakage check
  io/datasets.py               manifest reader and image loader
  models/encoder.py            frozen BiomedCLIP image and text encoders
  concept/                     concept bank, Platt scaling, linear heads, valve statistic
  ood/conformal.py             split-conformal threshold (G1), conformal risk control (G2),
                               weighted conformal
  ood/baselines.py             embedding-space novelty scores used as comparators
  eval/metrics.py              AUROC, ECE, risk-coverage, clustered bootstrap
scripts/                       one script per stage; run_all.py runs the stages in order
configs/nih.yaml, vindr.yaml   the two configurations used in the article
data/concepts/                 concept bank definition
tests/test_smoke.py            checks of the split, the valve and the conformal routines
experiments/rerun_20261002/    result files and logs of the reported run
RUN_FROM_SCRATCH.md            step-by-step reproduction guide
```

## Data

The images are not redistributed. Both datasets are public.

- **NIH ChestX-ray14**, 112,120 images, from the NIH Clinical Center:
  <https://nihcc.app.box.com/v/ChestXray-NIHCC>.
- **VinDr-CXR**, distributed through PhysioNet under a credentialed data use agreement:
  <https://physionet.org/content/vindr-cxr/1.0.0/>. The study used the 512-pixel copy of the
  15,000 labeled training studies that was redistributed on Kaggle from the VinBigData challenge
  release: <https://www.kaggle.com/datasets/awsaf49/vinbigdata-512-image-dataset>.

The BiomedCLIP weights are downloaded from the Hugging Face hub on first use
(`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`). The DenseNet-121 comparator uses
the `densenet121-res224-nih` weights of TorchXRayVision.

The splits are not stored as files. Each patient (NIH) or image (VinDr-CXR, which has no patient
identifier) is assigned by a hash of the seed and the identifier, so the same label file and
seed give the same partition. The head and Platt parameters are refitted by stage `s09`.

## Reproducing the results

```
python scripts/run_all.py --check     # data, environment and GPU checks
python scripts/run_all.py             # all 41 stages, about 40 minutes on one consumer GPU
```

`RUN_FROM_SCRATCH.md` gives the environment, the downloads, the expected split counts and the
tolerance to expect. The smoke test needs no data and no GPU:

```
python tests/test_smoke.py
```

## Where the reported results come from

Paths are relative to `experiments/rerun_20261002/`.

| Item in the article | Stage | File |
|---|---|---|
| Table 1, concept AUROC and ECE at both sites, with the DenseNet-121 column | `s19`, `s30` | `results/s19_vindr_concepts.json`, `results/s30_densenet_auroc.json` |
| Local Platt refit at the deployment site | `s26` | `results/s26_local_platt.json` |
| Tables 2, 4 and 6, the two guarantees and the one-sided rule on the primary split | `s31` | `results/s31_clean_protocol.json` |
| Table 3, valve ablation | `s24` | `results/s24_valve_ablation.json` |
| Table 5, acceptance decomposition, and Table 9, G2 sample size | `s23` | `logs/27_s23.log`, blocks (B) and (C) |
| Table 7 and Figure 2, primary split and five re-splits | `s31`, `s33` | `results/s31_clean_protocol.json`, `results/s33_multiseed.json` |
| Table 8, G1 sample size | `s18` | `results/s18_sample_efficiency.json` |
| Table 10, heads on the same frozen features | `s20`, `s32` | `results/s20_blackbox_baseline.json`, `results/s32_mlp_comparator.json` |
| Table 11, localization | `s15`, `s28`, `s29` | `results/s15_localization.json`, `results/s28_localization_bonferroni.json`, `results/s29_gradcam_baseline.json` |
| Table 12, weighted conformal | `s23`, `s27` | `logs/27_s23.log`, block (D), and `results/s27_weighted_sweep.json` |
| Novelty-detection comparison | `s05` | `results/s05_baselines.json` |
| In-vocabulary and out-of-vocabulary denominators for G1 | `s17` | `results/s17_known_abnormal.json` |
| View shift within NIH and group-conditional calibration | `s07`, `s11` | `results/s07_shift.json`, `results/s11_multigroup.json` |
| Appendix figures, coverage curves and score shift | `s13` | `results/fig2_coverage.png`, `results/fig_score_shift.png` |

Figure 2 is drawn from the `s31` and `s33` result files:

```
python scripts/make_fig_transfer.py --results experiments/rerun_20261002/results --out fig_transfer.png
```

`results/s33_seed42.json` is a consistency check, the re-split script run on the primary split.

## Two calibration protocols

`results/s31_clean_protocol.json` holds two blocks.

- `clean` is the protocol of the guarantee tables. The Platt scalars and the decision threshold
  `tau` are fitted on the NIH validation split, and the conformal thresholds on the NIH
  calibration split. At VinDr-CXR, `tau` and `lambda` are fitted on two disjoint halves of the
  calibration split, and the G1 threshold on the whole of it.
- `paper` is the coupled protocol, in which the Platt scalars, `tau` and both conformal
  thresholds are fitted on the same calibration split. The article reports it as a sensitivity
  column, and every stage other than `s31` and `s33` uses it. The article marks the affected
  tables as coupled.

## Notes on the code as archived

- The scripts are archived exactly as they were run, so the logs correspond to this code.
- The package name `ovcbmr` and a few script names (`s01_prepare_cxrlt.py`, the `cxrlt` concept
  file and configuration) come from an earlier plan built around a different dataset. They are
  kept because the pipeline calls them.
- Console messages keep their development annotations. Some banners print reference values next
  to the computed ones, for example `expect ~0.085` in `s23` and `paper 0.085` in `s27`. These
  come from an earlier partition of the data and are not reported in the article. The computed
  values are the ones reported.
- Stages `s03`, `s04`, `s06` and `s08` evaluate a zero-shot variant of the model that the
  article does not report. Stage `s05` supplies the novelty-detection comparison and uses the
  zero-shot concept scores of `s03`.

## License and citation

The code and the result files are released under the MIT License (see `LICENSE`). If you use
them, please cite the article and this archive. The DOI of the archive is given on its Zenodo
page.
