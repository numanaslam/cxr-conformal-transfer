# Codebook

Also provided with the article as Supplemental Article S3.

# Purpose

This codebook gives the meaning of every categorical variable that the code or the result files record
as a number, so that each number can be converted to its category. Variables that are true quantities
(rates, AUROC, counts, thresholds) are listed in the README (Supplemental Article S2, Section 7) and are
not repeated here except where a key name is abbreviated.

# 1. Per-study label files (labels.csv)

The builder scripts `build_manifest_nih.py`, `build_manifest_vindr.py` and `build_iuxray.py` write one row
per image (NIH) or study (VinDr-CXR, Indiana). These files are derived from the third-party datasets and
are not redistributed; they are rebuilt by the scripts.

| Column | Type | Codes |
|---|---|---|
| `image_id` | text | file name of the image |
| `patient_id` | text | patient identifier (NIH); study identifier at VinDr-CXR; `iu` followed by the report identifier at Indiana |
| `projection` | text | `PA` = posteroanterior, `AP` = anteroposterior (NIH `View Position`); `PA` is written for every Indiana frontal image |
| `split` | text | `train`, `calib`, `val`, `test`; empty in the file when the split is assigned later by the seeded hash |
| `site` | text | `vindr` = VinDr-CXR, `iuxray` = Indiana University collection |
| `Atelectasis`, `Cardiomegaly`, `Consolidation`, `Pleural Effusion`, `Infiltration`, `Mass`, `Nodule`, `Pleural Thickening` | integer | `1` = finding present, `0` = finding absent |
| `Normal` | integer | `1` = normal study (NIH `No Finding`; VinDr-CXR `No finding`; Indiana sole MeSH heading "normal"), `0` = not normal |
| other finding columns at NIH (for example `Pneumonia`, `Hernia`) and VinDr-CXR | integer | `1` = present, `0` = absent; held out from the vocabulary and not used in the article |

Derived study categories used throughout the code and the result files:

| Category | Rule | Key in result files |
|---|---|---|
| Normal | `Normal` = 1 | `normal` |
| In-vocabulary abnormal | `Normal` = 0 and at least one of the eight findings = 1 | `invocab`, `in-vocab-abnormal` |
| Out-of-vocabulary abnormal | `Normal` = 0 and all eight findings = 0 | `oov` |
| Any abnormal | `Normal` = 0 | `any`, `any-abnormal` |

When a study is given a binary reference label for G2, `1` = abnormal (`Normal` = 0) and `0` = normal.

# 2. Codes used inside the analysis

| Variable | Codes |
|---|---|
| G2 decision for a study | accepted as normal when A(x) <= tau - lambda; accepted as abnormal (flagged) when A(x) >= tau + lambda; deferred otherwise. Counts are stored as `cleared`, `flagged` and `deferred` (per 1,000 studies under `workload`) |
| G1 decision for a study | deferred when s(x) > q-hat, otherwise not deferred |
| Domain-classifier label | `0` = study from the first site of the pair, `1` = study from the second site |
| Seed (`seed`, `seeds`) | identifier of a random split, not a quantity: `42` = primary split; `0` to `19` = re-splits; `0` to `4` = the five re-splits of the full-label analysis |
| Repartition index | `0` to `199`, identifier of one redraw of calibration and test studies with the predictor fixed |
| Bootstrap index | `0` to `999`, identifier of one resample |

# 3. Result files (JSON)

| Key or value | Codes |
|---|---|
| Site names | `nih` = NIH ChestX-ray14, `vindr` = VinDr-CXR, `iuxray` = Indiana University collection (the `_nihtax` files use the matched Indiana labels) |
| `c->d` | threshold calibrated at site `c` and applied at site `d`; `c->c` = local calibration |
| `c|d` | pair of sites `c` and `d` (domain classifier, shift measures) |
| `NIH on-site`, `NIH->VinDr` and similar | same as `c->d`, written in full in the earlier files |
| `true` / `false` | JSON booleans; `bound_met` = rate at or below its target; `naive_violates_alpha` = uncorrected rule above its target |
| `alpha`, `alpha_g1`, `alpha_g2`, `alpha_risk`, `alpha_flag` | target levels (`0.05` for G1, `0.10` for G2) |
| `onesided_0.1`, `onesided_0.05`, `onesided_0.02`; `acc1s_*`, `miss1s_*` | one-sided rule at target 0.10, 0.05 or 0.02; acceptance and missed-abnormal rate under it |
| `g1` | false-deferral rate |
| `g2`, `cer`, `joint_error` | confident-error rate (joint) |
| `acc`, `accept`, `coverage` | acceptance |
| `err_given_accept`, `error_given_accept` | error among accepted studies |
| `miss`, `miss_given_accept`, `joint_miss` | missed abnormal studies among accepted (conditional) and jointly |
| `accnaive`, `cernaive`, `naive_margin` | rule with the uncorrected empirical margin |
| `fixed_confidence` | fixed confidence threshold (accept when A >= 0.8 or A <= 0.2) |
| `symmetric`, `one-sided`, `fn` | G2 loss: any error, or only missed abnormal studies among accepted predicted-normal studies |
| `clean`, `paper` | clean protocol (primary) and coupled protocol (sensitivity) |
| `q_hat`, `qhat` | G1 threshold; `tau` decision threshold; `lam`, `lambda` G2 margin |
| `ffr` | share of in-vocabulary abnormal studies that the G1 rule defers (the G1 false-deferral rate, under its earlier name) |
| `lo95`, `p2.5`, `p97.5`, `ci` | lower 95% bound, 2.5th and 97.5th percentiles, 95% interval |
| `frac_above` | share of re-splits or repartitions above the target |
| `m`, `n`, `n_*` | number of calibration studies (`m`) or of studies, patients or images (`n_*`) |
| `view` group | `AP`, `PA` |
| `sex` group | `F` = female, `M` = male |
| `age_band` group | `<40`, `40-59`, `60-79`, `80+` years |
| `grid` | side of the square grid used in the localization analysis (`14` = 14 by 14 cells) |
| `C` | inverse regularization strength of the logistic regression (a quantity, not a category) |
