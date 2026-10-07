# Run the whole paper from scratch

This rebuilds every number in the paper on a clean GPU machine: download the two public
datasets, compute the embeddings, train the small heads, and run every experiment.

All commands run from the repo root, the folder that contains `ovcbmr\`, `scripts\` and
`configs\`. They are written for Windows PowerShell.

## 1. What you need

| Item | Detail |
|---|---|
| GPU | NVIDIA, 8 GB or more. Tested on an RTX 5070 (12 GB). |
| Disk | About 100 GB free while NIH downloads (45 GB archives + 45 GB extracted). The archives can be deleted afterwards. |
| Accounts | A free Kaggle account, for VinDr-CXR and the NIH label file. |
| Software | Miniforge or Anaconda. |

## 2. Environment

If an `ovcbmr` environment already exists, reuse it and add the three packages that the later
stages need:

```powershell
conda activate ovcbmr
pip install torchxrayvision scikit-image kaggle
```

To build it new:

```powershell
conda create -y -n ovcbmr python=3.10
conda activate ovcbmr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install torchxrayvision scikit-image kaggle
```

The `cu128` line matters on RTX 50-series cards. `scripts\setup_env.ps1` installs torch 2.4.1,
which is older than those cards, so do not use it with them. Step 4 tests the GPU and tells
you if the build is wrong.

## 3. Download the data

Three downloads. None of them needs the GPU.

**NIH images (45 GB, resumable).** Twelve archives from the official NIH links:

```powershell
python scripts\pdownload.py --list scripts\nih_urls.txt --out data\raw\nih --workers 4 --extract
```

If a file fails validation, NIH has rotated that link. Open
<https://nihcc.app.box.com/v/ChestXray-NIHCC>, copy the new link into `scripts\nih_urls.txt`,
and run the same command again; finished files are skipped.

**Kaggle login.** On kaggle.com open Settings, then API, then "Create New Token". Save the
`kaggle.json` it gives you to `%USERPROFILE%\.kaggle\kaggle.json`. That file is a personal
credential, so keep it private.

**NIH label file and VinDr-CXR (512-pixel release, a few GB):**

```powershell
kaggle datasets download -d nih-chest-xrays/data -f Data_Entry_2017.csv -p data\raw\nih
kaggle datasets download -d awsaf49/vinbigdata-512-image-dataset -p data\raw\vindr --unzip
```

If the first command leaves `Data_Entry_2017.csv.zip`, unzip it in place. The file from the NIH
Box page, `Data_Entry_2017_v2020.csv`, also works; put it in `data\raw\nih\`.

When the downloads finish the folders must look like this:

```
data\raw\nih\Data_Entry_2017.csv
data\raw\nih\images\00000001_000.png ...          112,120 files
data\raw\vindr\vinbigdata\train.csv
data\raw\vindr\vinbigdata\train\<id>.png ...       15,000 files
```

## 4. Check before running

```powershell
python scripts\run_all.py --check
```

This counts the images, reads the file headers, and runs a small test on the GPU. Every line
should say `ok`. A line marked `FAIL` says what is missing. A `warn` about image width and
height means the localization stages need a size file; pass it with
`--dims <csv with image_id, dim0, dim1>`.

## 5. Run

```powershell
python scripts\run_all.py
```

This runs 41 stages in order and stops at the first one that fails. `python scripts\run_all.py
--list` shows them. What to expect:

- **Total time** is about 40 minutes of computing on an RTX 5070, after the downloads.
- **The two long stages** are the NIH embeddings (stage 4, about 7 minutes) and the localization
  (`s15`, about 17 minutes). Stage 4 prints nothing while it works. Watch `nvidia-smi` in a
  second window; GPU use should be high. It finishes with one line per split.
- **The GPU must be free.** The embedding stages use a batch of 512 images. If another training
  job is running on the card, wait for it, or lower `compute.batch_size` in `configs\nih.yaml`
  and `configs\vindr.yaml` to 128.
- **If Windows hangs on data loading**, set `data.num_workers: 0` in both config files.
- **Each stage's output** is saved in `experiments\logs\`, numbered in run order.

If a stage fails, the last lines say which one and how to resume:

```powershell
python scripts\run_all.py --from s16
```

Other useful forms:

```powershell
python scripts\run_all.py --no-gpu              # only the stages that use cached embeddings
python scripts\run_all.py --only s31 s33        # just these stages
python scripts\run_all.py --skip s29 s30        # everything except the torchxrayvision stages
python scripts\run_all.py --groups review       # one group: nih vindr cross review gpu figs
```

## 6. Output

The run ends by writing one file:

```
experiments\paper6_results_<date>_<time>.zip
```

It holds every result JSON, the figures, and all the logs.

## What the stages are

| Group | Stages | Needs GPU |
|---|---|---|
| `nih` | Build the NIH manifest, split by patient, compute embeddings, train the nine linear heads, run the source-hospital experiments | embeddings only |
| `vindr` | Build the VinDr manifest, split, compute embeddings | embeddings only |
| `cross` | The cross-hospital experiments reported in the paper (s12 to s23) | no |
| `review` | Analyses added during review: valve ablation, missed-finding control, local recalibration, weighted-conformal sweep, clean calibration protocol, nonlinear comparator, multiple seeds (s24 to s33) | no |
| `gpu` | Saliency and localization, and the trained DenseNet comparisons (s14, s15, s29, s30) | yes |
| `figs` | Corrected localization table and the paper figures | no |

## Will the numbers match the paper?

The numbers in the paper are the output of this pipeline, run from scratch on 2026-10-02. The
result files and logs of that run are kept in `experiments/rerun_20261002/`.

Check the split first. Stage `nih_split` must print these counts:

```
train 78607 images / 21622 patients     calib 10711 / 2997
val   10939 images /  3067 patients     test  11863 / 3119
```

The split is a hash of the seed and the patient identifier, so it depends on the identifier
strings in the label file. A manifest built from a different file, or with the identifiers
written differently, gives a different partition under the same seed. That happened once in this
project, and the single-split results moved with it. If the counts differ, the per-split numbers
will not match the paper, although the conclusions should, as stage `s33` shows.

With the same split, embeddings can still differ in the last decimal places between GPU models
and library versions, so expect agreement to about two or three decimals, not bit-for-bit.

To confirm that the re-split script and the main pipeline agree on the primary split:

```powershell
python scripts\s33_multiseed.py --config configs\vindr.yaml --seeds 42 --out s33_seed42.json
```

Its NIH-calibrated rows should match the `clean` block of `s31_clean_protocol.json`
(G1 0.046 / 0.055, G2 acceptance 0.50 / 0.25, confident-error 0.116 / 0.031). The
VinDr-calibrated G2 rows can differ slightly, because the two scripts halve the VinDr
calibration set with different hash salts.
