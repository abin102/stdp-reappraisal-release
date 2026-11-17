# SoftHebb with the matched random baseline (Table 5)

These are the results behind Section 3.9 and Table 5 of the paper, with the procedure in
Section 2.13 (Methods) and Section S18 of the Supplementary material. SoftHebb (Journé et al.,
ICLR 2023) trains a three-layer convolutional network without labels. Each network is compared
with the same network with the weight updates disabled (random) and with a linear classifier of
the same form fitted to the pixels (pixels). 114 runs:

| dataset | training images | seeds | runs |
|---|---|---|---|
| CIFAR-10 | whole split (50,000) | 0-2 | 9 |
| OrganAMNIST, BloodMNIST, PathMNIST, TissueMNIST | 10,000 | 0-4 | 60 |
| OrganAMNIST, BloodMNIST, PathMNIST | whole split | 0-4 | 45 |

each as trained, random and pixels.

## Files

| file | contents |
|---|---|
| `runs/<dataset>__<mode>__n<N>__seed<S>/result.json` | one run: test accuracy at the last readout epoch (`test_acc_last`), the readout curve, how far the weights moved, timings. `n0` = whole training split. The test logits (`test_logits.npz`) are not shipped. |
| `table5_values.json` | the values of Table 5 (`scripts/score/softhebb_table5.py`) |
| `pilot_summary.{md,json}` | per-seed values and the pre-registered readings (`scripts/score/softhebb_summary.py`) |
| `pilot_README_2026-11-16.md` | the experiment's own notes as they were at the end of the runs: the rules fixed before each stage, the checks made, and the limits stated in advance. Its file and script names are those of the original folder (see below). |
| `provenance/` | records of the original folder: the launcher's log of all runs (`pilot_log.txt`, with the SHA-256 of the notes, scripts and SoftHebb's `demo.py` at the start of each stage), and the stage-1 notes and summary (`README_stage1_2026-11-11.md`, `pilot_summary_stage1.{md,json}`), which the stage-2 notes call `logs/*stage1*`. The stage-1 scripts are not shipped. |

## Reproduce Table 5 from the shipped runs

```bash
PYTHONPATH=. python scripts/score/softhebb_table5.py    # prints the rows of Table 5, writes table5_values.json
PYTHONPATH=. python scripts/score/softhebb_summary.py   # writes pilot_summary.{md,json}
```

Both read `results/softhebb/runs/` and need only numpy. `scripts/checks/check_results.py --only
softhebb_table5 softhebb_summary` reruns them in a scratch folder and compares the output with the
shipped files.

## Rerun the networks

**SoftHebb's code is not included.** The runs import the single-file demo released by the SoftHebb
authors (`demo.py`) unchanged. Their repository has no licence file, so it is not redistributed here.
Get it from the authors and check out the commit the runs used:

```bash
git clone https://github.com/NeuromorphicComputing/SoftHebb.git third_party/SoftHebb
git -C third_party/SoftHebb checkout f4252307633ba38a26ab52504f5fd40352f5a6fd   # 2023-10-27, "Update README.md"
```

`third_party/` (at the repository root) is ignored by git. To keep the clone elsewhere, set
`SOFTHEBB_DIR` to its folder. `scripts/train/softhebb_run.py` stops with these instructions when
`demo.py` is missing, and also when its SHA-256 differs from that of the file the runs used
(`0e44b4e6e830670f2bcd57b8fe845e29472c8c77d108b3db3a2dcb12a5d3630b`).

Data: the four MedMNIST sets are read from `data/<name>.npz` (28 px) and downloaded through
`medmnist` when absent, as for the rest of the paper. CIFAR-10 is read from
`data/cifar-10-batches-py/`:

```bash
python -c "import torchvision; torchvision.datasets.CIFAR10('data', download=True)"
```

Then, on one GPU (several GPU-hours for all 114 runs):

```bash
# one run
PYTHONPATH=. python scripts/train/softhebb_run.py --dataset bloodmnist --mode trained --n_train 10000 --seed 0 --device cuda:0
# all 114 runs, one after another, then the summary
DEVICE=cuda:0 bash scripts/train/softhebb_runs.sh
# Table 5 from the new runs
PYTHONPATH=. python scripts/score/softhebb_table5.py --runs experiments/runs/softhebb --out experiments/runs/softhebb
```

New runs go to `experiments/runs/softhebb/` (not tracked), so the shipped results are not
overwritten; a run that already has a `result.json` is skipped. The paper's runs used PyTorch 2.7
on a CUDA GPU; a rerun on other hardware or library versions will not match the shipped values to
the last digit.

## Release copies and their originals

The experiment was run in a folder `experiments/softhebb_pilot/` of the authors' working
repository. The release keeps its scripts with these names:

| original | release |
|---|---|
| `run_softhebb.py` | `scripts/train/softhebb_run.py` |
| `run_pilot.sh` | `scripts/train/softhebb_runs.sh` |
| `score_softhebb.py` | `scripts/score/softhebb_summary.py` |
| `table5_values.py` | `scripts/score/softhebb_table5.py` |
| `runs/`, `pilot_summary.*`, `table5_values.json`, `README.md` | `results/softhebb/runs/`, `pilot_summary.*`, `table5_values.json`, `pilot_README_2026-11-16.md` |
| `logs/pilot.log`, `logs/*stage1*` (notes and summary) | `results/softhebb/provenance/` (`pilot.log` renamed `pilot_log.txt`) |
| `third_party/SoftHebb/` | not shipped (see above) |

Only paths and messages changed: the folders the scripts read and write, the location of the
SoftHebb clone (with a check of `demo.py`), a message when a dataset is missing, the default device
(`cuda:0`), and in the launcher the removal of a wait for other users' jobs on a shared GPU. The
computation is unchanged; the two scorers reproduce the shipped `table5_values.json` and
`pilot_summary.{md,json}` byte for byte.

The runs were made in two stages (2026-11-11 and 2026-11-16), each with its rules written before
its runs (`pilot_README_2026-11-16.md`). Stage 2 reused the stage-1 runs (seeds 0-2) and added
seeds, TissueMNIST and the pixel readout. The stage-1 script differed from the final one only by
lacking the pixel mode and TissueMNIST; the trained and random runs are computed by the same code.
