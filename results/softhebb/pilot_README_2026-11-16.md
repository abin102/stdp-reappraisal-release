# SoftHebb pilot (stage 1 run on 2026-11-11; stage 2 written 2026-11-16 before its runs, GPU 0)

**Stage 2 (five seeds, TissueMNIST, pixel readout) is described in the section "Stage 2" at the end, written before
any of its runs. Stage-1 copies of this file, the scripts and the summary are in `logs/*stage1*`.**

**Stage 2 done: all 76 runs finished 2026-11-16 13:33 on GPU 0, no failures; README and scripts unchanged during the run
(SHA-256 in `logs/pilot.log` match). Results: `pilot_summary.md`; values for the draft Table 5: `table5_values.py`/`.json`.**
- CIFAR-10, seeds 0-2: trained 80.22 ± 0.43 (paper 80.31 ± 0.14), random 73.76 ± 0.09, pixels 41.27 ± 0.23.
- 10,000 images, 5 seeds (stage-2 readings): OrganAMNIST mixed (+0.48, 3 of 5), BloodMNIST does not add (-1.08, 0 of 5),
  PathMNIST adds (+1.47, 5 of 5), TissueMNIST does not add (-2.30, 0 of 5). Stage-1 readings unchanged. Trained beats
  its input (pixels) on all four, 5 of 5 seeds each; so does random.
- Whole splits, 5 seeds: OrganAMNIST +0.64 (5 of 5), BloodMNIST -1.08 (0 of 5), PathMNIST -4.34 (0 of 5).

**Stage 1 done: all 38 runs finished 2026-11-11 21:15. Results: `logs/pilot_summary_stage1.md`.**
- Reproduction check met: CIFAR-10 79.72% (paper 80.3%, bar 78.3%); untrained weights 73.86%, so training adds 5.86 points there.
- Medical sets at 10,000 training images (the rule's readings): OrganAMNIST mixed (mean gain +0.46), BloodMNIST does not
  add (-0.61, negative in 3 of 3 seeds), PathMNIST adds (+1.40, positive in 3 of 3).
- Whole training splits (reported, no reading): OrganAMNIST +0.72, BloodMNIST -0.82, PathMNIST -4.27 (negative in 3 of 3;
  the sign differs from the 10,000-image result).
- The untrained network is already strong (about 90% on OrganAMNIST, 92.5% on BloodMNIST).

Question: on medical images, does SoftHebb training add accuracy over the same network with untrained weights? It is
the first step of the learning-rule benchmark and the "compare with the state of the art" addition to the STDP paper.

- Code: `run_softhebb.py` imports the released single-file demo of Journe et al. (ICLR 2023) unchanged from
  `third_party/SoftHebb/` (github.com/NeuromorphicComputing/SoftHebb, commit f425230; the repository has no licence
  file, so it is used here and not redistributed). One part is written out in our script instead of imported: the
  demo's learning-rate scheduler does not run under the installed PyTorch (2.7); `set_lr()` computes the same rule.
- Control ("random"): the same pass over the training images with the weight updates switched off; BatchNorm
  statistics are still estimated. Same readout, same images, same seeds.
- Data: OrganAMNIST, BloodMNIST, PathMNIST at 28 px, official train and test splits, zero-padded to 32 px;
  CIFAR-10 as the reproduction check.
- Rule, fixed before any run: in `score_softhebb.py` (reproduction bar 78.3% on CIFAR-10; per dataset at 10,000
  training images, "adds" if trained beats random in 3 of 3 seeds, "does not add" if in 0 of 3, otherwise "mixed").
- Run: `nohup setsid bash run_pilot.sh > /dev/null 2>&1 &` (GPU 1 only; 38 runs). Summary: `pilot_summary.md`.

Limits, stated before the runs: the network and learning rates are CIFAR-10's, not tuned for medical images; three
seeds; the network has 24,576 readout features against 400 neurons in the STDP paper's matrix, and the readout is
the demo's trained linear classifier, not that paper's logistic-regression probe, so the numbers sit beside the
paper's tables, not inside them; a raw-pixel readout is not run here.

Checked on 2026-11-11 (CPU, tiny settings): both modes run end to end; in "random" no weight moves and in "trained"
every layer moves (the script stops otherwise). Not checked: any accuracy, and the time per run on the GPU.

## Stage 2 (written 2026-11-16, before any of its runs)

Purpose: five seeds for the readings, a fourth medical set, and a comparison with the network's input, so that the
SoftHebb section can be read like the STDP paper's tables (trained against random, and network against input).

Correctness checks done before stage 2 (2026-11-16):
- The released demo is the network the paper reports on CIFAR-10. The paper's Table 1 gives 80.3; its Section 4
  gives 80.31 ± 0.14 over 4 runs (Appendix A.6), for three SoftHebb convolution layers (96, 384, 1536) and a linear
  classifier (Table 4). The settings match the paper's Appendix A.6: one unsupervised epoch at batch 10, readout
  batch 64 for 50 epochs, Adam at 0.001, halved at 20, 35, 50, 60, 70, 80 and 90% of training. The repository's
  preset for this model (`presets.json`, `4SoftHebbCnnCIFAR`) has the same channels, kernels, temperatures, learning
  rates, activations, pooling and dropout as the demo.
- `set_lr()` gives the same per-neuron learning rates as the demo's `WeightNormDependentLR.get_lr()`: the largest
  difference over 6 real update steps was 0.0 (CPU check). It is called once before the loop, as the demo's
  scheduler does when constructed, and after every update.
- Stage 1 gave 79.72% on CIFAR-10, seed 0. That is 0.59 points under the paper's mean, inside the 2-point bar but
  outside the paper's ± 0.14. Seeds 1 and 2 below show whether this is seed spread. Possible causes, not tested: the
  PyTorch version (2.7 here, 1.7.1 in the paper) and the paper's own training script being used there instead of the
  demo.
- The weight scale does not separate the two networks much. After one pass, the trained third layer's mean neuron
  norm is 17 to 26 against 25 at initialization. The first two layers are followed by BatchNorm, which removes a
  common scale.

Runs added (queue `run_pilot.sh`, one after another on one GPU; a finished run is skipped):
1. At 10,000 training images: seeds 3 and 4 for OrganAMNIST, BloodMNIST and PathMNIST; TissueMNIST (8 classes,
   28 px greyscale, padded and repeated as the others) with seeds 0 to 4, trained and random.
2. CIFAR-10, whole training split: seeds 1 and 2, trained and random.
3. Whole training splits: seeds 3 and 4 for OrganAMNIST, BloodMNIST and PathMNIST. TissueMNIST's whole split
   (165,466 images, about 4 GPU-hours for 10 runs) is not run.
4. A pixel readout ("pixels") for every dataset, training size and seed above. It uses the same readout procedure
   as the network (dropout 0.5, one linear layer, Adam at 0.001, the demo's step schedule, batch 64, 50 epochs,
   score at the last epoch). Its input is the 3 x 32 x 32 image, standardized per channel with the mean and
   standard deviation of the training images used. The linear layer has PyTorch's default initialization and
   3,072 inputs instead of 24,576.

Readings, fixed now (`score_softhebb.py` implements them), per medical set at 10,000 training images, seeds 0 to 4:
- Training (trained against random, paired by seed): "adds" if trained is higher in at least 4 of 5 seeds,
  "does not add" if in at most 1 of 5, otherwise "mixed".
- Input (trained against pixels, paired by seed): "beats its input" if trained is higher in at least 4 of 5 seeds,
  "does not beat its input" if in at most 1 of 5, otherwise "mixed".
- A reading needs all runs it uses. The stage-1 readings (seeds 0 to 2) are kept as registered and shown beside the
  stage-2 ones; where they differ, both are reported.
Reported without a reading: CIFAR-10 seeds 0 to 2 (mean and spread, beside the paper's 80.31 ± 0.14), random against
pixels, the whole splits, means and standard deviations over seeds.

Limits, stated before the runs: the network and learning rates are still CIFAR-10's; five seeds; the pixel readout is
the same procedure as the network's readout, not the STDP paper's logistic-regression probe.

Run: `DEVICE=cuda:0 nohup setsid bash run_pilot.sh > /dev/null 2>&1 &` (about 3 GPU-hours). Summary:
`pilot_summary.md`.
