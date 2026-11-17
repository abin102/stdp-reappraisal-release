# SoftHebb pilot (prepared and run on 2026-11-11, GPU 0)

**Done: all 38 runs finished 2026-11-11 21:15. Results: `pilot_summary.md`.**
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
