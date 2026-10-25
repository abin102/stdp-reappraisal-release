# Positive control — the voltage-STDP implementation reproduces Clopath 2010

> **NOTE (2026-10-11): The label 'Clopath 14:8 (LTP-dominant)' is swapped relative to Clopath's published amplitudes (A_LTD 14e-5 > A_LTP 8e-5); the measured numbers are valid. The plasticity code is correct, but the network's neuron is not Clopath's (an AdEx neuron with Diehl & Cook membrane constants), so 'the failure is not an implementation error' overstates. Current state: `HANDOFF.md`.**


**Why this exists.** Mandatory per [`paper_drafts/PUBLICATION_SCOPE.md`](../paper_drafts/PUBLICATION_SCOPE.md)
(action #2) and [`CLAIMS_FOR_REVIEW.md`](../paper_drafts/CLAIMS_FOR_REVIEW.md) (line 171):
before we can claim "voltage-Clopath STDP fails to learn in the competitive FC
regime," we must show the failure is **not a buggy implementation**. A reviewer
will point to VDSP (Garg 2022) — voltage plasticity *works* for classification —
and ask "why does yours fail?" This control answers: the plasticity code is
faithful; the failure is a **config/regime** effect.

**Method.** We drive the **exact** plasticity step the failing FC runs use —
`models.fc_stdp_layer._make_adex_voltage_step` (LTP = `A₊·x̄_pre·[v−θ₊]₊·[ū₊−θ₋]₊`,
LTD = `A₋·x_pre·[ū₋−θ₋]₊`) — with classic spike-pairing protocols. A strong
"teacher" synapse evokes the postsynaptic spike (mimicking the experimental
depolarising pulse); Δw is measured on a separate weak "plastic" synapse. The
voltage thresholds are the repo's actual values **θ₊=−45.3, θ₋=−70.6 mV — exactly
Clopath 2010's published thresholds** (τ₊=7, τ₋=10 also canonical).
Script: [`scripts/checks/clopath_positive_control.py`](../scripts/checks/clopath_positive_control.py).
Figure: [`clopath_positive_control.png`](clopath_positive_control.png).

## Result — the canonical STDP window is reproduced ✓

STDP timing window, Δw per pre/post pairing (15 pairs, fixed config):

| condition | Δw(pre→post, LTP) | Δw(post→pre, LTD) | window? |
|---|--:|--:|---|
| **Clopath 2010** (v_rest=θ₋=−70.6, A₊:A₋=14:8) | **+1.04e−5** | **−2.38e−5** | ✓ Hebbian (LTP for pre→post, LTD for post→pre) |
| B3 amplitudes, Clopath v_rest (A 1:3) | +3.9e−6 | −6.0e−5 | ✓ but LTP suppressed |
| **B3 task config** (v_rest=−65, A 1:3) | −6.2e−4 | −1.4e−3 | ✗ pure depression, no LTP lobe |

The teacher evoked exactly 1 post-spike per pairing in every run (verified). At
Clopath's parameters the rule produces the textbook asymmetric window — **the
equations are correctly implemented.** (Qualitative sign/asymmetry reproduction,
not a quantitative fit; the window is mildly LTD-leaning at the low 5 Hz pairing
rate, itself consistent with Clopath/Sjöström low-frequency depression.)

## Full curve reproduction — standalone equations + production overlay (2026-06-15)

Beyond driving the production function (above), we also **implemented Clopath's
equations independently from scratch** and checked all three of his canonical
curves, then overlaid the production code on the same protocols.
Script: [`scripts/figures/clopath_curve_reproduction.py`](../scripts/figures/clopath_curve_reproduction.py);
figure: [`clopath_curve_reproduction.png`](clopath_curve_reproduction.png).

Standalone rule (independent transcription of the paper, *not* calling repo code):
```
LTD (on each pre-spike):  Δw −= A_LTD · [ū_d − θ₋]₊
LTP (every timestep):     Δw += A_LTP · x̄_pre · [u − θ₊]₊ · [ū_p − θ₋]₊
```
θ₋=−70.6, θ₊=−45.3 (Clopath's published thresholds); ū_d, ū_p = low-pass post
voltages, x̄_pre = pre-spike trace.

| Clopath curve | result | reproduced? |
|---|---|---|
| **(A) Voltage dependence** (Fig 1) | Δw=0 below θ₋; **LTD** between θ₋,θ₊; sharp flip to **LTP** above θ₊=−45.3 | ✅ textbook (the defining voltage-gating signature) |
| **(B) STDP timing window** (Fig 2a) | pre→post LTP, post→pre LTD — standalone **and** production agree | ✅ |
| **(C) Frequency dependence** (Fig 2b) | potentiation grows with pairing rate (1→50 Hz); standalone **and** production agree | ✅ trend only — the low-freq LTD sign-flip is parameter-sensitive and **not** claimed |

The independent standalone and the production `_make_adex_voltage_step` produce the
**same** window and frequency curves ⇒ "the equations match the paper" **and** "the
production code matches the equations."

**Honest scope of this claim (state it):** this proves the **plasticity rule is
faithfully implemented** — it does NOT claim (i) bit-exact quantitative fidelity
(we reproduce shapes/signs/crossover, not Clopath's exact Δw magnitudes), (ii) the
full Fig-2b crossover (only the potentiation-with-frequency trend; the low-freq LTD
lobe needs parameter fitting), or (iii) validation of the full network pipeline
(only the plasticity step — encoding/AdEx-under-drive/inhibition/weight-management
are covered by the shared-infrastructure argument below, not by this). A numerical
diff against the Brian2/NEST `clopath_synapse` reference would upgrade "faithful
rule" to "matches the published reference implementation" — not yet done.

## How we know it's not a bug (reviewer-facing argument)

"We ran a control and it looked right" is not a proof. The argument that this
refutes a **buggy implementation** has a specific structure:

**1. It exercises the *same* code, not a reimplementation.** The control imports and
drives `models.fc_stdp_layer._make_adex_voltage_step` — the exact function the
failing FC runs call. A clean re-implementation that reproduced Clopath would prove
nothing about the FC code; by feeding controlled spike-pairs into the *production*
function, any bug in the plasticity equations would equally corrupt this output.
Control and failing runs share one code path — they stand or fall together.

**2. It reproduces a falsifiable, published signature at the rule's own published
parameters.** Clopath 2010's defining directional prediction is the STDP window
(pre→post LTP, post→pre LTD). At his published θ₊=−45.3/θ₋=−70.6 (already the repo's
values — not tuned to fake the result) the function gives the correct sign on both
arms with the correct asymmetry (+1.0e−5 / −2.4e−5).

**3. The result has discriminating power — a bug would have shown.** Reproducing the
window is informative because genuine bugs fail visibly:

| bug class | what the window would have shown |
|---|---|
| wrong sign on the LTP or LTD term | inverted window, or all-one-sign |
| swapped / mis-filtered ū₊, ū₋ voltage traces | wrong shape/width, no clean asymmetry |
| broken voltage gate `[v−θ]₊` (wrong threshold / no rectification) | no voltage dependence, no crossover at θ₊ (panel C) |
| broken presynaptic trace `x̄_pre` | flat window — no timing dependence at all |
| broken Clopath normalisation | saturated / garbage magnitudes |

We observed the correct directional, timing-dependent window ⇒ all of these are
ruled out. A reviewer cannot claim "you swapped a sign / your traces are wrong" —
those bugs are inconsistent with the curve obtained.

**4. Two corroborating arguments (the control is not the only evidence).**
(a) *Shared infrastructure is exonerated by the other rules.* Voltage shares the
neuron model, encoding, scaffold and readout with pair/triplet-STDP, which learn
(85–87% MNIST); a bug in the shared machinery would break all three — it doesn't,
so only the voltage-specific plasticity (the part this control validates) is at
issue. (b) *The same voltage code reaches 96.6% in 2-layer conv* — a fundamentally
broken implementation could not learn in any regime; it can, so the FC failure is
regime-specific, not a dead rule.

**Honest scope of the claim.** This proves the **plasticity equations are faithfully
coded** — it does NOT claim bit-exact quantitative fidelity (it's a qualitative
sign/asymmetry reproduction), nor does it exercise the full network path (only the
plasticity step; the rest is covered by argument 4a). We reproduced two canonical
signatures (timing window + voltage gating, panel C); the Sjöström
frequency-dependence was parameter-sensitive and is **not** claimed. The logic is
therefore two-step: this control ⇒ *the code is correct, so the failure is in the
config/regime*; the Clopath-regime follow-up (below) ⇒ *the failure persists even at
Clopath's own config, so it is intrinsic, not a bad config either.* Both reviewer
escape routes — "buggy code" and "broken config" — are closed only by the pair.

## Bonus — *why* the FC task config fails (two departures from Clopath)

The same code collapses to **pure depression** under the B3 task config because
that config differs from Clopath in two LTP-suppressing ways:

1. **Inverted amplitude ratio.** Clopath is LTP-dominant (A_LTP:A_LTD ≈ 14:8 ≈
   1.75); B3 is **LTD-dominant 1:3** (= 0.33). B3's A₊ was cut to 0.001 precisely
   to stop LTP runaway under L1 weight renormalisation (CLAUDE.md tuning note) —
   so the **weight-management scheme forces an anti-Hebbian amplitude balance.**
2. **Tonic LTD at rest.** B3 uses `v_rest=−65` > `θ₋=−70.6`, so the LTD gate
   `[ū₋−θ₋]₊` is **already open at rest** (normalised value **0.30**) — *every*
   presynaptic spike depresses, regardless of postsynaptic activity. Clopath sets
   `v_rest = θ₋ = −70.6` exactly, giving **0.00** LTD at rest (panel C). This
   tonic depression is a direct contributor to the decay-to-silence failure seen
   in [`voltage_native_weightmgmt.md`](voltage_native_weightmgmt.md).

## Paper implications

- **Closes the "buggy implementation" objection (C5/C8).** The plasticity
  reproduces Clopath's STDP window at Clopath's parameters. Cite this against the
  VDSP "voltage works" prior.
- **Sharpens the failure claim.** Voltage doesn't fail because the rule is broken;
  it fails because the **competitive-FC weight-management constraints (L1 renorm →
  LTD-dominant amplitudes) + v_rest>θ₋ push it into a depression-dominant regime**
  that cannot do Hebbian potentiation. This ties the failure to the
  weight-management coupling — consistent with "fails under both weight schemes."
- **Definitive follow-up — DONE 2026-06-15 → INTRINSIC**
  ([`voltage_clopath_regime.md`](voltage_clopath_regime.md)). Ran FC voltage at
  **Clopath's native regime** (`v_rest=−70.6`=θ₋ kills tonic LTD; LTP-dominant 14:8
  amplitudes; **no L1 renorm**; native ū² homeostasis), sweeping A_plus×init on
  MNIST. **All trained cells still collapse to silence** (Δ=−59 to −68 pp,
  fire/img≤0.6) while random baselines fire fine (137–279 spk/img, 77–79%).
  ⇒ the failure is **INTRINSIC to the competitive-FC regime, not** a
  weight-management artifact. Voltage fails under the D&C scheme (forgetting) AND
  its own native homeostasis at Clopath's parameters (silence). The positive
  control + this follow-up together close both reviewer objections: "buggy code"
  and "foreign weight rule."

## One-line for the paper
> *A positive control confirms our voltage-STDP code reproduces Clopath's
> spike-timing window at his published parameters; the FC failure is therefore a
> regime effect — the competitive scaffold's weight management forces an
> LTD-dominant, tonic-depression configuration — not an implementation error.*
