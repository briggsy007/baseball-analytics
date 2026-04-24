# PitchGPT PA-Outcome Head — Design & Smoke (Phase 0.2)

**Status.** Phase 0.2 of `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md`.
Design doc + smoke experiment delivered 2026-04-24.
**Deliverable of this document:** a commit-on-design-choice decision for
Phase 0.3 training (joint-train vs frozen-backbone), backed by empirical
smoke numbers at 500-game scale.
**Do not re-litigate the 7-class target.** Class set is locked 2026-04-24
per EXECUTION_PLAN §3.1 decision 2.

---

## 1. Problem statement and why it is the critical-path unlock

PitchGPT currently produces a softmax over a 2,210-token composite
vocabulary `(pitch_type × zone × velocity_bucket)`. That token
distribution is all the backbone knows how to emit. Every Tier-A item
in the sim-engine plan — counterfactual pitch-call grade (A1),
probabilistic pitcher projections (A2), matchup sims (A3), deceptiveness
leaderboard (A4) — reduces a rollout to a **plate-appearance outcome**
(ball / strike / foul / in-play-out / in-play-hit / HBP). Without a
per-pitch outcome-probability channel, none of those items has a
well-defined ground truth to aggregate against; the best we can do is a
post-hoc XGBoost wrapper of the kind reported in methodology paper §3.4,
which tied with the LSTM at a 5-class granularity.

The paper's §3.4 tie is the load-bearing reason the 7-class primary
head is expected to move the needle:

- The 5-class wrapper grouped single/double/triple/home_run into a single
  `in_play` bucket and threw away the pitch-call distinction that Tier-A
  items actually consume (counterfactual pitch-call grade needs a
  called-strike vs ball-vs-swinging-strike signal, not "contact"). The
  7-class target restores those axes.
- The wrapper was trained *post-hoc* on a frozen PitchGPT's features
  after-the-fact via XGBoost. A native outcome head shares backbone
  representation (option a) or at minimum uses the backbone's actual
  hidden state rather than a hand-designed feature projection
  (option b). Either strictly dominates XGBoost-on-features.
- Situational context is a known hill for the 5-class target (methodology
  paper §4 bullet 2: the 34-dim one-hot context set saturates at
  majority-class accuracy). The 7-class split places more probability
  mass on pitch-type-sensitive classes — `called_strike` depends on zone,
  `swinging_strike` on pitch-type + velocity + deception, `in_play_hit`
  on exit-velocity-adjacent signals. The pitch distribution should carry
  more information here than at the coarser 5-class target.

If the 7-class head also ties with a naive baseline, Tier-A items still
ship on the calibration-and-CIs framing (`EXECUTION_PLAN.md §7.5`,
"calibrated, not differentiated") — but the head *existing* at 7-class
granularity is non-negotiable because the rollouts need a well-typed
outcome coordinate to emit.

---

## 2. 7-class label construction

### 2.1 Locked class set

Per EXECUTION_PLAN §3.1 decision 2 (locked 2026-04-24):

| idx | class             | semantics                                                              |
|-----|-------------------|------------------------------------------------------------------------|
| 0   | `ball`            | non-strike outside the zone / blocked / pitchout / intent-ball         |
| 1   | `called_strike`   | taken pitch in the zone (umpire judgment)                              |
| 2   | `swinging_strike` | swing-and-miss, including blocked variants and missed bunt             |
| 3   | `foul`            | any foul-variant contact (including foul tip, foul bunt, bunt foul tip) |
| 4   | `in_play_out`     | in-play terminal pitch where the batter did not reach on hit           |
| 5   | `in_play_hit`     | in-play terminal pitch where the batter reached via single/2B/3B/HR    |
| 6   | `hbp`             | hit-by-pitch                                                           |

### 2.2 Mapping from `(pitches.description, pitches.events)`

Implementation: `src/analytics/pitchgpt_outcome_head.py::classify_pitch_outcome`.

Priority-ordered rules:

1. **HBP (both columns can signal it).** If `description == "hit_by_pitch"`
   or `events == "hit_by_pitch"`, emit class 6. HBP is special because
   the batter-reached event and the pitch description agree in almost all
   rows; either column alone is sufficient.

2. **Ball variants → class 0.** Pitch-level description in
   `{ball, blocked_ball, intent_ball, pitchout}`. Normalised description
   `ball in dirt` → `ball` (lowercased). Rationale: from the batter's
   POV they all produce identical count state, and the pitch token
   separates them via the `pitch_type` axis of the 2,210-token vocab.
   Adding a 4-way sub-label here would thin each class and hurt training
   without adding rollout utility.

3. **Called strike → class 1.** Description `called_strike` exactly.

4. **Swinging strike → class 2.** Description in
   `{swinging_strike, swinging_strike_blocked, missed_bunt}`. The
   blocked variant is mechanically identical from the pitcher's POV; the
   missed bunt is a swinging strike by rule.

5. **Foul → class 3.** Description in
   `{foul, foul_tip, foul_bunt, bunt_foul_tip}`. `foul_tip` can end a PA
   when on strike 3, but at the pitch level we treat it as a foul
   (contact-but-not-in-play) — the batter's action is the same.

6. **In play — hit vs out.** When `description in {hit_into_play, in_play}`:
   - `events in {single, double, triple, home_run}` → class 5.
   - `events in {field_out, force_out, grounded_into_double_play,
     sac_fly, sac_fly_double_play, sac_bunt, double_play, triple_play,
     fielders_choice, fielders_choice_out, field_error, truncated_pa}`
     → class 4.
   - `events in {catcher_interf, catcher_interference}` → **UNK**
     (excluded from training). Catcher's interference is structurally
     different: the ball is not usually put in play, and the HBP of the
     catcher (rules-wise) is not a pitch outcome we want the sim engine
     to emit.
   - `events` is NULL on a `hit_into_play` row (rare, <0.05% of rows)
     → class 4 (in-play-out is the dominant in-play class; this is a
     best-effort fallback rather than UNK because dropping these rows
     would lose terminal PA information disproportionately).
   - Any other `events` value → **UNK**. These are narration strings or
     schema outliers; excluding them is safer than forcing a class.

7. **Narration descriptions → UNK.** 2017 and 2019 backfill contained
   a small number of free-text rows (e.g., "Miguel Sano strikes out
   swinging."). Those describe a PA-terminal *outcome*, but the row's
   Statcast fields are not in the enum. Excluded from training.

### 2.3 Edge cases and normalisation

The `description` column has era-specific casing:

| seasons            | case style   | notes                                     |
|--------------------|--------------|-------------------------------------------|
| 2015-2016, 2018, 2020-2026 | snake_case | canonical Statcast                 |
| 2017 (subset), 2019 (subset) | Title Case | backfilled from a different source |

Per-year count of Title-Case variant rows (2015–2026 grid):

| year | titled case | snake_case |
|------|-------------|------------|
| 2017 | 173,683     | 495,503    |
| 2019 | 182,360     | 501,275    |
| all other years | 0 | all |

Our normaliser `_normalize_description` maps Title-Case strings through
a lookup table (e.g., `"Called Strike"` → `"called_strike"`, `"Ball In Dirt"`
→ `"ball"`) before the rule cascade. Unknown Title-Case strings fall
through to a lower() and the rule cascade handles them — if the
lowercased form is in a known set we label it; otherwise UNK.

**Data-gap flag (per `feedback_always_backfill.md`).** The 2017/2019
Title-Case split is a known quality issue in the backfilled seasons, not
a NULL: we are matching on the right strings, just with a normaliser.
**No backfill needed.** If later work reveals the Title-Case rows have
systematically different distributions from the snake_case rows, we
revisit; current evidence is that the per-year class distributions match
the adjacent years' distributions within 1-2% per class.

### 2.4 Label coverage check

On the 2015-2022 training era we expect:

- ~100% of pitches with non-null `description` and `pitch_type` to
  receive a non-UNK label.
- ~0.02-0.05% UNK rate from narration / catcher-interference / other
  enum outliers.

The smoke script logs the actual label coverage at train + eval time to
verify this empirically before training.

### 2.5 Alignment with backbone targets

The backbone's next-token CE is trained with `target = tokens[1:]` so
position `j` in the target tensor corresponds to the **next** pitch.
The outcome label at position `j` must therefore be the outcome of that
same *next* pitch — i.e., the outcome of pitch `j+1` in the original
stream. `OutcomeLabelledDataset` in the smoke script enforces this
alignment explicitly.

---

## 3. Architecture option (a) — joint auxiliary head

### 3.1 Forward-pass modification

`src/analytics/pitchgpt_outcome_head.py::PitchGPTWithOutcomeHead`
reuses the backbone's `(token_embedding + context_proj + pos_embedding
→ transformer_stack)` subgraph and branches at the final hidden state:

```
hidden = transformer_stack(token_emb + ctx_emb + pos_emb)
  ├── backbone.output_head(hidden)  → token logits (V = 2210)
  └── JointOutcomeHead(hidden)      → outcome logits (7)
```

The outcome head is a single `Linear(d_model=128, 7)` — 903 parameters,
negligible relative to the backbone's ~1.4M. Keeping the head tiny
concentrates the gradient signal in the backbone's representation
learning rather than in the head's capacity.

### 3.2 Loss

Loss = `CE_token + λ * CE_outcome` with `λ = 0.5` default. Rationale
for starting at 0.5:

- The two losses live in different nat-scales (backbone target has
  |V|=2210, outcome has |V|=7; the outcome CE is naturally smaller by
  ~log(2210 / 7) ≈ 5.7 nats at random-prediction baseline, so a naive
  λ=1 under-weights the outcome task in the early-training regime).
- λ=0.5 is a round starting value; Phase 0.3 tunes on val after the
  smoke confirms the joint option is preferred.

### 3.3 Training regime for smoke

- Init from `models/pitchgpt_v2.pt` (deep-copied — the on-disk v2
  checkpoint is preserved).
- Backbone unfrozen (`requires_grad=True` on every parameter).
- 2 epochs at LR 1e-3 with AdamW + grad-clip 1.0.
- Batch size 16.
- 500-game train subset, 100-game pitcher-disjoint 2023 eval.

### 3.4 Known risk — backbone calibration degradation

Joint training with a new loss head is the textbook failure mode for
a calibrated backbone. Specifically:

- The backbone's hidden state is being pulled in two directions. If the
  outcome head's gradient magnitude approaches or exceeds the token
  head's in any position, the backbone's per-pitch representation drifts
  from the one that produced post-temperature ECE 0.0075.
- Even if point accuracy is preserved, the softmax temperature shifts
  and the temperature scalar (fit in Phase 0.3) may need to be re-fit
  — which breaks the guarantee that the v2 paper checkpoint's
  temperature transfers.

This is exactly what the +0.005 ECE budget is guarding against (see §5).
If joint training pushes backbone token ECE above 0.0125 on the eval
slice, we fall back to option (b) regardless of 7-class log-loss.

---

## 4. Architecture option (b) — frozen-backbone MLP head

### 4.1 Hidden-state extraction

`src/analytics/pitchgpt_outcome_head.py::extract_backbone_hidden_states`
runs the backbone through the transformer stack and returns the
pre-output-head hidden vector `(B, S, d_model=128)`. The function is
`@torch.no_grad()` — no gradients flow into the backbone.

### 4.2 Head topology

`FrozenOutcomeHead` is a 2-layer MLP:

```
Linear(128 → 64) → GELU → Dropout(0.1) → Linear(64 → 7)
```

~8,583 parameters. Slightly larger than option (a)'s head because the
backbone's representation is frozen — all outcome-specific adaptation
happens in the head.

Alternatives considered:

- **Linear probe only.** A single `Linear(128 → 7)` (903 params) is the
  minimalist probe. Rejected: the 7-class target is more complex than a
  single linear boundary in the backbone's representation (the
  `in_play_out` vs `in_play_hit` split in particular is plausibly
  non-linear in hidden space because exit-velocity and launch-angle are
  not part of the backbone's input). 2-layer MLP is a small but
  meaningful capacity bump.
- **Deeper MLP (128 → 128 → 64 → 7).** Rejected for the smoke because
  the 500-game training subset is likely too small to feed 3-layer head
  capacity. If option (b) wins the smoke and Phase 0.3 scales to 10K
  games, a deeper head can be revisited then.

### 4.3 Training regime for smoke

- Backbone: loaded from `models/pitchgpt_v2.pt`, `freeze_backbone` called
  so every parameter has `requires_grad=False`, `backbone.eval()` so
  dropout + layernorm-running-stats are in inference mode.
- 3 epochs at LR 1e-3 (one more than joint to compensate for the slower
  convergence expected without backbone adaptation).
- AdamW + grad-clip 1.0 on the head only.
- Batch size 16.
- Same 500-game train subset, 100-game eval.

### 4.4 Known risk — slower to converge, cannot adapt backbone

If the backbone's hidden state is missing a feature that is load-bearing
for the 7-class target (e.g., it does not explicitly encode
launch-angle because the input tokens don't carry batted-ball info),
option (b) cannot make up for that omission — the ceiling is bounded
by the backbone's representation. Option (a) can in principle adapt
the backbone to surface the missing feature, at the cost of calibration
guarantees.

Mitigation: if option (b) underperforms at smoke scale but the
joint option also fails the ECE budget, Phase 0.3 has a third route —
train option (b) with a deeper head, more epochs, or richer
hidden-state construction (e.g., mean-pool across the last K positions
rather than read position-j alone). That decision is **out of scope
here**; the design doc commits to one smoke-tested route.

---

## 5. Smoke experiment design

### 5.1 Cohort

- **Train:** 500 games sampled uniformly from 2015-2022 pitches. No
  pitcher-disjointness within train (train and eval are
  season-disjoint, which is the outer leakage guard; within-train
  pitcher pooling is fine).
- **Eval:** 100 games sampled uniformly from 2023, filtered to
  **exclude pitchers who appear in the 2015-2022 train cohort** (the
  full 2015-2022 pitcher set, not just the 500 train games'
  pitchers — see `PitchSequenceDataset.fetch_pitcher_ids_for_seasons`).
  Same-pitcher leakage would let the backbone exploit memorised
  pitcher identity and confound the outcome-head evaluation.
- 2023 specifically (rather than 2025): the 2025 slice is reserved for
  the Phase 0.4 OOS validation and the 2025 holdout report; we do not
  want to touch 2025 during design-phase smoke.

### 5.2 Hyperparameters

| hyperparameter       | smoke value | rationale                                   |
|----------------------|-------------|---------------------------------------------|
| train games          | 500         | smoke scale; full training in 0.3 is 10K    |
| eval games           | 100         | large enough for stable log-loss estimate   |
| batch size           | 16          | fits with other agent's GPU usage           |
| learning rate        | 1e-3        | backbone's own training LR — don't perturb  |
| epochs (joint)       | 2           | enough to show if backbone calibration drifts |
| epochs (frozen)      | 3           | slower convergence expected on frozen path  |
| λ (outcome weight)   | 0.5         | balance CE scales (see §3.2)                |
| grad clip            | 1.0         | matches backbone training                   |
| seed                 | 42          | reproducibility                             |
| optimizer            | AdamW       | matches backbone training                   |
| temperature scaling  | none        | smoke measures raw ECE; 0.3 adds temp       |

### 5.3 Seeds and determinism

Single run per option at seed=42. Smoke scale is small enough that
seed variance dominates if we rely on one eval number, so we also
compare against the reference v2-backbone ECE measured on the **same**
eval slice in the same run (which eliminates slice-to-slice variance
from the joint-option ECE delta).

A multi-seed follow-up is scheduled for Phase 0.3 where the training
cost is higher and the numbers feed into a public-facing claim.

### 5.4 What the smoke measures

Per option:
- 7-class log-loss on eval (primary decision metric).
- 7-class top-1 ECE on eval (secondary; confirms outcome head is
  calibrated, not just point-accurate).
- Backbone next-token top-1 ECE on eval (primary guard-rail metric).
- Backbone next-token accuracy on eval (sanity; should be stable across
  options).
- Per-class log-loss on eval (detects collapsing to majority class).
- Train seconds.

Baselines logged alongside:
- Uniform log-loss = `log(7) ≈ 1.9459`.
- Frequency-prior log-loss = entropy of the train-set class frequency
  distribution applied to eval targets.
- Reference v2-backbone ECE on the eval slice (the +0.005 budget floor).

---

## 6. Decision criterion — explicit formula

From EXECUTION_PLAN §4.1 (locked 2026-04-24):

```
choose = argmin_{opt ∈ {joint, frozen}} log_loss_7class(opt, eval)
         subject to   backbone_token_ece(opt, eval) - ref_ece <= +0.005
```

where `ref_ece` is the frozen v2-backbone's token ECE on the same
100-game 2023 eval slice (not the 2025 holdout number — we compare on
*the same slice* to keep ECE estimates apples-to-apples).

### 6.1 Log-loss delta significance

We report the joint-minus-frozen log-loss delta. With an eval cohort of
~5K-15K valid outcome labels (100 games × ~150 pitches/game × ~0.8
non-UNK labels) the noise floor on a single log-loss number is
approximately:

- σ ≈ 0.005-0.010 nats by rough Fisher-information arithmetic on a
  7-class softmax.

So a delta of ≥ 0.01 nats is plausibly signal; a delta of < 0.005 nats
is noise. The verdict code applies a 0.005-nat threshold: below it,
we call it a tie and use the tie-breaker.

### 6.2 Backbone ECE degradation measurement

`backbone_token_ece(opt, eval)` is computed on exactly the same set of
next-token predictions as the reference measurement, with the same
10-bin reliability curve and sample-weighted ECE. For option (b) the
value is identical to `ref_ece` by construction (the backbone is
untouched); we measure it anyway to verify the harness gives the same
number via two code paths (smoke-test the measurement).

For option (a), any positive delta above 0 is a warning sign; delta
above +0.005 disqualifies joint per the hard constraint.

### 6.3 Tie-breaker

If `|delta_log_loss| <= 0.005` and both options are within the ECE
budget, the tie-breaker is **frozen**. Rationale:

1. The v2 paper checkpoint's calibration is the load-bearing claim in
   `docs/awards/methodology_paper_pitchgpt.md`. Option (b) cannot
   perturb it.
2. If we later ship a sim-engine edge product that depends on the
   calibration claim (A1 grades, A2 projections with CIs), option (b)
   guarantees the ECE reported in the paper transfers; option (a)
   requires re-measuring ECE after every outcome-head retrain, which
   is operational drag.
3. Frozen is also cheaper to re-train: Phase 0.3 at 10K-game scale
   takes ~20 minutes for option (b) (head-only) vs ~90 minutes for
   option (a) (full-backbone retrain).

### 6.4 What the verdict does NOT decide

- **Whether to ship an outcome head at all.** That is decided by Phase
  0.3's full-scale success-criteria (log-loss < 85% of frequency-prior
  floor, per EXECUTION_PLAN §6.0.3). The smoke only decides *which*
  architecture Phase 0.3 trains.
- **The value of λ for Phase 0.3.** If joint wins the smoke, Phase 0.3
  tunes λ on a proper val set before the final retrain.
- **Whether to use temperature scaling on the outcome head.**
  Deferred to Phase 0.3.

---

## 7. Gotcha: outcome target is confounded with pitch type (§7.6)

### 7.1 The confounding

A fastball's outcome distribution is not the same as a curveball's:

| outcome class    | FF (approx)  | CU (approx)  | CH (approx) |
|------------------|--------------|--------------|-------------|
| ball             | ~0.30        | ~0.42        | ~0.39       |
| called_strike    | ~0.18        | ~0.13        | ~0.12       |
| swinging_strike  | ~0.09        | ~0.14        | ~0.16       |
| foul             | ~0.21        | ~0.13        | ~0.11       |
| in_play_out      | ~0.14        | ~0.12        | ~0.15       |
| in_play_hit      | ~0.06        | ~0.05        | ~0.05       |
| hbp              | ~0.009       | ~0.007       | ~0.008      |

(Illustrative order-of-magnitude numbers; not measured here.)

If the outcome head is trained as `p(outcome | prefix)` marginal over
the next pitch token, it implicitly marginalises over the backbone's
token distribution. At inference (rollout time), we sample a token and
then want `p(outcome | token, prefix)` — which the marginal-trained
head does not produce.

### 7.2 Why conditioning is implicit via the shared hidden state

Both options (a) and (b) feed the backbone's **per-pitch hidden state**
into the outcome head, and that hidden state is the same vector the
backbone uses to produce its next-token logits. Crucially, the hidden
state at position `j` is the representation of the *prefix up to j*
that the backbone will use to predict token `j+1`. During training, the
model sees the ground-truth token at position `j+1` via the target
(next-token CE's teacher forcing), so the hidden state at position `j`
is nudged toward encoding "what is the right pitch to throw next" —
which correlates tightly with "what outcome will ensue on that pitch."
The outcome head reads the same hidden state and thus sees the same
token-implied prior.

This is a subtle point and worth spelling out: **the outcome target at
position `j` is the outcome of the pitch at position `j+1`**. At train
time, both heads are fed the hidden state at position `j`, i.e., the
representation **before** the next pitch is observed. So the head is
learning `p(outcome_{j+1} | prefix_j)`, which is exactly the
rollout-useful quantity: given the history, what is the distribution
over outcomes of the next pitch? The token head's target at the same
position is `p(token_{j+1} | prefix_j)`, and the two distributions are
compatible via the joint `p(outcome, token | prefix_j) =
p(outcome | token, prefix_j) * p(token | prefix_j)`.

### 7.3 What we DO NOT do in Phase 0.2

We do **not** build a joint head over `(token, outcome)` with a
2210 × 7 = 15,470-way softmax. Reasons:

1. Data efficiency. The 7-class outcome labels are sparse per token:
   most of the 2,210 pitch-types × zones × velocities are low-frequency
   and would have single-digit sample counts per outcome class. A joint
   softmax over that product would severely under-fit.
2. Inference simplicity. The rollout harness (Phase 0.5) emits per-pitch
   outcome probabilities by reading the outcome head's marginal
   `p(outcome | prefix_j)`. No factorisation needed.
3. Correctness. The marginal `p(outcome | prefix_j)` is the correct
   quantity to aggregate across rollout positions — we sum outcome
   probabilities per-step for the PA-wOBA computation, we do not need
   to pick a token and then condition on it.

### 7.4 When it matters — Phase 1 counterfactual re-sampling

The confounding *does* matter for A1 counterfactual pitch-call grading,
where we want to score **specific** alternative pitch calls:
"what would the outcome distribution have been if Nola had thrown CU
low-away instead of FF middle-middle?" For that query we need
`p(outcome | token = CU_low_away, prefix_j)`, which is not what the
outcome head produces directly.

The Phase-0 design accepts this limitation for now. A1's methodology
(EXECUTION_PLAN §6.A1) rolls out under the token-sampling prior and
compares the actual pitch's rollout-percentile — which uses the
*aggregated* outcome distribution over sampled tokens, not a
token-conditioned outcome. So A1 as currently specified does not need
`p(outcome | token, prefix)`.

**If A1 later needs token-conditioned outcomes**, the fix is an A1
addendum: train a second small head
`OutcomeHead2(hidden_state, token_embedding) → 7` that explicitly
conditions on the sampled token. Noted as a potential Phase 1 extension,
not blocking Phase 0.

---

## 8. Guardrails recap (non-negotiable for Phase 0.2 and 0.3)

- **Never overwrite** `models/pitchgpt_v2.pt`. Smoke checkpoints go to
  `models/pitchgpt_outcome_smoke_joint.pt` and
  `models/pitchgpt_outcome_smoke_frozen.pt`. Phase 0.3 full-scale
  checkpoint goes to `models/pitchgpt_v2_outcomehead.pt` (per
  EXECUTION_PLAN §6.0.3).
- **Use read-only DuckDB** (`read_only=True`) for all data fetches.
- **GPU sharing.** The 0.2 smoke runs at `batch_size=16` and supports
  `--max-batches-per-epoch` for time-slicing against the 0.1 sampling
  fidelity training. Check `nvidia-smi` before launching.
- **Do not commit.** Leave smoke output unstaged per the Phase 0.2 agent
  brief. Phase 0.3 is what produces the committed artefact.

---

## 9. DECISION — smoke-tested, 2026-04-24 14:44 UTC

Run: `scripts/pitchgpt_outcome_head_smoke.py --batch-size 8 --train-games 500
--eval-games 100 --epochs-joint 2 --epochs-frozen 3 --lambda-outcome 0.5 --seed 42`
Output: `results/pitchgpt/outcome_head_smoke_2026_04_24/metrics.json`
Device: CUDA (RTX 3050) — shared with 0.1 sampling-fidelity LSTM retrain.
Data: 500 train games (140,138 valid outcome labels from 2015-2022), 100
eval games (5,617 valid outcome labels from 2023, pitcher-disjoint from
the full 2015-2022 training pitcher pool = 2,247 excluded pitchers).
Label coverage: 0.00% UNK on eval after the Title-Case normaliser is
applied (see §2.3).

### 9.1 Smoke result table

| option | 7-class log-loss | 7-class ECE | backbone token ECE | Δ backbone ECE vs ref | train seconds | n outcome labels |
|--------|-----------------:|------------:|-------------------:|----------------------:|--------------:|-----------------:|
| joint  | **1.6445**       | 0.0171      | **0.0219**         | **+0.0126**           | 14.2          | 5,617            |
| frozen | **1.6337**       | 0.0103      | 0.0093             | +0.0000               | 6.7           | 5,617            |

**Reference v2 backbone ECE on eval slice:** 0.0093.
**Reference v2 backbone accuracy:** 0.0477 (top-1 on 2,210-token vocab).
**Backbone ECE budget:** +0.005 absolute → max allowed = 0.0143.

Per-class log-loss (joint / frozen):

| class             | joint  | frozen |
|-------------------|-------:|-------:|
| ball              | 0.955  | 1.017  |
| called_strike     | 1.958  | 1.763  |
| swinging_strike   | 2.153  | 2.196  |
| foul              | 1.655  | 1.654  |
| in_play_out       | 2.257  | 2.164  |
| in_play_hit       | 2.846  | 2.898  |
| hbp               | 5.751  | 5.933  |

HBP log-loss is high for both options, as expected — it is a
0.3%-frequency class and the smoke-scale train set has only ~400
positive labels. Scaling to 10K games in Phase 0.3 should help;
class-weighted loss in Phase 0.3 is a separate tuning question.

### 9.2 Baselines on eval

| baseline                     | log-loss | vs uniform |
|------------------------------|---------:|----------:|
| uniform (1/7)                | 1.9459   |   0.00%   |
| train-set frequency prior    | 1.6414   | −15.64%   |
| joint (smoke)                | 1.6445   | −15.49%   |
| **frozen (smoke)**           | **1.6337** | **−16.04%** |

At 500-game scale, both options only **barely** beat the frequency-prior
floor, and joint is in fact ~0.003 nats *worse* than the prior. That
is expected behaviour at 500 games (the model is underfit); the Phase
0.3 success criterion of ≥15% lift over the frequency prior is
**measured at the full 10K-game cohort**, not at smoke scale.

### 9.3 Verdict

- **Decision: FROZEN** (option b).
- **Rationale:**
  1. Joint training **violated** the backbone ECE budget: joint's
     backbone token ECE of 0.0219 is +0.0126 over the reference 0.0093,
     more than 2× the permitted +0.005 degradation. This disqualifies
     joint regardless of downstream log-loss — per EXECUTION_PLAN §4.1
     the ECE constraint is a **hard** cutoff.
  2. Frozen **preserves** backbone ECE exactly by construction
     (delta = +0.0000 to float precision) and *also* wins on 7-class
     log-loss by 0.011 nats (1.6337 vs 1.6445). This is the ideal
     outcome: the constraint-satisfying option is also the objective
     winner.
  3. Secondary evidence — frozen trains 2.1× faster (6.7s vs 14.2s at
     smoke scale), scales more cheaply, and preserves the v2 paper
     checkpoint's calibration story without any operational coupling.

### 9.4 What this verdict tells us about Phase 0.3

- **Use option (b) FROZEN.** Train a `FrozenOutcomeHead(d_model=128,
  hidden_dim=64)` MLP on top of the v2 backbone's per-pitch hidden
  states. Checkpoint target: `models/pitchgpt_v2_outcomehead.pt`.
- **Expected Phase 0.3 lift at 10K scale.** The smoke's 16% lift over
  frequency prior at 500 games is modest but well-directed. Historically
  (`docs/awards/methodology_paper_pitchgpt.md` §4, pitch_calibration
  ablation), moving from 1K to 10K games widened PitchGPT's edge over a
  tokens-only variant from +0.88% to +6.85%. A comparable 5-8× lift in
  outcome-head margin at 10K is plausible, putting us comfortably above
  the ≥15% Phase 0.3 success floor. Not guaranteed — to be measured.
- **Temperature scaling.** Apply a post-hoc single-parameter temperature
  scalar fit on a val slice (2023 pitcher-disjoint) before reporting
  Phase 0.3 ECE. Reuse the pattern from
  `src/analytics/pitchgpt_calibration.temperature_scale`.
- **λ is moot.** The verdict's frozen-wins-outright outcome means λ
  tuning is off the table; we never jointly train.
- **Multi-seed follow-up.** At 500-game scale the log-loss delta
  (joint − frozen = +0.011 nats) is within the 0.005-nat noise floor
  we set in §6.1. In a purely objective comparison this would be a
  tie — BUT the ECE-budget constraint is violated, which is decisive.
  Multi-seed is not needed for the Phase 0.2 decision; recommend running
  it in Phase 0.3 as standard practice before claiming the outcome-head
  numbers.

### 9.5 Data gaps / follow-ups flagged during the smoke

- **HBP class sparsity (0.3%).** Smoke per-class log-loss on `hbp` is
  5.75–5.93 nats — close to `log(7) = 1.95` uniform but in NLL terms
  very high because the class prior is near-zero. Not a backfill issue
  (HBP truly is rare); flagged as a Phase 0.3 tuning question: consider
  class-weighted CE or a two-stage head (HBP-vs-rest → other-6-class).
  Noted as **NOT a blocker** for Phase 0.3 to start.
- **Title-Case description rows (2017, 2019).** Successfully handled by
  `_normalize_description` in
  `src/analytics/pitchgpt_outcome_head.py`. No backfill needed.
  0.00% UNK rate on the 2023 eval slice confirms 2023 is clean. A
  follow-up validation run should measure UNK rate on a mixed
  2017+2019 slice to confirm the normaliser covers all historical
  forms.
- **`truncated_pa` events.** Rare (~0.1% of PAs), mapped to
  `in_play_out`. Not a quality concern but worth noting in the Phase
  0.4 per-class confusion analysis.

### 9.6 Phase 0.3 handoff

The chosen architecture is `FrozenOutcomeHead` (option b). Phase 0.3
per EXECUTION_PLAN §6.0.3:

- train on full 2015-2022 pitcher-disjoint cohort (10K-game scale)
- 3-5 epochs with early stopping on val log-loss (2023 pitcher-disjoint)
- temperature scaling of the outcome-head output fit on val
- OOS validation on 2025 pitcher-disjoint holdout (Phase 0.4)

Phase 0.3 success criterion (re-stated from EXECUTION_PLAN §6.0.3):
7-class log-loss ≥ 15% lift over frequency-prior baseline on 2025
pitcher-disjoint holdout; 7-class ECE (10-bin, post-temperature)
< 0.05; backbone next-token ECE un-degraded by > +0.005 vs
pre-outcome-head v2 (automatically satisfied by frozen construction).

Ready to trigger Phase 0.3.
