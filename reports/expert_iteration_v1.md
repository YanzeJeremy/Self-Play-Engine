# Expert iteration v1: self-play SFT on hidden-rule RPS

*Branch `yanze/expert-iteration-v1`. Generation and selection for `ei-v1` are done (section "Data from the first run"); SFT and evaluation are not yet run.*

## Question

Can supervised fine-tuning on the model's **own successful self-play** teach it to find a hidden opponent pattern by reasoning? The pattern is never stated in any prompt or training example.

Success means beating the **held-out copycat** opponent better than the base model does, along with the other rules, under the same evaluation.

This is variant A from the discussion: expert iteration, where the model is trained on its own replies from games it won. The other variants were B, opponent-move prediction, and C, oracle demonstrations, a control only.

## Design

### Opponents

The new `hidden_rule_rps` game is copycat RPS with a configurable opponent rule (`self_play/games/hidden_rule_rps/`). Rules, prompts, phases, opponent lines and rewards are all unchanged. Only the opponent's move rule differs:

| Rule | Opponent plays | Always-rotate gets | Role |
|---|---|---|---|
| `copy_last` | the player's previous move (copycat) | wins every round | **held out**: evaluation only |
| `counter_last` | what beats the player's previous move | draws every round | train |
| `beaten_by_last` | what the player's previous move beats | loses every round | train |
| `copy_two_back` | the player's move from two rounds ago | loses ~every round | train |
| `repeat_self` | its own opening move, every round | wins 1 round in 3 | train |
| `cycle` | what beats its own previous move | depends on the phase | train |

The always-rotate column is checked in `tests/test_expert_iteration.py`. Three of the training rules depend on the player's own past moves, as copycat does. A memorized rotation fails all three, so improvements can't come from the rotation shortcut.

### Stages (`scripts/run_expert_iteration.sh`)

| # | Stage | What happens |
|---|---|---|
| 1 | generate | vLLM serves base gemma-4-31b-it, data-parallel over both GPUs. 200 games × 20 rounds against each of the 5 training rules (1,000 games). Sampled at T=1.0, top_p 0.95, top_k 64, max 2,048 tokens. Standard prompts, no forecasts, seeds 0–199. |
| 2 | select | `select_sft_data.py`; details below. |
| 3 | control | With `--control random` (default): an equally sized control dataset from the **same training games**. Random round-2+ calls are taken whatever their outcome, matched count for count per opponent and call kind (`select_sft_data.py --mode random`). It tests whether the success filter adds anything beyond exposure to the task. |
| 4 | sft | Once per arm (`sft`: selected data; `rand`: control data, with the same number of steps). LoRA SFT with prime-rl (`make_sft_config.py`): rank 16, alpha 32, lr 1e-4 cosine, 1 epoch, seq_len 8,192, 16 packed rows per step, both GPUs. Gemma-4 needs the `gemma4` renderer and the fused LM head off. The frozen base and the LoRA weights are held in bfloat16, as in prime-rl's LoRA example. A checkpoint is saved every quarter of the run (at least every 25 steps); only the latest is kept. |
| 5 | export | Per arm. `export_lora_adapter.py`: reads only the LoRA tensors from the DCP checkpoint and writes the adapter as prime-rl's own adapter broadcast does (`adapter_model.safetensors` + `adapter_config.json`). prime-rl has no exporter for SFT checkpoints. **Shared key/value layers:** gemma-4's 10 full-attention layers (5, 11, …, 59) have no `v_proj`, because their values are the `k_proj` output. vLLM loads `k_proj` into both the K and V slots but applies LoRA per slot, so the `k_proj` adapter is also written as `v_proj` for exactly those layers. Otherwise the served values would lack the trained update. **Fidelity check:** after writing, the adapter is reloaded with vLLM's own loader, and every module's update, including the shared V slots, is compared with the checkpoint's on identical random inputs. A mismatch fails the export (`fidelity.json`). |
| 6 | evaluate | One vLLM server with the base model plus both adapters. First, `check_adapter.py` compares base and each adapter's likelihood of that adapter's training examples through the server. It stops the run if an adapter changes nothing, if (full run) it doesn't make its training data more likely, or if the check can't run (`--allow_unchecked` overrides this, recorded as bypassed). Then greedy (T=0) evaluation of both models. **10 rounds:** held-out copycat, 100 games on seeds 0–99 (the same condition as the earlier B0 primary baseline, 43.9% of rounds 2–10 won); each training rule, 40 games on fresh seeds 50,000–50,039, never used for training. **20 rounds:** copycat, 50 games; each training rule, 20 games. **Diagnostic suite** (30 games per condition, 20 rounds, seeds 70,000+, frozen before training), listed below. Each adapter is named and evaluated under a hash of its weights. |
| 7 | summarize | `summarize.py` pairs runs game by game on seeds, per rule, horizon and diagnostic condition, with the strict checks (complete, same condition except the model). Diagnostics are scored without round 1 and without forced rounds, and add recovery and before/after-switch rates; there are round-by-round win rates throughout. Bootstrap 95% intervals are taken over games. Outputs: `summary.md` (sft vs base), `summary-rand.md` (rand vs base) and `summary-sft-vs-rand.md` (does the success filter matter), each with a `.json`. |

**The diagnostic suite** (hidden-rule game options, `self_play/games/hidden_rule_rps/game.py`):

| Condition | Settings | What it separates |
|---|---|---|
| shuffled phrases (copycat) | the generic lines are drawn at random each round, not tied to the round; the move claims stay on their rounds | play that depends on which phrase precedes which move |
| forced moves (copycat) | 2 seeded rounds where the player's move is forced; the history shows it; scored without them | recovery after an unexpected history. "Always beat my own last move" survives this, so it isn't enough alone |
| rule switch | copycat, then an unannounced switch to counter_last at round 11 | a memorized rotation (draws after the switch) from adapting to the opponent |
| lag-3 (held out) | the opponent copies the player's move from 3 rounds ago; never trained on | discovery of an unseen reactive rule |
| shuffled phrases (counter_last) | as above, on a training rule | the same phrase dependence where the success data had it |

### Data selection: what counts as "good"

- **Which rounds qualify.** A round from the 2nd on qualifies when the player **won it and won ≥ 80% of that round and the 4 after it**, with at least 3 of those rounds played. Round 1 is excluded because it's chance.
- **The examples.** Both calls that decided a qualifying round become examples: the **thinking turn** before the move (its notes drive the move) and the **move** itself (its final, accepted attempt). Each is stored as the exact prompt plus the model's reply, ending in `[STOP]`. Talk turns are not trained.
- **Always dropped:**
  - calls without a readable answer;
  - moves that weren't accepted;
  - replies cut at the token limit;
  - examples over seq_len − 256 tokens;
  - duplicates.
- **Validation split:** 5% of games, chosen by game *before* any example is taken, so no game contributes to both sets.
- **Budget, balanced across opponents** (`--target_min` 5,000 to `--target_max` 10,000; a budget target, not a requirement):
  - within each rule, examples are ordered by higher reward-to-go, then earlier rounds, with at most 16 per game;
  - the budget N is allocated across rules **before** anything is taken: the largest N up to the target where every rule gets at most **35% of the final train set** (`--max_rule_share`);
  - when one or two rules dominate the yields, N shrinks and `stats.json` marks it `diversity_limited`; if no N satisfies the cap (fewer than 3 rules with data), selection fails and says so;
  - validation takes the same proportions from the validation games;
  - every count reported is computed from the files written.
- **Threshold: strict only by default** (`--thresholds 0.8`). A uniformly random player passes the 80% filter in 3.7% of rounds, and the relaxed 60% filter in 13.6%, so relaxing would admit a lot of luck. Relaxing is opt-in (`--thresholds 0.8,0.6`), and every example records whether it passed the strictest threshold (`meta.strict`).
- **The run fails below 500 examples** (`--hard_min`).
- **Outputs:**
  - `data/dataset/`: what SFT reads;
  - `data/meta/`: per example, its rule, seed, round, call, window win fraction, strictness and tokens;
  - `data/stats.json`: per rule, the examples, unique games and strict examples, the qualification rate next to the random-play rate, the allocation, whether the run was diversity-limited, and what was dropped.

## Running

In the 2×H200 allocation, from the repo root. **Stop your own `vllm serve` first**: the pipeline starts and stops its own servers, and refuses to start if the GPUs or port 8000 are busy. Use tmux.

```bash
bash scripts/run_expert_iteration.sh --smoke true    # tiny run through every stage first (~30-60 min)
bash scripts/run_expert_iteration.sh                 # the full experiment
```

- **Where outputs go:** everything lands in `$storage_dir/expert_iteration/<name>/` (default name `ei-v1`; the smoke run is `ei-v1-smoke`), in `configs/`, `runs/`, `data/`, `sft/`, `adapter/`, `logs/` and `stages/`. The result is `summary.md`.
- **Resuming:** run the same command again. Finished stages are skipped and unfinished evaluation runs resume. An interrupted SFT resumes from its latest checkpoint, or restarts clean if it never saved one.
- **Rerunning part:** `--redo_from <stage>` reruns that stage and everything after it. Redoing SFT, or anything before it, moves the old training folders to `sft.replaced-<time>/` and `sft-random.replaced-<time>/` rather than deleting them; delete them yourself to free ~65 GB each. Runs of a redone stage whose condition changed are moved to `runs/archive/`; runs whose condition didn't change are reused.
- **Stopping early:** `--stop_after <stage>` stops successfully once that stage is done. For example, `--stop_after select` generates and selects data, then frees the GPUs. Rerun without the flag to continue.
- **Changing settings:** a stage's settings are recorded when it **starts** (`stages/<stage>.params`). Changing one of them afterwards, even for an interrupted stage, is refused unless you redo that stage with `--redo_from`. Settings for stages that never started can change freely.
- **Every setting is a flag**, e.g. `--gen_games`, `--target_min`, `--lr`, `--epochs`, `--lora_rank` and `--eval_games`. See the script header.

**Time estimate for the full run.** Per-call latencies come from the B0 primary run (about 8 s per call at 10 games in flight per GPU).

| Stage | Estimate |
|---|---|
| generate | ~7 h, 50,000 calls (done for `ei-v1`: 6 h 50 min) |
| control | minutes (CPU) |
| SFT | ~2–3 h per arm, about 9M tokens each; ~4–6 h for both |
| evaluate | ~7–8 h: 3 models × (7,500 calls at 10 rounds + 7,500 at 20 rounds + 7,500 diagnostic) |
| server starts | ~10 min each |
| **Remaining for `ei-v1`** | **about 12–14 h** |

To shorten evaluation, drop the 20-round evaluation (`--eval_long_games 0`) or use fewer diagnostic games (`--diag_games 20`). Both are evaluate-stage settings, free to change until evaluation starts.

Disk use is about 65 GB per SFT checkpoint (two arms), since each holds the whole model, plus a few GB of traces.

## What was verified, and what only your GPUs can verify

**Verified here, on CPU:**

| Check | How |
|---|---|
| The opponent rules | Per-rule tests. `copy_last` is prompt-for-prompt identical to `copycat_rps`. The base copycat eval config is the same condition as the B0 primary run. |
| Selection | Window and threshold, both calls kept, round 1 excluded, unusable calls dropped, rule caps, dataset columns. |
| SFT config | Passes prime-rl's own validation (`sft --dry-run`). |
| The data | prime-rl's gemma-4 renderer renders our examples with loss on the reply only, including `[STOP]`. Token counts match the selector's to within a few tokens. |
| The export | Exact values and the prime-rl adapter key format; refuses wrong ranks and incomplete adapters. **Shared key/value layers:** loaded and packed with vLLM's own adapter loader on CPU, the V slot of a shared layer reproduces the trained values. Without the copy it misses the whole update. |
| Selection budget | The share cap holds on the final dataset; a one-opponent dataset is refused; with five opponents a small target is split evenly (no alphabetical truncation); reported counts equal the written ones; random-play pass rates are 1/27 and 33/243. |
| The orchestrator | Runs end to end with stand-ins for vLLM, eval.sh and sft. The SFT stand-in uses prime-rl's own run-folder validator. Covered: an SFT crash before any checkpoint restarts clean; one after a checkpoint resumes; an identical rerun skips every stage; changing a finished stage's setting is refused; `--redo_from sft` moves the old folder aside, trains fresh, and evaluates the new adapter under its own name. |

**First exercised by the smoke run on GPUs:**
- the adapter check (`check_adapter.py`) against the live server, which is the end-to-end numerical comparison of base and adapter;
- vLLM data-parallel serving, and LoRA serving of gemma-4;
- prime-rl's 4-day-old gemma-4 trainer with LoRA, and its memory use;
- the real checkpoint's key names: the export stops with the actual names if they differ;
- whether the 124 GB of CPU RAM in job 6490009 is enough while the model loads. If SFT is killed for memory, request more (`--mem`).

## Data from the first run (`ei-v1`, 2026-10-17)

Generation took about 6 h 50 min (01:55–08:45 PDT): 1,000 games, 0 errors, and 39 of about 50,000 replies hit the token limit (dropped). Selection was strict (0.8):

| Opponent rule | Rounds won, 2–20 | Rounds won, 11–20 | Rounds passing the filter (random play: 3.7%) | Train examples | From games | Share |
|---|---|---|---|---|---|---|
| repeat_self | 94.4% | 100.0% | 93.1% | 2,622 (capped) | 190 | 35.0% |
| cycle | 87.0% | 99.6% | 82.6% | 2,622 (capped) | 187 | 35.0% |
| beaten_by_last | 50.0% | 68.1% | 26.6% | 1,489 | 134 | 19.9% |
| counter_last | 39.3% | 44.0% | 8.9% | 451 | 75 | 6.0% |
| copy_two_back | 36.7% | 39.5% | 4.8% | 308 | 77 | 4.1% |
| **Total** | | | | **7,492** (+ 388 validation) | 663 | |

The "passing the filter" column measures the selection window (a won round followed by ≥80% wins), not the win rate. The win-rate columns are the base model's actual play. A review confirmed: 7,492 examples from 663 training games, strict metadata, the share cap on the final set, and no game or exact example shared between training and validation.

That's 8.8M tokens, with the longest example at 3,197 tokens. The run is `diversity_limited`: the two easy rules hit the 35% cap.

- **The base model already handles the two rules that don't react to it**: about 100% of rounds 11–20 against repeat_self and cycle.
- **Its weakness is recognizing how its own moves drive the opponent.** It wins 68% of late rounds against beaten_by_last, where repeating its last move wins, a common "stick with what won" heuristic. It wins about 40% against counter_last and copy_two_back, close to chance.

So the data that could teach "model the opponent as a function of my own moves" is the thinnest: 759 examples from 152 games. That is the honest limit of one round of expert iteration on this base model.

**What the selected reasoning says, and what it doesn't.** An earlier version of this report concluded from keyword counts that SFT on this data would "most likely teach stronger superstitions". That overstated the evidence:
- **The counts are crude.** About half the selected move replies mention the opponent's phrases, and very few mention the player's own past moves. But some mention phrases only to reject them before correctly identifying a cycle or a constant move.
- **There are examples of correct reactive-rule inference.** For instance, training example 1837 (counter_last) states "they play the move that beats my previous move", supports it with accurate history, and plays the winning counter.
- **The ~75% copycat ceiling applies only to a fixed period-4 action policy,** not to every reply that mentions phrases.
- **Reasoning text is imperfect evidence** of what drives behavior (Turpin et al., 2023).

What can be said: the data is 70% the two non-reactive rules, and the reactive rules where the base model is weakest contribute only 759 examples.

**A concrete shortcut risk:** 35% of the data is `cycle`. Beating a cycling opponent and beating copycat can both be done by a fixed rotation. So holding copycat out of training does not by itself make a copycat gain evidence of discovery. The diagnostic suite exists for this: the rule switch, the lag-3 held-out rule, and shuffled phrases.

**The random control is less different than its name suggests.** Matched per opponent and call kind, 77% of its examples come from won rounds and 60% would pass the strict filter, because the easy rules are mostly won anyway. Its contrast with the selected data is concentrated in the reactive rules.

## Plan (agreed with the second review)

1. Run the GPU smoke test: `bash scripts/run_expert_iteration.sh --smoke true`. It confirms training, checkpoint recovery, export and its fidelity check, LoRA serving, and the adapter check.
2. Keep the evaluation frozen as above: the original 10- and 20-round conditions plus the diagnostic suite, with the same conditions and seeds for every model.
3. Train on the current strict dataset as the pilot, and report its composition as a limitation.
4. Train the random-data control (same games, matched proportions and steps) and compare base, outcome-selected SFT and random SFT. If both SFT models improve similarly, the gain comes from task exposure, not from the success filter.
5. **Next iteration, not yet built:** select candidates more robustly. Sample several replies from the same history, score each across several continuations (varied phrases, occasional forced moves), and keep those that do consistently well downstream, using future outcomes only for scoring. All branches from one history stay in the same split.
6. Opponent-prediction training (variant B) is deferred: it can reward the same shortcuts when histories carry the same correlations.

## Review fixes (after `8cfbf57`)

| # | Issue | Fix |
|---|---|---|
| 1 | The exported adapter dropped the shared key/value update in the 10 full-attention layers, so the served model would miss part of the training. | The `k_proj` adapter is also written as `v_proj` for those layers, found from the model's config by vLLM's own rule. A CPU test with vLLM's loader and packing checks the served values numerically. At serving time, `check_adapter.py` compares base and adapter through the live server. |
| 2 | Retrying SFT, or `--redo_from sft`, failed with the real trainer, which refuses a used run folder. Only the final checkpoint was saved. | Retry: `--resume` from the latest checkpoint, or `--clean` when the previous attempt saved none. Redo: the old SFT folder is moved aside and a fresh one used. Intermediate checkpoints are saved. The stand-in trainer now uses prime-rl's own run-folder validator, so tests catch this. |
| 3 | The 35% cap applied to the target, not to the dataset actually selected, so one rule could make up 100% of it. | The cap is enforced on the final dataset, by an allocation computed before selection. An impossible cap fails with the per-rule yields. |
| 4 | Trimming to the target could drop whole rules alphabetically, while the report claimed otherwise. | The budget is allocated across rules up front, and reported counts come from the written files. |
| Design | Strict and relaxed selection should stay distinguishable; count unique games; evaluate adaptation over time. | Strict-only by default; `meta.strict` on every example; unique games per rule; random-play pass rates beside the observed rates; a 20-round evaluation with round-by-round win rates. Settings are now kept per stage, so a finished stage's settings are protected while later ones can change. |

### Third review (after `9923806`)

| # | Issue | Fix |
|---|---|---|
| 1 | The adapter check passed when it couldn't run, and showed only that the adapter has *an* effect. | It now fails when it can't run; `--allow_unchecked` overrides this, recorded as `bypassed`. Export fidelity is checked at export time: vLLM's loader reloads the adapter, and every module's update is compared with the checkpoint's on identical inputs, shared V slots included. |
| 2 | An interrupted SFT could resume under changed settings. | A stage's settings are recorded when it starts and protected from then on; changing them needs `--redo_from`. |
| 3 | `--redo_from evaluate` with changed settings failed on the old runs' identity check. | A redone stage moves its runs of an older condition to `runs/archive/` and reruns them; runs whose condition is unchanged (the copycat and diagnostic runs, when only `--eval_skip` changed) are reused. |
| Design | Stronger evaluation before training; a random-data control; a tempered interpretation. | The diagnostic suite (shuffled phrases, forced moves, a rule switch, a held-out lag-3 rule); a random-data control arm trained for the same steps; the report's interpretation corrected, with actual win rates added. |

**Still open, as the review notes:**
- A gain on held-out copycat alone doesn't establish general pattern discovery. More held-out rules and action-intervention tests would strengthen that claim.
- More robust candidate selection (step 5 of the plan above).

## Caveats

- **The data comes from the base model's successes, and most are easy rules** (see "Data from the first run"). The per-rule cap limits the imbalance, but the rules that depend on the player's own moves contribute only 759 examples.
- **One round of expert iteration.** If the SFT model improves on the training rules, a second round (generate with the adapter, select, train again) is the natural next step.
- **The held-out copycat result is the one that speaks to discovery.** In-distribution gains on fresh seeds of the training rules are weaker evidence.
