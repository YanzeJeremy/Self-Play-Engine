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
| 3 | sft | LoRA SFT with prime-rl (`make_sft_config.py`): rank 16, alpha 32, lr 1e-4 cosine, 1 epoch, seq_len 8,192, 16 packed rows per step, both GPUs. Gemma-4 needs the `gemma4` renderer and the fused LM head off. The frozen base and the LoRA weights are held in bfloat16, as in prime-rl's LoRA example. A checkpoint is saved every quarter of the run (at least every 25 steps); only the latest is kept. |
| 4 | export | `export_lora_adapter.py`: reads only the LoRA tensors from the DCP checkpoint and writes the adapter as prime-rl's own adapter broadcast does (`adapter_model.safetensors` + `adapter_config.json`). prime-rl has no exporter for SFT checkpoints. **Shared key/value layers:** gemma-4's 10 full-attention layers (5, 11, …, 59) have no `v_proj`, because their values are the `k_proj` output. vLLM loads `k_proj` into both the K and V slots but applies LoRA per slot, so the `k_proj` adapter is also written as `v_proj` for exactly those layers. Otherwise the served values would lack the trained update. |
| 5 | evaluate | One vLLM server with the base model plus the adapter. First, `check_adapter.py` compares base and adapter likelihoods on training examples through the server. It stops the run if the adapter changes nothing, or (full run) if it doesn't make its own training data more likely. Then greedy (T=0) evaluation of both models. **10 rounds:** held-out copycat, 100 games on seeds 0–99 (the same condition as the earlier B0 primary baseline, 43.9% of rounds 2–10 won); each training rule, 40 games on fresh seeds 50,000–50,039, never used for training. **20 rounds:** copycat, 50 games; each training rule, 20 games. The adapter is named and evaluated under a hash of its weights. |
| 6 | summarize | `summarize.py` pairs base and SFT game by game on seeds, per rule and horizon, with the strict checks (complete, same condition except the model). It adds round-by-round win rates to tell faster discovery from better early moves. Writes `summary.md` and `summary.json`. |

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
- **Rerunning part:** `--redo_from <stage>` reruns that stage and everything after it. Redoing SFT, or anything before it, moves the old training folder to `sft.replaced-<time>/` rather than deleting it; delete that folder yourself to free its ~65 GB checkpoint.
- **Stopping early:** `--stop_after <stage>` stops successfully once that stage is done. For example, `--stop_after select` generates and selects data, then frees the GPUs. Rerun without the flag to continue.
- **Changing settings:** a finished stage keeps its settings (`stages/<stage>.params`). Changing one of them is refused unless you redo that stage with `--redo_from`. Settings for stages that haven't run yet can change freely.
- **Every setting is a flag**, e.g. `--gen_games`, `--target_min`, `--lr`, `--epochs`, `--lora_rank` and `--eval_games`. See the script header.

**Time estimate for the full run.** Per-call latencies come from the B0 primary run (about 8 s per call at 10 games in flight per GPU).

| Stage | Estimate |
|---|---|
| generate | ~7 h (50,000 calls) |
| SFT | ~2–3 h (about 25M tokens) |
| evaluate | ~1.5 h |
| server starts | ~10 min each |
| **Total** | **about 11–12 h** |

Disk use is about 65 GB for the SFT checkpoint, which holds the whole model, plus a few GB of traces.

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

| Opponent rule | Rounds that qualify (random play: 3.7%) | Train examples | From games | Share |
|---|---|---|---|---|
| repeat_self | 93.1% | 2,622 (capped) | 190 | 35.0% |
| cycle | 82.6% | 2,622 (capped) | 187 | 35.0% |
| beaten_by_last | 26.6% | 1,489 | 134 | 19.9% |
| counter_last | 8.9% | 451 | 75 | 6.0% |
| copy_two_back | 4.8% | 308 | 77 | 4.1% |
| **Total** | | **7,492** (+ 388 validation) | 663 | |

That's 8.8M tokens, with the longest example at 3,197 tokens. The run is `diversity_limited`: the two easy rules hit the 35% cap.

- **The base model already solves the rules that don't depend on its own moves** (93% and 83%).
- **On beaten_by_last it often wins** (27%). Repeating its last move beats that rule, and "stick with what won" is a common heuristic, so these wins may not reflect inference.
- **It is near chance on copy_two_back** (4.8%, against 3.7% for random play), and only modestly above chance on counter_last.

So the data that could teach "model the opponent as a function of my own moves" is the thinnest: 759 examples from 152 games. That is the honest limit of one round of expert iteration on this base model.

**What the selected reasoning says.** The table below comes from keyword counts over the 3,747 selected move replies, so treat it as rough:

| Rule | Moves | Phrase-based ("the history for this phrase…") | About the player's own past moves | Cycle or sequence |
|---|---|---|---|---|
| beaten_by_last | 745 | 89% | 0% | 10% |
| copy_two_back | 154 | 76% | 3% | 45% |
| counter_last | 226 | 73% | 14% | 29% |
| cycle | 1,311 | 36% | 4% | 93% |
| repeat_self | 1,311 | 52% | 1% | 19% |
| **all** | 3,747 | **56%** | **3%** | 45% |

Even on the rules that depend on the player's own moves, the reasoning that won is mostly the phrase superstition seen in the baselines. That superstition wins because the opponent's lines repeat every 4 rounds, and the copy-style opponents turn a consistent superstition into a self-confirming pattern. Outcome filtering selects reasoning that *won*, not reasoning that *understood*. SFT on this data will most likely teach these superstitions more firmly. Against copycat, phrase-based play is capped at about 75% of rounds (period 4 against a rule that needs period 3; see the copycat baseline report), so a gain there would not show discovery.

Options before spending GPU time on SFT:
1. **Run SFT on this data anyway, as a pilot.** It answers "what does outcome-filtered self-imitation teach here?" in about 4–5 h; the expected answer is "stronger superstitions".
2. **Remove the spurious cue and regenerate.** Make the opponent's lines uninformative (silent, or not tied to a 4-round cycle) for generation and evaluation, then rerun. Your plan treats speech as a separate observation-distribution ablation, so this would be a new condition with its own baseline.
3. **Supervise opponent prediction (variant B).** Train on what the opponent actually did next, a signal that doesn't depend on winning for the wrong reason.
4. **Run the control the review suggests.** SFT on equally sized, randomly selected data tests whether the success filter adds anything.

## Review fixes (after `8cfbf57`)

| # | Issue | Fix |
|---|---|---|
| 1 | The exported adapter dropped the shared key/value update in the 10 full-attention layers, so the served model would miss part of the training. | The `k_proj` adapter is also written as `v_proj` for those layers, found from the model's config by vLLM's own rule. A CPU test with vLLM's loader and packing checks the served values numerically. At serving time, `check_adapter.py` compares base and adapter through the live server. |
| 2 | Retrying SFT, or `--redo_from sft`, failed with the real trainer, which refuses a used run folder. Only the final checkpoint was saved. | Retry: `--resume` from the latest checkpoint, or `--clean` when the previous attempt saved none. Redo: the old SFT folder is moved aside and a fresh one used. Intermediate checkpoints are saved. The stand-in trainer now uses prime-rl's own run-folder validator, so tests catch this. |
| 3 | The 35% cap applied to the target, not to the dataset actually selected, so one rule could make up 100% of it. | The cap is enforced on the final dataset, by an allocation computed before selection. An impossible cap fails with the per-rule yields. |
| 4 | Trimming to the target could drop whole rules alphabetically, while the report claimed otherwise. | The budget is allocated across rules up front, and reported counts come from the written files. |
| Design | Strict and relaxed selection should stay distinguishable; count unique games; evaluate adaptation over time. | Strict-only by default; `meta.strict` on every example; unique games per rule; random-play pass rates beside the observed rates; a 20-round evaluation with round-by-round win rates. Settings are now kept per stage, so a finished stage's settings are protected while later ones can change. |

**Still open, as the review notes:**
- A gain on held-out copycat alone doesn't establish general pattern discovery. More held-out rules and action-intervention tests would strengthen that claim.
- An SFT control on equally sized, randomly selected data would test whether the success filter adds value.

## Caveats

- **The data comes from the base model's successes, and most are easy rules** (see "Data from the first run"). The per-rule cap limits the imbalance, but the rules that depend on the player's own moves contribute only 759 examples.
- **One round of expert iteration.** If the SFT model improves on the training rules, a second round (generate with the adapter, select, train again) is the natural next step.
- **The held-out copycat result is the one that speaks to discovery.** In-distribution gains on fresh seeds of the training rules are weaker evidence.
