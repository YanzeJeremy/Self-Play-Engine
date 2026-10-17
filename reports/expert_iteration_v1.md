# Expert iteration v1: self-play SFT on hidden-rule RPS

*Branch `yanze/expert-iteration-v1`. This is the pipeline only; there are no results yet.*

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
| 3 | sft | LoRA SFT with prime-rl (`make_sft_config.py`): rank 16, alpha 32, lr 1e-4 cosine, 1 epoch, seq_len 8,192, 16 packed rows per step, both GPUs. Gemma-4 needs the `gemma4` renderer and the fused LM head off. The frozen base and the LoRA weights are held in bfloat16, as in prime-rl's LoRA example. |
| 4 | export | `export_lora_adapter.py`: reads only the LoRA tensors from the DCP checkpoint and writes the adapter exactly as prime-rl's own adapter broadcast does (`adapter_model.safetensors` + `adapter_config.json`). prime-rl has no exporter for SFT checkpoints. |
| 5 | evaluate | One vLLM server with the base model plus the adapter. Greedy (T=0), 10 rounds, both models. Held-out copycat: 100 games, seeds 0–99; this is the same condition as the earlier B0 primary baseline (43.9% of rounds 2–10 won). Each training rule: 40 games on fresh seeds 50,000–50,039, never used for training. The adapter is named and evaluated under a hash of its weights. |
| 6 | summarize | `summarize.py` pairs base and SFT game by game on seeds, per rule, with the strict checks (complete, same condition except the model). Writes `summary.md` and `summary.json`. |

### Data selection: what counts as "good"

- **Which rounds qualify.** A round from the 2nd on qualifies when the player **won it and won ≥ 80% of that round and the 4 after it**, with at least 3 of those rounds played. Round 1 is excluded because it's chance.
- **The examples.** Both calls that decided a qualifying round become examples: the **thinking turn** before the move (its notes drive the move) and the **move** itself (its final, accepted attempt). Each is stored as the exact prompt plus the model's reply, ending in `[STOP]`. Talk turns are not trained.
- **Always dropped:**
  - calls without a readable answer;
  - moves that weren't accepted;
  - replies cut at the token limit;
  - examples over seq_len − 256 tokens;
  - duplicates.
- **Caps:**
  - 16 examples per game;
  - **35% of the target per rule**, so easy rules can't dominate;
  - within a rule, higher reward-to-go first, then earlier rounds.
- **Target: 5,000–10,000 examples.**
  - If the 80% threshold yields fewer than 5,000, the threshold is relaxed to 60%. If that's still short, the run goes ahead with what there is and the shortfall is reported.
  - The run fails below 500 examples.
- **Validation split:** 5% of games, chosen by game.
- **Outputs:** `data/dataset/` (what SFT reads), `data/meta/` (each example's rule, seed, round, call and tokens) and `data/stats.json` (yields per rule, the threshold used, and what was dropped).

## Running

In the 2×H200 allocation, from the repo root. **Stop your own `vllm serve` first**: the pipeline starts and stops its own servers, and refuses to start if the GPUs or port 8000 are busy. Use tmux.

```bash
bash scripts/run_expert_iteration.sh --smoke true    # tiny run through every stage first (~30-60 min)
bash scripts/run_expert_iteration.sh                 # the full experiment
```

- **Where outputs go:** everything lands in `$storage_dir/expert_iteration/<name>/` (default name `ei-v1`; the smoke run is `ei-v1-smoke`), in `configs/`, `runs/`, `data/`, `sft/`, `adapter/`, `logs/` and `stages/`. The result is `summary.md`.
- **Resuming:** run the same command again. Finished stages are skipped, and unfinished evaluation runs resume.
- **Rerunning part:** `--redo_from <stage>` reruns that stage and everything after it.
- **Changing settings** for an existing `--name` is refused; use a new name.
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
| The export | Exact values and the prime-rl adapter key format; refuses wrong ranks and incomplete adapters. |
| The orchestrator | Runs end to end with stand-ins for vLLM, eval.sh and sft. An identical rerun skips every stage. Changed settings are refused. `--redo_from sft` gets fresh evaluations under a new adapter name. |

**First exercised by the smoke run on GPUs:**
- vLLM data-parallel serving, and LoRA serving of gemma-4;
- prime-rl's 4-day-old gemma-4 trainer with LoRA, and its memory use;
- the real checkpoint's key names: the export stops with the actual names if they differ;
- whether the 124 GB of CPU RAM in job 6490009 is enough while the model loads. If SFT is killed for memory, request more (`--mem`).

## Caveats

- **The data comes from the base model's successes, and most may be easy rules.** `repeat_self` and some `cycle` games are winnable by modeling the opponent's own sequence. The rules that depend on the player's moves may yield little. The per-rule cap limits this imbalance, and `stats.json` shows the yields. If they're very lopsided, that is itself a result: the base model rarely solves those rules even by chance.
- **One round of expert iteration.** If the SFT model improves on the training rules, a second round (generate with the adapter, select, train again) is the natural next step.
- **The held-out copycat result is the one that speaks to discovery.** In-distribution gains on fresh seeds of the training rules are weaker evidence.
