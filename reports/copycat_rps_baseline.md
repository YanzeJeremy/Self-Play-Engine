# Copycat RPS: engine overview and gemma-4-31b-it baseline

*Yanze Wang, 2026-10-16. Repo code unmodified at `25910d5`; this branch adds eval configs, two analysis scripts and this report.*

## TL;DR

- The engine runs end to end on a real model through the verifiers path: vLLM → `scripts/eval.sh` → `SelfPlayEnv` → `Runner` → `CopycatRPSGame`. gemma-4-31b-it produced 0 invalid answers and 0 truncated replies in 3,500 calls.
- **gemma-4-31b-it never discovers the copycat rule.** No call in 50 games states "the opponent plays my previous move".
- **10-round games:** it wins about 44–45% of rounds 2–10 (random 33%, optimal 100%), with no difference between greedy and sampled decoding.
- **100-round games:** it climbs from 40% to about 70% of rounds won. The gain comes from **self-fulfilling wrong theories**, not from discovering the rule:
  - "the opponent cycles S → R → P", which against a copycat *is* the optimal rotation;
  - "after phrase X it plays Y", which plateaus near 75%.
- These results point at likely reasons RL doesn't improve the model (section 6). They are hypotheses until an RL run tests them.

## 1. The task

Copycat rock-paper-scissors is a test env for RL / self-play training of LLM game agents. The opponent plays whatever the agent played in the previous round. The optimal policy is therefore to beat your own last move: rock → paper → scissors → rock …

RL training on this env currently does not improve the model. The goal is a model that solves the game, by RL, SFT or any other method.

## 2. How the engine works

| Layer | Where | Role |
|---|---|---|
| Game | `self_play/game.py` | The contract every game implements: state (JSON only, so any game can be checkpointed), legality, transitions, observations, prompt text, rewards (game outcome and per-step). |
| Runner | `self_play/runner.py` | Drives one game. It schedules calls by dialogue mode, makes the model calls, handles talk, notes and invalid answers (retry, then fall back to the first legal move), keeps the record, and checkpoints after every call. |
| Prompting | `self_play/prompting.py` | Every call is a fresh context: system prompt plus one user message (rules, history, the player's notes, instruction). Reply format: `<thinking>…</thinking><answer>…</answer>[STOP]`. |
| verifiers env | `self_play_verifiers/` | `SelfPlayEnv` maps players to seats `seat0`–`seat5`. `StatelessHarness` sends each call as its own prompt. The taskset is infinite, one seed per game. Rewards: game outcome (`game`) or mean round result (`moves`), optional forfeit for invalid answers. |
| Training / eval | `prime-rl/` (submodule) | RL (GRPO by default), SFT, and `vf-eval` via `scripts/eval.sh`. |

There are three kinds of call:
- **think:** rewrite the private notes, the only memory between calls;
- **talk:** say something to the table;
- **answer:** the move.

The five dialogue modes (action, response window, discussion, simultaneous, thinking) decide which calls a phase makes.

## 3. The copycat RPS game (`self_play/games/copycat_rps/`)

- **Players:** one LLM player against a **rule-based** opponent. The opponent is not an LLM and receives no prompt.
- **Opponent's move:** round 1 is drawn from the seed; from round 2 on it is the player's previous move.
- **Opponent's lines:** one line per round, in a fixed 4-round cycle: a boast, a comment, a chat line, then "I will play X", which is true on a seeded coin flip. The cycle also rotates the dialogue modes.
- **Calls:** the player has a think call every round, a talk call in odd rounds and an answer call every round. A 10-round game is 25 calls, a 100-round game 250.
- **Rewards:** game reward is 1 for winning more rounds, 0.5 for a tie, 0 for losing. Per round, a win is 1, a draw 0.5, a loss 0.

## 4. Setup notes for this cluster (USC CARC)

1. **Python headers.** Build `prime-rl/.venv` on a uv-managed Python. On the system `/usr/bin/python3.12` there is no `Python.h`, and vLLM's Triton fails to compile `cuda_utils.c`:
   ```bash
   UV_PYTHON_INSTALL_DIR=<dir> uv python install 3.12
   cd prime-rl && uv sync --all-extras --python <that python>
   ```
2. **nvcc.** vLLM needs nvcc because FlashInfer JIT-compiles kernels at startup. Load CUDA before activating the venv:
   ```bash
   module load ver/2607 gcc/15.3.0 cuda/13.0.3      # matches torch cu130
   source prime-rl/.venv/bin/activate
   vllm serve google/gemma-4-31b-it --port 8000 --max-model-len 16384
   ```
   One H200 holds the model. The default 262k context is unnecessary.
3. **Stale test.** `pytest tests/` gives 58 passed, 1 failed. The failure is `test_env.py::test_a_game_runs_through_seat0_and_fills_its_trace`, which predates the `game_outcome` metric added in `13cd059`.

## 5. Results

Configs are in `eval_configs/` (`baseline_*.toml`, `long100_*.toml`). They are copies of `smoke.toml` with only the game count, concurrency, temperature, rounds and `max_tokens` changed.

**Sampling:**
- "Sampled" means temperature 1.0, with gemma's served defaults `top_k=64`, `top_p=0.95`.
- "Greedy" means temperature 0.
- RL rollouts sample, so the sampled run shows what RL would have to work with.

| Run | Games | Games W/T/L | Rounds 2+ won | Beat-own-last moves (rounds 2+) | Copy/mirror mentions |
|---|---|---|---|---|---|
| Scripted: optimal policy | 100 × 10 rounds | 100/0/0 | 100% | 100% | – |
| Scripted: uniform random | 100 × 10 rounds | – | ≈ 33% (0.53 mean round result) | ≈ 33% | – |
| Scripted: always rock (= invalid fallback) | 100 × 10 rounds | – | 0% won, all draws | 0% | – |
| gemma, 10 rounds, greedy | 20 | 11/0/9 | 43.9% | 43.9% | 4 of 500 calls |
| gemma, 10 rounds, sampled | 20 | 11/0/9 | 45.0% | 45.0% | 4 of 500 calls |
| gemma, 100 rounds, greedy | 10 | 10/0/0 | 63.8% | 63.8% | 6 of 2,500 calls |

Against this opponent, winning a round and beating your own last move are the same thing, so those two columns always match.

**100-round trend** (greedy, % of rounds won per 10-round window):

| Rounds | 1–10 | 11–20 | 21–30 | 31–40 | 41–50 | 51–60 | 61–70 | 71–80 | 81–90 | 91–100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Won | 40% | 54% | 61% | 70% | 69% | 70% | 70% | 76% | 64% | 61% |

Per-game final scores range from 40W/39L (seed 1, never settles) to 100W/0L (seed 4).

### What the model does

- **It models only the opponent's column of the history.** It looks for move cycles, links each phrase to a move, and weighs the claims. It essentially never compares the opponent's move with its own previous move, although every history line puts them side by side.
- **The copy/mirror mentions are not the rule.** On reading, almost all are about phrase sequences or dialogue. The closest is greedy seed 7 at round 9 of 10: "the opponent played Paper, which was the same move I made in Round 4 (Mirroring)". It raised this as one option among several, too late to matter.
- **Successes come from self-fulfilling theories:**
  - **Cycle illusion** (sampled 10-round seed 10 at 10/10; 100-round seed 4 at 100/100). "Round % 3 == 1: Scissors, Round % 3 == 2: Rock, Round % 3 == 0: Paper… opponent will play Paper. To win, I must play Scissors." Beating an imagined fixed 3-cycle *is* the rotation, so the copycat keeps the illusion true forever.
  - **Phrase superstition** (most 100-round games). "'The player is doing whatever they want'… the opponent has played Paper for the last 7 consecutive times this phrase was used." Playing a fixed move per phrase makes the agent's play repeat every 4 rounds. A period-4 sequence can beat its own previous move in at most 3 of every 4 rounds, which is consistent with the observed plateau around 70–76%. This cap is analysis, not a measurement.
- **Sampling adds almost no exploration.** Greedy and sampled are indistinguishable at 10 rounds.
- **Replies are long:** mean think reply about 1,950 characters at 10 rounds and 3,300 at 100 rounds (max 8,765). Talk replies are about 500 characters, answers about 550.

## 6. Implications for RL (hypotheses, untested)

1. **Credit assignment.**
   - prime-rl turns each stateless call into its own training sample (one branch per call). GRPO then gives every call in a game the **same game-level advantage**, about 25 calls in a 10-round game.
   - A move's own result is diluted by the other moves and by long think and talk replies.
   - The env already computes per-step rewards (`trace.info["step_rewards"]`), but no prime-rl algorithm uses them. `checklist.md` item 5 plans one.
2. **Exploration.** The correct hypothesis is almost never sampled, so there is little to reinforce.
3. **A cheaper attractor.** The cycle illusion is fully rewarded against this opponent. RL could converge on it without the model learning the real rule. An eval that tells them apart would need, e.g., an opponent opening or a mid-game perturbation that breaks the illusion.
4. **Cost.** Episodes of 25–250 calls with long replies make every RL step expensive.
5. **No RL config in the repo.** The setup that "isn't working" can't be reproduced exactly yet.

## 7. Candidate next steps

1. Reproduce the failure: stock GRPO on the unmodified env with a model trainable on 2 × H200 (Qwen3-4B or 8B).
2. Per-call credit: a prime-rl algorithm that credits each move with its own round's result, baselined against the same round of other games in the group.
3. SFT warm-start on rotation games generated by the engine.
4. An eval that separates understanding the copycat from the cycle illusion.

## Reproducing

```bash
# Reference scores (CPU)
python scripts/scripted_copycat_baseline.py --seeds 100 --rounds 10

# LLM runs (vLLM server up; see section 4)
bash scripts/eval.sh --config eval_configs/baseline_greedy.toml  --name base-gemma31b-greedy  --base_url http://<node>:8000/v1
bash scripts/eval.sh --config eval_configs/baseline_sampled.toml --name base-gemma31b-sampled --base_url http://<node>:8000/v1
bash scripts/eval.sh --config eval_configs/long100_greedy.toml   --name long100-gemma31b-greedy --base_url http://<node>:8000/v1

# Analysis
python scripts/analyze_copycat_games.py --traces $storage_dir/eval_runs/<name>/traces.jsonl --show 2
```

The run outputs (`$storage_dir/eval_runs/…`) are not committed. They are on CARC under `/project2/jonmay_1426/yanzewan/Self-Play-Engine/self_play_storage/eval_runs/`.
