# Prediction protocol v1: implementation notes (M1) and baseline tooling (M2)

*Implements sections 2–7 and 13 (M1, M2 tooling) of [strategic_reasoning_training_plan.md](strategic_reasoning_training_plan.md). Branch `yanze/prediction-protocol-v1`, based on `54cf5ed`.*

Training (M3+) is not part of this branch. The plan gates it on validated B0/B1 baselines, which still need to be run (section 4).

## 1. What changed

| File | Change |
|---|---|
| `self_play/prediction.py` (new) | Modes (`off`, `opponent_action_v1`), `PredictionSpec`, the parser and validator, scoring helpers (Brier, log loss, argmax), constants (sum tolerance `1e-3`, log-loss floor `1e-8`, schema version 1). |
| `self_play/prompting.py` | `PREDICTION_SYSTEM_PROMPT` for move calls in prediction mode; `build_messages(system_prompt=...)`, default unchanged; `prediction_format(labels)`, which describes the schema without example numbers; `prompt_templates()` for hashing. |
| `self_play/game.py` | Optional hooks with no-op defaults: `prediction_targets`, `get_prediction_spec`, `resolve_prediction_target`, `render_prediction_request`, `prompt_templates`. |
| `self_play/games/copycat_rps/` | Target `opponent_action`. Labels are rock/paper/scissors; `decision_id` is the round index from the observation; the observed move is read from completed history. The plan's exact instruction text, which names no strategy. |
| `self_play/runner.py` | `prediction_mode`, a `call_id` per call (stable through replay), forecasts parsed into move-call entries, `prediction_decisions()`, and a run fingerprint saved in and checked against snapshots. |
| `self_play/checkpoint.py` | `run_fingerprint()` and `prompt_template_hash()`. |
| `self_play_verifiers/env.py` | Config `prediction_mode` (default `off`) and `allow_legacy_checkpoints` (default false). Fingerprints from each seat's resolved model, client type, renderer and sampling. `trace.info["predictions"]` plus `prediction_*` metrics. Rewards are untouched. |
| `scripts/eval.sh` | Writes `manifests/<time>.json` via `scripts/write_run_manifest.py` before every vf-eval invocation. |
| `scripts/write_run_manifest.py` (new) | Commits and dirty flags (root, prime-rl, verifiers), the eval config, protocol constants and template hash, the server's `/v1/models` and `/version`, and from the HF cache the model revision, generation-config defaults and chat-template hash. |
| `scripts/analyze_copycat_games.py` | The legacy summaries are kept. Added: forecast coverage with every status, accuracy, Brier, log loss (with the number of zero-probability observations), coverage-adjusted Brier, action consistency, adaptation windows, output costs, per-game rows, `--pair` (paired by seed, bootstrap 95% CI over games) and `--json_out`. |
| `self_play_verifiers/env.py`, resume fix | Seat sampling fields saved as `null` are treated as unset when a config is reloaded (see section 6). |
| `scripts/run_prediction_baseline.sh` (new) | Runs a stage (`smoke` / `primary` / `long`) of B0/B1 conditions, then the paired analysis. |
| `eval_configs/prediction_*.toml` (new) | Smoke (3 games × 6 rounds); primary B0/B1 × t0/t1 (100 × 10, 2,048 tokens); long B0/B1 × t0/t1 (20 × 100, 4,096 tokens). `top_p = 0.95` and `top_k = 64` are pinned in every config. |

## 2. Protocol behavior

- **`off` is byte-identical to the old engine.** `tests/fixtures/legacy_prompts_copycat.json` holds every prompt of a scripted 8-round game (all four modes, talk, a retry, a fallback, an invalid thinking turn), recorded from `54cf5ed` before any change. Off mode reproduces it exactly.
- **Only move calls change in `opponent_action_v1`.**
  - They get `PREDICTION_SYSTEM_PROMPT`, and the user message adds the forecast instruction after the move request.
  - Thinking and talk calls are byte-identical to `off`.
  - The number and schedule of calls are unchanged.
- **The reply format for move calls:**
  ```
  <thinking>…</thinking>
  <prediction>{"rock": …, "paper": …, "scissors": …}</prediction>
  <answer>…</answer>
  [STOP]
  ```
- **Forecasts are measured only.** They never affect move parsing, retries, fallbacks or rewards. They never appear in later prompts (notes and history are unchanged).
- **Parser statuses:**
  - `valid`, `missing`, `unclosed_block`, `duplicate_block`, `misplaced` (not before the top-level `<answer>`);
  - `invalid_json` (also NaN/Infinity, which are not JSON), `duplicate_keys`, `invalid_labels`, `invalid_values` (booleans and non-numbers included), `invalid_sum`.
  - A block inside `<thinking>`, or after `[STOP]`, is not a forecast.
  - Sums within `1e-3` of 1 are normalized, with the raw values and the `normalized` flag kept; anything further off is invalid and never repaired.
- **Decision records** (`trace.info["predictions"]["decisions"]`) are one per executed decision:
  - **Fields:** `call_id`, `attempt_call_ids`, `decision_id`, `step_id`, raw, parsed and normalized forecast, status, `observed_target`, `executed_action`, `action_source`, `round_reward`.
  - **Which forecast is scored:** after a retry, the final attempt's; after a fallback, the last attempt's, with the action marked `fallback`. Forced actions have no forecast and are excluded from coverage.
- **Fingerprints.**
  - **Contents:** protocol, a prompt-template hash (engine prompt texts plus the game's module and prompt sources), and each seat's model, client type, renderer and sampling (`max_tokens` included). Also the runner settings, retries and talk window. Endpoint URLs are left out, so a run can resume against the same model served elsewhere.
  - **Partial and finished checkpoints** are reused only under an identical fingerprint.
  - **Older checkpoints without one** are refused unless `allow_legacy_checkpoints = true`.

### Choices the plan left open

| Question | Choice |
|---|---|
| Unclosed or misplaced blocks | Separate statuses `unclosed_block` and `misplaced`. A forecast with no top-level `<answer>` at all is `misplaced`. |
| Duplicate JSON keys | Separate status `duplicate_keys`. |
| NaN / Infinity | `invalid_json`, since they are not valid JSON. |
| `call_id` format | `call-<n>`, counting the game's calls from 0. Episode identity comes from the trace. |
| Argmax ties | Go to the earliest label in rock, paper, scissors order. |
| Action consistency | Computed only for moves the model chose (`action_source = model`), not fallbacks. |
| What the fingerprint includes | Resolved sampling (so `max_tokens` changes count), but not the endpoint URL. The prompt-template hash covers the game's whole module source, which is conservative: any change to the game counts as a new condition. |

## 3. Tests

`pytest tests/`: **107 passed, 1 failed.** The failure is pre-existing and not caused by this branch: `tests/test_env.py::test_a_game_runs_through_seat0_and_fills_its_trace` expects no `game_outcome` metric, which commit `13cd059` added.

| Test file | Covers |
|---|---|
| `tests/test_prediction.py` | Parser boundary cases and scores on known distributions (uniform Brier 2/3, log-loss floor, argmax ties). |
| `tests/test_prediction_protocol.py` | Off-mode prompts equal the recorded legacy prompts. Prediction mode changes only move calls. The request names no strategy. Forecasts never reach later prompts. The round-1 forecast prompt doesn't depend on the hidden opening. Forecast validity never changes the game. Forecast-to-decision association with thinking, retries, fallbacks and two-player simultaneous play. Call ids. Unsupported games and modes are refused. Resume after a crash at several points gives an identical record and decisions. Finished checkpoints are refused under changed protocol, model, sampling or retries. Legacy checkpoints need permission. The template hash follows prompt texts. |
| `tests/test_env.py` (added tests) | `trace.info["predictions"]` and metrics written and rewards unchanged. Metric values on known forecasts. Switching condition on a reused checkpoint directory is refused. A reloaded (resumed) config keeps the run's sampling. |

## 4. Running the M2 baseline

From the repo root on the eval node, with the vLLM server up. See [copycat_rps_baseline.md](copycat_rps_baseline.md) for the CARC setup.

```bash
module load ver/2607 gcc/15.3.0 cuda/13.0.3 && source prime-rl/.venv/bin/activate
bash scripts/run_prediction_baseline.sh --stage smoke   --base_url http://<node>:8000/v1
bash scripts/run_prediction_baseline.sh --stage primary --base_url http://<node>:8000/v1
# measure primary costs (input/output tokens, wall time in the analysis) before:
bash scripts/run_prediction_baseline.sh --stage long    --base_url http://<node>:8000/v1
```

- **Run names:** `rps-predv1-<b0|b1>-gemma31b-<t0|t1>-r<rounds>-n<games>`, each with a fresh checkpoint folder.
- **Rerunning the script** skips finished runs and resumes unfinished ones.
- **Analysis:** paired B1 − B0 results go to `$storage_dir/eval_runs/predv1-<stage>-…-analysis.json`, and the console shows the summary.
- **Any single run:**
  ```bash
  python scripts/analyze_copycat_games.py --traces <traces.jsonl>
  ```

Budget, from the plan: the primary stage is 4 conditions × 100 games × 25 calls = 10,000 calls; the long stage is 4 × 20 × 250 = 20,000 calls. At the measured legacy rate (about 200 s per 10-round game at 10 in parallel), the primary stage takes roughly 4 × 100 × 200 s / 10 ≈ 2.2 h. Long games scale with prompt length, so price them from the primary measurements.

## 5. Smoke run (2026-10-16, gemma-4-31b-it, 3 games × 6 rounds, greedy)

Run with `scripts/run_prediction_baseline.sh --stage smoke` against the live server. It checks the plumbing, not a result.

- **B0 (off):** 3 games, 45 calls, every request sent with the pinned sampling. 0 invalid actions, 0 truncations.
- **B1 (opponent_action_v1):**
  - The run was interrupted after 60 s and resumed. Each game continued from its checkpoint and made only its remaining calls (8, 8 and 9 of 15). All 25 resumed calls carried the pinned sampling.
  - 18 of 18 forecasts were valid (coverage 1.0), with 0 invalid actions and 0 truncations.
  - Brier 0.868, log loss 1.374, accuracy 1/18. These come from 18 forecasts and are not a finding.
- **The first resume attempt failed safely.** It was refused by the fingerprint check (`differs in ['agents']`), which exposed the bug in section 6. After the fix, the same checkpoints resumed.

## 6. Erratum: resumed eval runs dropped their sampling settings

`scripts/eval.sh --resume true` reruns vf-eval on the saved `configs/resolved/eval.json`.

**The bug:**
- In that file each seat's sampling lists every field, and the unset ones are written as `null`.
- When reloaded, those nulls count as set fields. They then override the run's sampling when the seat is resolved.
- As a result, **every request of a resumed run was sent with only `stop`**. Temperature, top_p and max_tokens were dropped, and the vLLM server applied the model's `generation_config.json` defaults: temperature 1.0, top_k 64, top_p 0.95, no completion limit.

**Effect on [copycat_rps_baseline.md](copycat_rps_baseline.md):**
- Both 10-round runs there were completed with `--resume true`, after a first attempt failed with no server.
- The traces show all 500 calls of **both** "greedy" and "sampled" sent with `{"stop": ["[STOP]"]}`. Both runs are therefore the same condition, sampled at the server defaults, and **the report's "greedy vs sampled are indistinguishable" finding is an artifact**. Its 10-round "greedy" numbers are sampled-decoding numbers.
- The 100-round greedy run was not resumed and did send `temperature 0.0`; its results stand.
- Following the plan, that report is kept unchanged as the historical record, with this note as the correction.

**The fix** (`SelfPlayEnvConfig.unset_null_seat_sampling`): when a config is reloaded, `null` seat sampling fields are unset again. The test `test_a_reloaded_config_keeps_the_runs_sampling_for_every_seat` covers it.

**For future runs:** the run fingerprint now catches such drift, because a resume under different sampling is refused. The manifests and the per-call `sampling` recorded in traces make it auditable.

## 7. Not done here

- Seed manifests or disjoint seed ranges for the taskset. Every eval uses seeds `0..N-1`, so B0 and B1 are matched; the split for training data comes in M3.
- A prediction dataset, warm-up SFT, RL arms or adaptation suites (M3–M6).
- The M2 report itself, which needs the runs.
