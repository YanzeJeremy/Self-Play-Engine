#!/usr/bin/env bash
# Matched B0/B1 baseline for prediction protocol v1 (reports/strategic_reasoning_training_plan.md, section 6):
# the same model and seeds with prediction_mode "off" (B0) and "opponent_action_v1" (B1), then the paired analysis.
#
#   bash scripts/run_prediction_baseline.sh --stage smoke   --base_url http://<node>:8000/v1
#   bash scripts/run_prediction_baseline.sh --stage primary --base_url http://<node>:8000/v1   # 2 modes x t0/t1, 100 games x 10 rounds
#   bash scripts/run_prediction_baseline.sh --stage long    --base_url http://<node>:8000/v1   # 2 modes x t0/t1, 20 games x 100 rounds
#
# Stages:
#   smoke    B0 and B1, 3 games x 6 rounds, greedy. B1 is interrupted after --interrupt_after seconds and then
#            resumed, to exercise checkpoint resume with fingerprints through the real stack.
#   primary  eval_configs/prediction_{b0,b1}_{t0,t1}_r10.toml
#   long     eval_configs/prediction_{b0,b1}_{t0,t1}_r100.toml (measure smoke/primary costs first)
# Runs are named rps-predv1-<b0|b1>-<model_tag>-<t0|t1>-r<rounds>-n<games> (smoke: ...-smoke), each with its own
# checkpoint folder. An existing run is first checked to be the requested condition (scripts/run_identity.py
# check; a mismatch stops the script: use another --model_tag or move the old run away). Then a complete run is
# skipped and an unfinished one resumed, with --base_url moving it to the current server if needed. After each
# condition the run must hold every configured game, finished (run_identity.py complete), or the script stops
# with exit 1 before any analysis. The paired analysis refuses incomplete or mismatched pairs. It goes to
# $storage_dir/eval_runs/predv1-<stage>-<model_tag>-...-analysis.json.

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["base_url"]="none"          # the vLLM server, e.g. http://c04-06:8000/v1 (none: each config's localhost)
ARGS["model_tag"]="gemma31b"     # short model name used in run names
ARGS["interrupt_after"]="90"     # smoke stage: seconds before B1 is interrupted

REQUIRED_ARGS=("stage")
# --- Argument parsing (copy verbatim) ---
ALLOWED_FLAGS=("${REQUIRED_ARGS[@]}" "${!ARGS[@]}")
USAGE_STR="Usage: $0"
for req in "${REQUIRED_ARGS[@]}"; do
    USAGE_STR+=" --$req <value>"
done
for opt in "${!ARGS[@]}"; do
    if [[ ! " ${REQUIRED_ARGS[*]} " =~ " ${opt} " ]]; then
        if [[ -z "${ARGS[$opt]}" ]]; then
            echo "DEFAULT VALUE OF KEY \"$opt\" CANNOT BE BLANK"; exit 1
        fi
        USAGE_STR+=" [--$opt <value> (default: ${ARGS[$opt]})]"
    fi
done
function usage() { echo "$USAGE_STR"; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*)
            FLAG=${1#--}
            VALID=false
            for allowed in "${ALLOWED_FLAGS[@]}"; do
                if [[ "$FLAG" == "$allowed" ]]; then VALID=true; break; fi
            done
            if [ "$VALID" = false ]; then echo "Error: Unknown flag --$FLAG"; usage; fi
            ARGS["$FLAG"]="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

for req in "${REQUIRED_ARGS[@]}"; do
    if [[ -z "${ARGS[$req]}" ]]; then echo "Error: --$req is required."; FAILED=true; fi
done
if [ "$FAILED" = true ]; then usage; fi
# --- End argument parsing ---

# Print active variables
echo "Script: $0 Active variables:"
for key in "${!ARGS[@]}"; do
    echo "  -$key = ${ARGS[$key]}"
done

# Script logic below:
stage="${ARGS["stage"]}"
tag="${ARGS["model_tag"]}"
runs_dir="${EVAL_RUNS_DIR:-$storage_dir/eval_runs}"     # EVAL_RUNS_DIR, EVAL_SCRIPT and PREDV1_CONFIG_DIR exist for tests
eval_script="${EVAL_SCRIPT:-scripts/eval.sh}"
configs="${PREDV1_CONFIG_DIR:-eval_configs}"

fail() { echo "Error: $*"; exit 1; }

is_complete() { python scripts/run_identity.py complete --run_dir "$runs_dir/$1"; }

run_condition() {  # run_condition <config> <run name>: leaves the run complete, or stops the script
    local config="$1" name="$2" run_dir="$runs_dir/$2"
    if [[ -f "$run_dir/configs/resolved/eval.json" ]]; then
        python scripts/run_identity.py check --run_dir "$run_dir" --config "$config" || fail "$name exists but is another condition"
        if is_complete "$name" > /dev/null; then echo "== $name: complete, skipping"; return 0; fi
        echo "== $name: resuming"
        bash "$eval_script" --name "$name" --resume true --base_url "${ARGS["base_url"]}" || fail "evaluation of $name failed (exit $?)"
    else
        echo "== $name: starting"
        bash "$eval_script" --config "$config" --name "$name" --base_url "${ARGS["base_url"]}" || fail "evaluation of $name failed (exit $?)"
    fi
    is_complete "$name" || fail "$name is incomplete after evaluation; rerun this script to resume it"
}

pairs=()
case "$stage" in
    smoke)
        b0="rps-predv1-b0-$tag-t0-r6-n3-smoke"; b1="rps-predv1-b1-$tag-t0-r6-n3-smoke"
        run_condition "$configs/prediction_smoke_b0.toml" "$b0"
        if [[ ! -d "$runs_dir/$b1" ]]; then
            # deliberate interruption, to exercise resume: the only evaluation allowed to stop early
            echo "== $b1: starting, to be interrupted after ${ARGS["interrupt_after"]}s"
            timeout --signal=INT "${ARGS["interrupt_after"]}" bash "$eval_script" --config "$configs/prediction_smoke_b1.toml" --name "$b1" --base_url "${ARGS["base_url"]}"
            echo "== $b1: interrupted (exit $?); checkpoints: $(ls "$runs_dir/$b1/checkpoints"/*.json 2>/dev/null | wc -l)"
        fi
        run_condition "$configs/prediction_smoke_b1.toml" "$b1"
        pairs+=("$b0 $b1")
        ;;
    primary|long)
        if [[ "$stage" == "primary" ]]; then rounds=10; games=100; else rounds=100; games=20; fi
        for t in t0 t1; do
            b0="rps-predv1-b0-$tag-$t-r$rounds-n$games"; b1="rps-predv1-b1-$tag-$t-r$rounds-n$games"
            run_condition "$configs/prediction_b0_${t}_r${rounds}.toml" "$b0"
            run_condition "$configs/prediction_b1_${t}_r${rounds}.toml" "$b1"
            pairs+=("$b0 $b1")
        done
        ;;
    *) fail "--stage must be smoke, primary or long" ;;
esac

for pair in "${pairs[@]}"; do
    read -r b0 b1 <<< "$pair"
    python scripts/analyze_copycat_games.py --pair "$runs_dir/$b0/traces.jsonl" "$runs_dir/$b1/traces.jsonl" \
        --json_out "$runs_dir/predv1-$stage-$tag-${b0#rps-predv1-b0-$tag-}-analysis.json" || fail "paired analysis of $b0 / $b1 refused"
done
echo "== all conditions complete; paired analyses written"
