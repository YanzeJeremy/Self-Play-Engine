#!/usr/bin/env bash
# Native evaluation: vf-eval on a config in eval_configs/, written under $storage_dir.
#
#   bash scripts/eval.sh --config eval_configs/smoke.toml --name smoke-001
#   bash scripts/eval.sh --config eval_configs/smoke.toml --name smoke-001 --base_url http://<node>:8000/v1
#   bash scripts/eval.sh --name smoke-001 --resume true      # continue an interrupted run
#   bash scripts/eval.sh --name smoke-001 --resume true --base_url http://<new node>:8000/v1   # ... against a moved server
#
# The run lands in $storage_dir/eval_runs/<name>/ (vf-eval's traces.jsonl, configs/, logs/, and
# manifests/<time>.json from scripts/write_run_manifest.py, one per invocation)
# with mid-game checkpoints in its checkpoints/ folder. Checkpointing needs one rollout per
# task, so -r 1 is always passed. A resume reuses the run's saved config, checkpoint folder
# included; vf-eval reruns only unfinished games, and each continues from its checkpoint.

source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["config"]="none"
ARGS["resume"]="false"
ARGS["base_url"]="none"   # overrides the config's [client] base_url, e.g. a vLLM server on another node

REQUIRED_ARGS=("name")

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
name="${ARGS["name"]}"
if [[ ! "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "Error: --name may use only letters, digits, '.', '_' and '-'"; exit 1
fi
runs_dir="${EVAL_RUNS_DIR:-$storage_dir/eval_runs}"   # EVAL_RUNS_DIR: tests only
run_dir="$runs_dir/$name"

if [[ "${ARGS["resume"]}" == "true" || "${ARGS["resume"]}" == "yes" || "${ARGS["resume"]}" == "y" ]]; then
    saved_config="$run_dir/configs/resolved/eval.json"
    if [[ ! -f "$saved_config" ]]; then echo "Error: no saved run config at $saved_config"; exit 1; fi
    # vf-eval resumes only under the exact saved config, endpoint included: a moved server is
    # written into the saved config first (checked to serve the run's model, and recorded)
    if [[ "${ARGS["base_url"]}" != "none" ]]; then
        python scripts/migrate_endpoint.py --run_dir "$run_dir" --base_url "${ARGS["base_url"]}" || exit 1
    fi
    python scripts/write_run_manifest.py --run_dir "$run_dir" --config "$saved_config" || echo "Warning: no run manifest written"
    vf-eval @ "$saved_config" --resume
else
    if [[ "${ARGS["config"]}" == "none" ]]; then echo "Error: --config is required for a new run"; exit 1; fi
    base_url_args=()
    if [[ "${ARGS["base_url"]}" != "none" ]]; then base_url_args=(--client.base-url "${ARGS["base_url"]}"); fi
    python scripts/write_run_manifest.py --run_dir "$run_dir" --config "${ARGS["config"]}" --base_url "${ARGS["base_url"]}" || echo "Warning: no run manifest written"
    vf-eval @ "${ARGS["config"]}" -o "$runs_dir" --run.name "$name" --env.checkpoint-dir "$run_dir/checkpoints" -r 1 "${base_url_args[@]}"
fi
