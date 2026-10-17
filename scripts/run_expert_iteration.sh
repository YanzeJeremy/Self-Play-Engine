#!/usr/bin/env bash
# Expert iteration (variant A), end to end, for hidden-rule rock paper scissors:
#   1 generate   serve the base model (vLLM, data-parallel over --gpus) and let it play --gen_games games of
#                --gen_rounds rounds against each training rule (sampled, prompts as always, no forecasts)
#   2 select     keep the model's own replies from rounds it won and kept winning (no rule is ever written
#                into the data): scripts/expert_iteration/select_sft_data.py, aiming at --target_min..--target_max
#   3 sft        LoRA SFT of the base model on them with prime-rl (scripts/expert_iteration/make_sft_config.py)
#   4 export     the LoRA adapter, as a PEFT adapter vLLM can serve (scripts/expert_iteration/export_lora_adapter.py)
#   5 evaluate   serve base + adapter on one server; greedy evaluation of both on the held-out copycat
#                (copy_last: never trained on) and on fresh seeds of every training rule
#   6 summarize  paired base-vs-SFT comparison per rule: <experiment>/summary.md and summary.json
#
#   bash scripts/run_expert_iteration.sh --smoke true     # first: a tiny run through every stage (~30-60 min)
#   bash scripts/run_expert_iteration.sh                  # the full experiment (about a day on 2 H200)
#   bash scripts/run_expert_iteration.sh --stop_after select   # only generate and select the data (GPUs then free)
#
# Run it inside the GPU allocation (tmux recommended), from the repo root, with the GPUs free: it starts and stops
# its own vLLM servers (stop any vllm serve you started). Everything lands in $storage_dir/expert_iteration/<name>/
# (configs, runs, data, sft, adapter, logs, stages, summary). Rerunning the same command resumes: finished stages are
# skipped, unfinished evaluation runs resume. --redo_from <stage> reruns that stage and the ones after it. An
# experiment refuses to continue under different settings (params.txt); pick a new --name instead.
#
# Test hooks (CPU tests only): EI_NO_GPU=1 skips GPU checks and servers; EVAL_SCRIPT and EI_SFT_CMD replace eval.sh
# and prime-rl's sft; EI_ROOT replaces $storage_dir/expert_iteration.

if [[ -z "${EI_NO_GPU:-}" ]]; then
    # vLLM and prime-rl compile kernels with nvcc; load CUDA before the venv (scripts/utils.sh) so its python wins
    type module > /dev/null 2>&1 || { echo "Error: the module command is unavailable; load ver/2607 gcc/15.3.0 cuda/13.0.3 yourself"; exit 1; }
    module load ver/2607 gcc/15.3.0 cuda/13.0.3 > /dev/null 2>&1 || { echo "Error: module load ver/2607 gcc/15.3.0 cuda/13.0.3 failed"; exit 1; }
fi
source scripts/utils.sh || { echo "Could not source utils"; exit 1; }

declare -A ARGS
ARGS["name"]="ei-v1"
ARGS["model"]="google/gemma-4-31b-it"
ARGS["train_rules"]="counter_last,beaten_by_last,copy_two_back,repeat_self,cycle"
ARGS["eval_rules"]="copy_last,counter_last,beaten_by_last,copy_two_back,repeat_self,cycle"
ARGS["heldout"]="copy_last"
ARGS["gen_games"]="200"          # per training rule (5 rules: 1000 games)
ARGS["gen_rounds"]="20"
ARGS["gen_concurrency"]="4"      # games in flight per rule (all rules run at once)
ARGS["eval_games"]="40"          # per training rule, fresh seeds from --eval_skip
ARGS["eval_copycat_games"]="100" # held-out copycat, seeds 0..99 (the B0 baseline's seeds)
ARGS["eval_rounds"]="10"
ARGS["eval_skip"]="50000"
ARGS["eval_concurrency"]="4"
ARGS["target_min"]="5000"
ARGS["target_max"]="10000"
ARGS["lora_rank"]="16"
ARGS["lora_alpha"]="32"
ARGS["lr"]="1e-4"
ARGS["epochs"]="1"
ARGS["seq_len"]="8192"
ARGS["batch_size"]="16"
ARGS["gpus"]="0,1"
ARGS["port"]="8000"
ARGS["max_model_len"]="16384"
ARGS["smoke"]="false"
ARGS["redo_from"]="none"         # generate | select | sft | export | evaluate | summarize
ARGS["stop_after"]="none"        # stop (successfully) once this stage is done, e.g. select

REQUIRED_ARGS=()
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
smoke=false
if [[ "${ARGS["smoke"]}" == "true" ]]; then
    smoke=true
    ARGS["name"]="${ARGS["name"]}-smoke"
    ARGS["gen_games"]="2"; ARGS["gen_rounds"]="4"; ARGS["gen_concurrency"]="2"
    ARGS["eval_games"]="2"; ARGS["eval_copycat_games"]="2"; ARGS["eval_rounds"]="4"; ARGS["eval_concurrency"]="2"
    ARGS["target_min"]="1"; ARGS["target_max"]="300"; ARGS["batch_size"]="2"
fi

name="${ARGS["name"]}"; model="${ARGS["model"]}"; gpus="${ARGS["gpus"]}"; port="${ARGS["port"]}"
IFS=',' read -ra gpu_list <<< "$gpus"; num_gpus="${#gpu_list[@]}"
IFS=',' read -ra train_rules <<< "${ARGS["train_rules"]}"
IFS=',' read -ra eval_rules <<< "${ARGS["eval_rules"]}"
exp_dir="${EI_ROOT:-$storage_dir/expert_iteration}/$name"
runs_dir="$exp_dir/runs"
eval_script="${EVAL_SCRIPT:-scripts/eval.sh}"
base_url="http://localhost:$port/v1"
STAGES=(generate select sft export evaluate summarize)
mkdir -p "$exp_dir"/{configs,runs,logs,stages}
source scripts/eval_runs_lib.sh || { echo "Could not source scripts/eval_runs_lib.sh"; exit 1; }

# --- one experiment, one set of settings --------------------------------------------------------------
params="$(for key in $(printf '%s\n' "${!ARGS[@]}" | sort); do
    case "$key" in gpus|port|redo_from|stop_after) ;; *) echo "$key=${ARGS[$key]}" ;; esac
done)"
if [[ -f "$exp_dir/params.txt" ]]; then
    if [[ "$(cat "$exp_dir/params.txt")" != "$params" ]]; then
        echo "Error: $exp_dir was started with other settings; use a new --name. Differences (saved < > now):"
        diff <(cat "$exp_dir/params.txt") <(echo "$params"); exit 1
    fi
else
    echo "$params" > "$exp_dir/params.txt"
fi

if [[ "${ARGS["redo_from"]}" != "none" ]]; then
    found=false
    for stage in "${STAGES[@]}"; do
        [[ "$stage" == "${ARGS["redo_from"]}" ]] && found=true
        $found && rm -f "$exp_dir/stages/$stage.done"
    done
    $found || fail "--redo_from must be one of: ${STAGES[*]}"
fi
if [[ "${ARGS["stop_after"]}" != "none" && ! " ${STAGES[*]} " =~ " ${ARGS["stop_after"]} " ]]; then
    fail "--stop_after must be one of: ${STAGES[*]}"
fi
stage_done() { [[ -f "$exp_dir/stages/$1.done" ]]; }
mark_done() { date > "$exp_dir/stages/$1.done"; echo "== stage $1 done"; }
stop_if_requested() {
    if [[ "${ARGS["stop_after"]}" == "$1" ]]; then
        echo "== stopping after stage $1 (--stop_after); rerun without it (same settings) to continue"
        exit 0
    fi
}

# --- GPUs and vLLM ---------------------------------------------------------------------------------------
server_pid=""
gpus_free() {
    [[ -n "${EI_NO_GPU:-}" ]] && return 0
    local busy; busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
    [[ "$busy" -eq 0 ]] || fail "the GPUs are in use by $busy process(es) (nvidia-smi); stop them (e.g. your own vllm serve) first"
}
start_server() {  # start_server <label> [extra vllm serve args...]
    local label="$1"; shift
    if [[ -n "${EI_NO_GPU:-}" ]]; then echo "== (EI_NO_GPU) no $label server"; return 0; fi
    curl -s -m 3 "$base_url/models" > /dev/null && fail "something already serves on port $port; stop it first"
    gpus_free
    local log="$exp_dir/logs/vllm-$label.log"
    echo "== starting vLLM ($label) on GPUs $gpus, port $port; log: $log"
    CUDA_VISIBLE_DEVICES="$gpus" setsid vllm serve "$model" --port "$port" --data-parallel-size "$num_gpus" \
        --max-model-len "${ARGS["max_model_len"]}" --gpu-memory-utilization 0.9 "$@" > "$log" 2>&1 &
    server_pid=$!
    for _ in $(seq 1 720); do  # up to an hour: the first start compiles kernels
        if curl -s -m 3 "$base_url/models" | grep -q '"id"'; then echo "== vLLM ($label) ready"; return 0; fi
        kill -0 "$server_pid" 2> /dev/null || { tail -n 30 "$log"; server_pid=""; fail "vLLM ($label) exited during startup; see $log"; }
        sleep 5
    done
    fail "vLLM ($label) not ready after an hour; see $log"
}
stop_server() {
    [[ -z "$server_pid" ]] && return 0
    echo "== stopping vLLM"
    kill -TERM -- "-$server_pid" 2> /dev/null || kill -TERM "$server_pid" 2> /dev/null
    wait "$server_pid" 2> /dev/null
    server_pid=""
    for _ in $(seq 1 60); do
        [[ $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .) -eq 0 ]] && return 0
        sleep 2
    done
    echo "Warning: GPU processes remain after stopping vLLM"
}
trap stop_server EXIT

run_all() {  # run_all <list file> <run name prefix>: every "<rule> <config>" line as a run, all at once
    local list="$1" prefix="$2" pids=() rules=() failed=()
    while read -r rule config; do
        ( run_condition "$config" "$prefix-$rule" ) > "$exp_dir/logs/$prefix-$rule.log" 2>&1 &
        pids+=($!); rules+=("$rule")
    done < "$list"
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || failed+=("${rules[$i]}")
    done
    [[ ${#failed[@]} -eq 0 ]] || fail "runs failed for: ${failed[*]} (logs: $exp_dir/logs/$prefix-<rule>.log); rerun to resume"
    echo "== all $prefix runs complete"
}

# --- 1 generate ------------------------------------------------------------------------------------------
if ! stage_done generate; then
    python scripts/expert_iteration/make_configs.py --out_dir "$exp_dir/configs" --kind gen --model "$model" \
        --rules "${ARGS["train_rules"]}" --games "${ARGS["gen_games"]}" --rounds "${ARGS["gen_rounds"]}" \
        --temperature 1.0 --max_concurrent "${ARGS["gen_concurrency"]}" > "$exp_dir/configs/gen.list" || fail "make_configs (gen)"
    start_server base
    run_all "$exp_dir/configs/gen.list" "$name-gen"
    stop_server
    mark_done generate
fi
stop_if_requested generate

# --- 2 select --------------------------------------------------------------------------------------------
if ! stage_done select; then
    select_args=()
    # smoke: any won round counts and nothing is held out for validation (a handful of examples must all train)
    $smoke && select_args=(--thresholds 0.8,0.6,0.0 --min_window 1 --hard_min 1 --val_fraction 0)
    python scripts/expert_iteration/select_sft_data.py $(for rule in "${train_rules[@]}"; do echo --runs "$runs_dir/$name-gen-$rule"; done) \
        --out_dir "$exp_dir/data" --model "$model" --target_min "${ARGS["target_min"]}" --target_max "${ARGS["target_max"]}" \
        --max_tokens_per_example $(( ${ARGS["seq_len"]} - 256 )) "${select_args[@]}" | tee "$exp_dir/logs/select.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "data selection failed (log: $exp_dir/logs/select.log)"
    mark_done select
fi
stop_if_requested select

# --- 3 sft -----------------------------------------------------------------------------------------------
if ! stage_done sft; then
    sft_args=()
    $smoke && sft_args=(--max_steps 2)
    python scripts/expert_iteration/make_sft_config.py --data_dir "$exp_dir/data" --out "$exp_dir/sft/sft.toml" \
        --output_dir "$exp_dir/sft/outputs" --run_name "$name-sft" --model "$model" --num_gpus "$num_gpus" \
        --rank "${ARGS["lora_rank"]}" --alpha "${ARGS["lora_alpha"]}" --lr "${ARGS["lr"]}" --seq_len "${ARGS["seq_len"]}" \
        --batch_size "${ARGS["batch_size"]}" --epochs "${ARGS["epochs"]}" "${sft_args[@]}" || fail "make_sft_config"
    gpus_free
    echo "== SFT (log: $exp_dir/logs/sft.log)"
    ( cd prime-rl && CUDA_VISIBLE_DEVICES="$gpus" ${EI_SFT_CMD:-sft} @ "$exp_dir/sft/sft.toml" ) 2>&1 | tee "$exp_dir/logs/sft.log"
    [[ ${PIPESTATUS[0]} -eq 0 ]] || fail "SFT failed (log: $exp_dir/logs/sft.log)"
    mark_done sft
fi
stop_if_requested sft

# --- 4 export --------------------------------------------------------------------------------------------
if ! stage_done export; then
    python scripts/expert_iteration/export_lora_adapter.py --ckpt "$exp_dir/sft/outputs/$name-sft" --out_dir "$exp_dir/adapter" \
        --base_model "$model" --rank "${ARGS["lora_rank"]}" --alpha "${ARGS["lora_alpha"]}" || fail "adapter export"
    mark_done export
fi
stop_if_requested export

# The adapter is named, served and evaluated under a hash of its weights, so evaluation runs of an earlier
# adapter (e.g. before --redo_from sft) are never taken for this one's.
adapter_hash="$(sha256sum "$exp_dir/adapter/adapter_model.safetensors" 2> /dev/null | cut -c1-10)"
adapter_name="$name-sft-$adapter_hash"
sft_arm="sft-$adapter_hash"

# --- 5 evaluate ------------------------------------------------------------------------------------------
if ! stage_done evaluate; then
    [[ -n "$adapter_hash" ]] || fail "no exported adapter in $exp_dir/adapter"
    for arm in base "$sft_arm"; do
        arm_model="$model"; [[ "$arm" == "$sft_arm" ]] && arm_model="$adapter_name"
        python scripts/expert_iteration/make_configs.py --out_dir "$exp_dir/configs" --kind eval --model "$arm_model" \
            --rules "${ARGS["eval_rules"]}" --games "${ARGS["eval_games"]}" --copycat_games "${ARGS["eval_copycat_games"]}" \
            --rounds "${ARGS["eval_rounds"]}" --skip "${ARGS["eval_skip"]}" --temperature 0 \
            --max_concurrent "${ARGS["eval_concurrency"]}" --tag "$arm" > "$exp_dir/configs/eval-$arm.list" || fail "make_configs (eval $arm)"
    done
    start_server eval --enable-lora --max-lora-rank "${ARGS["lora_rank"]}" --max-loras 1 --lora-modules "$adapter_name=$exp_dir/adapter"
    run_all "$exp_dir/configs/eval-base.list" "$name-eval-base"
    run_all "$exp_dir/configs/eval-$sft_arm.list" "$name-eval-$sft_arm"
    stop_server
    mark_done evaluate
fi
stop_if_requested evaluate

# --- 6 summarize -----------------------------------------------------------------------------------------
if ! stage_done summarize; then
    python scripts/expert_iteration/summarize.py --runs_dir "$runs_dir" --name "$name" --sft_arm "$sft_arm" --rules "${ARGS["eval_rules"]}" \
        --heldout "${ARGS["heldout"]}" --out_json "$exp_dir/summary.json" --out_md "$exp_dir/summary.md" || fail "summary (see above)"
    mark_done summarize
fi
echo "== done: $exp_dir/summary.md (data: $exp_dir/data/stats.json)"
