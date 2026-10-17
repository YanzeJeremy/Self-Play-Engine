#!/usr/bin/env bash
# Expert iteration (variant A), end to end, for hidden-rule rock paper scissors:
#   1 generate   serve the base model (vLLM, data-parallel over --gpus) and let it play --gen_games games of
#                --gen_rounds rounds against each training rule (sampled, prompts as always, no forecasts)
#   2 select     keep the model's own replies from rounds it won and kept winning (no rule is ever written
#                into the data): scripts/expert_iteration/select_sft_data.py, aiming at --target_min..--target_max
#   3 sft        LoRA SFT of the base model on them with prime-rl (scripts/expert_iteration/make_sft_config.py)
#   4 export     the LoRA adapter, as a PEFT adapter vLLM can serve (scripts/expert_iteration/export_lora_adapter.py)
#   5 evaluate   serve base + adapter on one server; check that the adapter is applied (check_adapter.py); greedy
#                evaluation of both on the held-out copycat (copy_last: never trained on) and on fresh seeds of every
#                training rule, at --eval_rounds and again at --eval_long_rounds with fewer games
#   6 summarize  paired base-vs-SFT comparison per rule: <experiment>/summary.md and summary.json
#
#   bash scripts/run_expert_iteration.sh --smoke true     # first: a tiny run through every stage (~30-60 min)
#   bash scripts/run_expert_iteration.sh                  # the full experiment (about a day on 2 H200)
#   bash scripts/run_expert_iteration.sh --stop_after select   # only generate and select the data (GPUs then free)
#
# Run it inside the GPU allocation (tmux recommended), from the repo root, with the GPUs free: it starts and stops
# its own vLLM servers (stop any vllm serve you started). Everything lands in $storage_dir/expert_iteration/<name>/
# (configs, runs, data, sft, adapter, logs, stages, summary). Rerunning the same command resumes: finished stages are
# skipped, unfinished evaluation runs resume, an interrupted SFT resumes from its latest checkpoint (or restarts
# clean if it saved none). --redo_from <stage> reruns that stage and the ones after it (redoing SFT moves the old
# training folder aside to sft.replaced-<time>, never deletes it). --stop_after <stage> stops once it is done.
# A finished stage keeps its settings (stages/<stage>.params): changing one of them is refused unless that stage
# is redone (--redo_from); settings of stages not yet run can change freely.
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
ARGS["thresholds"]="0.8"         # selection win fractions to try, strictest first (relaxing admits luck: see select_sft_data.py)
ARGS["max_rule_share"]="0.35"    # most any opponent may make up of the final train set
ARGS["eval_long_rounds"]="20"    # a second, longer evaluation: discovery over time
ARGS["eval_long_games"]="20"     # per training rule (0: no long evaluation)
ARGS["eval_long_copycat_games"]="50"
ARGS["ckpt_interval"]="0"        # SFT steps between checkpoints (0: a quarter of the run, at least 25)
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
    ARGS["target_min"]="1"; ARGS["target_max"]="300"; ARGS["batch_size"]="2"; ARGS["max_rule_share"]="1.0"
    ARGS["eval_long_rounds"]="6"; ARGS["eval_long_games"]="1"; ARGS["eval_long_copycat_games"]="1"
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

# --- settings: a finished stage keeps its own ---------------------------------------------------------
declare -A STAGE_KEYS=(
    [generate]="model train_rules gen_games gen_rounds"
    [select]="target_min target_max thresholds max_rule_share seq_len"
    [sft]="model lora_rank lora_alpha lr epochs seq_len batch_size ckpt_interval"
    [export]="lora_rank lora_alpha"
    [evaluate]="eval_rules eval_games eval_copycat_games eval_rounds eval_skip eval_long_rounds eval_long_games eval_long_copycat_games"
    [summarize]="heldout"
)
stage_params() { for key in ${STAGE_KEYS[$1]}; do echo "$key=${ARGS[$key]}"; done; }

if [[ "${ARGS["redo_from"]}" != "none" ]]; then
    found=false
    for stage in "${STAGES[@]}"; do
        [[ "$stage" == "${ARGS["redo_from"]}" ]] && found=true
        $found && rm -f "$exp_dir/stages/$stage.done" "$exp_dir/stages/$stage.params"
    done
    $found || fail "--redo_from must be one of: ${STAGES[*]}"
    if [[ " generate select sft " =~ " ${ARGS["redo_from"]} " && -d "$exp_dir/sft" ]]; then
        aside="$exp_dir/sft.replaced-$(date +%Y%m%d-%H%M%S)"
        mv "$exp_dir/sft" "$aside"
        echo "== the previous SFT folder is kept at $aside (delete it to free its checkpoint's space)"
    fi
fi
if [[ "${ARGS["stop_after"]}" != "none" && ! " ${STAGES[*]} " =~ " ${ARGS["stop_after"]} " ]]; then
    fail "--stop_after must be one of: ${STAGES[*]}"
fi
mismatches=()
for stage in "${STAGES[@]}"; do
    [[ -f "$exp_dir/stages/$stage.done" ]] || continue
    saved="$exp_dir/stages/$stage.params"
    [[ -f "$saved" ]] || saved="$exp_dir/params.txt"   # stages finished before per-stage settings were kept
    [[ -f "$saved" ]] || continue
    for key in ${STAGE_KEYS[$stage]}; do
        line="$(grep -m 1 "^$key=" "$saved")"
        [[ -n "$line" && "$line" != "$key=${ARGS[$key]}" ]] && mismatches+=("$stage: ${line} (now ${ARGS[$key]})")
    done
done
if [[ ${#mismatches[@]} -gt 0 ]]; then
    echo "Error: finished stages were run with other settings; redo them (--redo_from <stage>) or use a new --name:"
    printf '  %s\n' "${mismatches[@]}"
    exit 1
fi

stage_done() { [[ -f "$exp_dir/stages/$1.done" ]]; }
mark_done() { stage_params "$1" > "$exp_dir/stages/$1.params"; date > "$exp_dir/stages/$1.done"; echo "== stage $1 done"; }
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

run_all() {  # run_all <list file> <run name prefix> [suffix]: every "<rule> <config>" line as run <prefix>-<rule><suffix>
    local list="$1" prefix="$2" suffix="${3:-}" pids=() rules=() failed=()
    while read -r rule config; do
        ( run_condition "$config" "$prefix-$rule$suffix" ) > "$exp_dir/logs/$prefix-$rule$suffix.log" 2>&1 &
        pids+=($!); rules+=("$rule")
    done < "$list"
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || failed+=("${rules[$i]}")
    done
    [[ ${#failed[@]} -eq 0 ]] || fail "runs failed for: ${failed[*]} (logs: $exp_dir/logs/$prefix-<rule>$suffix.log); rerun to resume"
    echo "== all $prefix$suffix runs complete"
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
    select_args=(--thresholds "${ARGS["thresholds"]}" --max_rule_share "${ARGS["max_rule_share"]}")
    # smoke: any won round counts and nothing is held out for validation (a handful of examples must all train);
    # later flags win, so these override the defaults above
    $smoke && select_args+=(--thresholds 0.8,0.6,0.0 --min_window 1 --hard_min 1 --val_fraction 0)
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
        --batch_size "${ARGS["batch_size"]}" --epochs "${ARGS["epochs"]}" --ckpt_interval "${ARGS["ckpt_interval"]}" \
        "${sft_args[@]}" || fail "make_sft_config"
    gpus_free
    # an earlier attempt of this stage: resume its latest checkpoint, or restart clean if it saved none
    # (prime-rl refuses a run folder with a previous run's artifacts unless told which)
    sft_run_dir="$exp_dir/sft/outputs/$name-sft"
    latest_ckpt="$(ls -d "$sft_run_dir"/checkpoints/step_*/trainer/.metadata 2> /dev/null | sort -V | tail -n 1)"
    sft_mode=()
    if [[ -n "$latest_ckpt" ]]; then
        sft_mode=(--resume)
        echo "== SFT: resuming from $(dirname "$(dirname "$latest_ckpt")")"
    elif python -c "import sys; from pathlib import Path; from prime_rl.utils.pathing import has_run_artifacts; sys.exit(0 if has_run_artifacts(Path(sys.argv[1])) else 1)" "$sft_run_dir"; then
        sft_mode=(--clean)
        echo "== SFT: an earlier attempt saved no checkpoint; restarting it clean"
    fi
    echo "== SFT (log: $exp_dir/logs/sft.log)"
    ( cd prime-rl && CUDA_VISIBLE_DEVICES="$gpus" ${EI_SFT_CMD:-sft} @ "$exp_dir/sft/sft.toml" "${sft_mode[@]}" ) 2>&1 | tee -a "$exp_dir/logs/sft.log"
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
long_suffix="r${ARGS["eval_long_rounds"]}"

# --- 5 evaluate ------------------------------------------------------------------------------------------
if ! stage_done evaluate; then
    [[ -n "$adapter_hash" ]] || fail "no exported adapter in $exp_dir/adapter"
    for arm in base "$sft_arm"; do
        arm_model="$model"; [[ "$arm" == "$sft_arm" ]] && arm_model="$adapter_name"
        python scripts/expert_iteration/make_configs.py --out_dir "$exp_dir/configs" --kind eval --model "$arm_model" \
            --rules "${ARGS["eval_rules"]}" --games "${ARGS["eval_games"]}" --copycat_games "${ARGS["eval_copycat_games"]}" \
            --rounds "${ARGS["eval_rounds"]}" --skip "${ARGS["eval_skip"]}" --temperature 0 \
            --max_concurrent "${ARGS["eval_concurrency"]}" --tag "$arm" > "$exp_dir/configs/eval-$arm.list" || fail "make_configs (eval $arm)"
        if [[ "${ARGS["eval_long_games"]}" -gt 0 ]]; then
            python scripts/expert_iteration/make_configs.py --out_dir "$exp_dir/configs" --kind eval --model "$arm_model" \
                --rules "${ARGS["eval_rules"]}" --games "${ARGS["eval_long_games"]}" --copycat_games "${ARGS["eval_long_copycat_games"]}" \
                --rounds "${ARGS["eval_long_rounds"]}" --skip "${ARGS["eval_skip"]}" --temperature 0 \
                --max_concurrent "${ARGS["eval_concurrency"]}" --tag "$arm-$long_suffix" > "$exp_dir/configs/eval-$arm-$long_suffix.list" \
                || fail "make_configs (eval $arm, long)"
        fi
    done
    start_server eval --enable-lora --max-lora-rank "${ARGS["lora_rank"]}" --max-loras 1 --lora-modules "$adapter_name=$exp_dir/adapter"
    if [[ -z "${EI_NO_GPU:-}" ]]; then
        check_args=()
        $smoke || check_args=(--require_improvement)
        python scripts/expert_iteration/check_adapter.py --base_url "$base_url" --base_model "$model" --adapter_name "$adapter_name" \
            --data_dir "$exp_dir/data" --out "$exp_dir/adapter_check.json" "${check_args[@]}" || fail "adapter check (see $exp_dir/adapter_check.json)"
    fi
    run_all "$exp_dir/configs/eval-base.list" "$name-eval-base"
    run_all "$exp_dir/configs/eval-$sft_arm.list" "$name-eval-$sft_arm"
    if [[ "${ARGS["eval_long_games"]}" -gt 0 ]]; then
        run_all "$exp_dir/configs/eval-base-$long_suffix.list" "$name-eval-base" "-$long_suffix"
        run_all "$exp_dir/configs/eval-$sft_arm-$long_suffix.list" "$name-eval-$sft_arm" "-$long_suffix"
    fi
    stop_server
    mark_done evaluate
fi
stop_if_requested evaluate

# --- 6 summarize -----------------------------------------------------------------------------------------
if ! stage_done summarize; then
    long_args=()
    [[ "${ARGS["eval_long_games"]}" -gt 0 ]] && long_args=(--long_suffix "-$long_suffix" --long_label "${ARGS["eval_long_rounds"]} rounds")
    python scripts/expert_iteration/summarize.py --runs_dir "$runs_dir" --name "$name" --sft_arm "$sft_arm" --rules "${ARGS["eval_rules"]}" \
        --heldout "${ARGS["heldout"]}" --main_label "${ARGS["eval_rounds"]} rounds" "${long_args[@]}" \
        --out_json "$exp_dir/summary.json" --out_md "$exp_dir/summary.md" || fail "summary (see above)"
    mark_done summarize
fi
echo "== done: $exp_dir/summary.md (data: $exp_dir/data/stats.json)"
