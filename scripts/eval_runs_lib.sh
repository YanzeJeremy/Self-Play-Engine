# Shared by scripts that drive evaluation runs (sourced, after scripts/utils.sh). Needs:
#   runs_dir     where runs live (eval.sh's EVAL_RUNS_DIR)
#   eval_script  scripts/eval.sh, or a test double
#   base_url     the server to use ("none": each config's own)
#
# run_condition <config> <run name>: leaves the run complete, or stops the calling script. An
# existing run must be the requested condition (scripts/run_identity.py check); a complete one is
# skipped, an unfinished one resumed against $base_url; afterwards every configured game must be
# finished (run_identity.py complete). With ARCHIVE_MISMATCHED_RUNS=1 (a stage being redone on purpose), an
# existing run of another condition is moved to $runs_dir/archive/<name>.<time> and the run starts afresh;
# runs of the same condition are still reused.

fail() { echo "Error: $*"; exit 1; }

is_complete() { python scripts/run_identity.py complete --run_dir "$runs_dir/$1"; }

run_condition() {
    local config="$1" name="$2" run_dir="$runs_dir/$2"
    if [[ -f "$run_dir/configs/resolved/eval.json" && "${ARCHIVE_MISMATCHED_RUNS:-0}" == "1" ]] \
        && ! python scripts/run_identity.py check --run_dir "$run_dir" --config "$config" > /dev/null; then
        # an explicitly redone stage: a run of an older condition is kept aside, never reused or deleted
        local archived="$runs_dir/archive/$name.$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$runs_dir/archive" && mv "$run_dir" "$archived"
        echo "== $name: was another condition; archived to $archived"
    fi
    if [[ -f "$run_dir/configs/resolved/eval.json" ]]; then
        python scripts/run_identity.py check --run_dir "$run_dir" --config "$config" || fail "$name exists but is another condition"
        if is_complete "$name" > /dev/null; then echo "== $name: complete, skipping"; return 0; fi
        echo "== $name: resuming"
        EVAL_RUNS_DIR="$runs_dir" bash "$eval_script" --name "$name" --resume true --base_url "$base_url" || fail "evaluation of $name failed (exit $?)"
    else
        echo "== $name: starting"
        EVAL_RUNS_DIR="$runs_dir" bash "$eval_script" --config "$config" --name "$name" --base_url "$base_url" || fail "evaluation of $name failed (exit $?)"
    fi
    is_complete "$name" || fail "$name is incomplete after evaluation; rerun to resume it"
}
