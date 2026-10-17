#!/usr/bin/env bash
# Test double for scripts/eval.sh (used via EVAL_SCRIPT): writes a fake run into $EVAL_RUNS_DIR with
# $STUB_GAMES finished games (default: all), logs its arguments to $EVAL_RUNS_DIR/stub_calls.log, and
# exits with $STUB_EXIT (default 0).
config=""; name=""; resume="false"; base_url="none"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) config="$2"; shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --resume) resume="$2"; shift 2 ;;
        --base_url) base_url="$2"; shift 2 ;;
        *) shift ;;
    esac
done
echo "name=$name resume=$resume base_url=$base_url" >> "$EVAL_RUNS_DIR/stub_calls.log"
config_args=()
if [[ -n "$config" ]]; then config_args=(--config "$config"); fi
python tests/fake_runs.py --runs_dir "$EVAL_RUNS_DIR" --name "$name" "${config_args[@]}" --games "${STUB_GAMES:-all}" --base_url "$base_url" || exit 1
exit "${STUB_EXIT:-0}"
