#!/usr/bin/env bash
# Test double for prime-rl's `sft` (used via EI_SFT_CMD, called as `stub_sft.sh @ <config.toml> [--resume|--clean]`).
# Like the real trainer it refuses a run folder holding a previous run's artifacts unless told --resume or --clean
# (prime-rl's own validate_run_dir decides), resumes only from an existing checkpoint, and writes checkpoints where
# prime-rl does (<output_dir>/<run.name>/checkpoints/step_<N>/trainer, DCP, keys app.model.<module>.lora_A.0 / .lora_B.0,
# the config's LoRA rank). STUB_SFT_CRASH=before_ckpt exits 1 after writing a log; =after_ckpt after checkpoint step 1.
# Every call is appended to <output_dir>/stub_sft_calls.log.
python - "$@" <<'PY'
import os, sys, tomllib
from pathlib import Path
import torch
from torch.distributed.checkpoint import save
from prime_rl.utils.pathing import validate_run_dir

args = sys.argv[1:]
cfg = tomllib.load(open(args[1], "rb"))
flags = args[2:]
output_dir = Path(cfg["output_dir"])
run_dir = output_dir / cfg["run"]["name"]
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / "stub_sft_calls.log", "a") as log:
    log.write(" ".join(flags or ["(fresh)"]) + "\n")
validate_run_dir(run_dir, output_dir=output_dir, resuming="--resume" in flags, clean="--clean" in flags)
if "--resume" in flags and not list(run_dir.glob("checkpoints/step_*/trainer/.metadata")):
    sys.exit("stub sft: --resume without a checkpoint")
(run_dir / "logs").mkdir(parents=True, exist_ok=True)
(run_dir / "logs" / "trainer.log").write_text("training\n")
crash = os.environ.get("STUB_SFT_CRASH")
if crash == "before_ckpt":
    sys.exit("stub sft: simulated crash before the first checkpoint")

rank = cfg["model"]["lora"]["rank"]

def checkpoint(step):
    model = {}
    for layer in range(2):
        for proj, (fan_in, fan_out) in {"self_attn.q_proj": (32, 48), "mlp.down_proj": (64, 32)}.items():
            module = f"model.language_model.layers.{layer}.{proj}"
            model[f"{module}.weight"] = torch.randn(fan_out, fan_in)
            model[f"{module}.lora_A.0"] = torch.randn(rank, fan_in)
            model[f"{module}.lora_B.0"] = torch.randn(fan_out, rank)
    save({"app": {"model": model}}, checkpoint_id=str(run_dir / f"checkpoints/step_{step}/trainer"))

if crash == "after_ckpt":
    checkpoint(1)
    sys.exit("stub sft: simulated crash after checkpoint step 1")
checkpoint(cfg["max_steps"])
print(f"stub sft: wrote {run_dir}/checkpoints/step_{cfg['max_steps']}/trainer ({' '.join(flags) or 'fresh'})")
PY
