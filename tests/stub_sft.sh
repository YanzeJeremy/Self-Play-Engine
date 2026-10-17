#!/usr/bin/env bash
# Test double for prime-rl's `sft` (used via EI_SFT_CMD, called as `stub_sft.sh @ <config.toml>`): writes a small DCP
# checkpoint with LoRA tensors where prime-rl would (<output_dir>/<run.name>/checkpoints/step_<max_steps>/trainer),
# with prime-rl's key layout (app.model.<module>.lora_A.0 / .lora_B.0) and the config's LoRA rank.
python - "$2" <<'PY'
import sys, tomllib, torch
from torch.distributed.checkpoint import save
cfg = tomllib.load(open(sys.argv[1], "rb"))
rank = cfg["model"]["lora"]["rank"]
model = {}
for layer in range(2):
    for proj, (fan_in, fan_out) in {"self_attn.q_proj": (32, 48), "mlp.down_proj": (64, 32)}.items():
        module = f"model.language_model.layers.{layer}.{proj}"
        model[f"{module}.weight"] = torch.randn(fan_out, fan_in)
        model[f"{module}.lora_A.0"] = torch.randn(rank, fan_in)
        model[f"{module}.lora_B.0"] = torch.randn(fan_out, rank)
path = f"{cfg['output_dir']}/{cfg['run']['name']}/checkpoints/step_{cfg['max_steps']}/trainer"
save({"app": {"model": model}}, checkpoint_id=path)
print(f"stub sft: wrote {path}")
PY
