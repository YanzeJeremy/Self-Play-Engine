"""Export the LoRA adapter from a prime-rl SFT checkpoint (DCP) as a PEFT adapter vLLM can serve.

    python scripts/expert_iteration/export_lora_adapter.py --ckpt <dir> --out_dir <adapter dir> \
        --base_model google/gemma-4-31b-it --rank 16 --alpha 32

prime-rl's SFT checkpoints are DCP-sharded and hold the whole model; nothing in prime-rl exports
the adapter from them (its PEFT writer only runs for online evals). This reads only the LoRA
tensors (keys `app.model.<module>.lora_A.0` / `.lora_B.0`, adapter index 0), and writes what
prime-rl's own adapter broadcast writes (FileSystemWeightSender + save_lora_config):
`adapter_model.safetensors` with keys `<module>.lora_A.weight` / `<module>.lora_B.weight` in
bfloat16, and `adapter_config.json`. --ckpt may be a checkpoint dir, a step dir, or a run's
output dir (the latest `step_*/trainer` below it is used). It refuses unless every adapted
module has both matrices with shapes A [rank, in] and B [out, rank].
"""

import json
import re
from pathlib import Path

import click

KEY = re.compile(r"^app\.model\.(?P<module>.+)\.(?P<which>lora_[AB])(?:\.(?P<index>\d+))?$")


def find_checkpoint(*, path: Path) -> Path:
    """The DCP directory (holding .metadata) at or below `path`; the latest step if several."""
    if (path / ".metadata").exists():
        return path
    candidates = [p.parent for p in path.rglob(".metadata")]
    if not candidates:
        raise SystemExit(f"Error: no DCP checkpoint (.metadata) under {path}")
    step = lambda p: max((int(m.group(1)) for part in p.parts if (m := re.fullmatch(r"step_(\d+)", part))), default=-1)
    return max(candidates, key=step)


def read_adapter(*, ckpt: Path, rank: int) -> dict:
    import torch
    from torch.distributed.checkpoint import FileSystemReader, load

    reader = FileSystemReader(str(ckpt))
    metadata = reader.read_metadata().state_dict_metadata
    wanted = {}
    for key, meta in metadata.items():
        match = KEY.match(key)
        if match and (match["index"] or "0") == "0":
            wanted[key] = (match["module"], match["which"], meta)
    if not wanted:
        sample = [k for k in metadata if "lora" in k][:5] or list(metadata)[:5]
        raise SystemExit(f"Error: no LoRA tensors (app.model.<module>.lora_A.0) in {ckpt}; keys look like {sample}")
    state = {key: torch.empty(tuple(meta.size), dtype=meta.properties.dtype) for key, (_, _, meta) in wanted.items()}
    load(state, storage_reader=reader)
    adapter, modules = {}, {}
    for key, (module, which, _) in wanted.items():
        adapter[f"{module}.{which}.weight"] = state[key].to(torch.bfloat16).contiguous()
        modules.setdefault(module, {})[which] = state[key].shape
    for module, parts in modules.items():
        if set(parts) != {"lora_A", "lora_B"}:
            raise SystemExit(f"Error: {module} has only {sorted(parts)}")
        a, b = parts["lora_A"], parts["lora_B"]
        if len(a) != 2 or len(b) != 2 or a[0] != rank or b[1] != rank:
            raise SystemExit(f"Error: {module} has lora_A {tuple(a)} and lora_B {tuple(b)}, not [rank={rank}, in] and [out, rank]")
    return adapter


@click.command()
@click.option("--ckpt", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--out_dir", required=True, type=click.Path(path_type=Path))
@click.option("--base_model", required=True)
@click.option("--rank", required=True, type=int)
@click.option("--alpha", required=True, type=float)
@click.option("--dropout", default=0.0, type=float)
def main(ckpt, out_dir, base_model, rank, alpha, dropout):
    from safetensors.torch import save_file

    checkpoint = find_checkpoint(path=ckpt)
    adapter = read_adapter(ckpt=checkpoint, rank=rank)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(out_dir / "adapter_model.safetensors"))
    targets = sorted({key.split(".lora_")[0].rsplit(".", 1)[-1] for key in adapter})
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model,
        "r": rank, "lora_alpha": alpha, "lora_dropout": dropout, "bias": "none",
        "target_modules": targets, "modules_to_save": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "source.json").write_text(json.dumps({"checkpoint": str(checkpoint)}, indent=2))
    print(f"exported {len(adapter) // 2} adapted modules ({', '.join(targets)}) from {checkpoint} to {out_dir}")


if __name__ == "__main__":
    main()
