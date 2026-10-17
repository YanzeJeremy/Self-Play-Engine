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

Shared key/value projections (Gemma 4's `attention_k_eq_v` full-attention layers): there the
model has a k_proj and no v_proj, and the values are the k_proj output, so in training the
k_proj LoRA update changes both keys and values. vLLM packs q/k/v into one qkv_proj, loads the
k_proj weights into both the K and V slots, but applies LoRA per slot, and leaves a missing
v_proj adapter empty: the served values would lack the update. So for those layers (read from
the base model's config, by vLLM's own rule) the k_proj adapter is also written as the v_proj
adapter. source.json lists them.

Fidelity: after writing, the adapter is reloaded with vLLM's own LoRA loader (CPU) and, for every
adapted module, the update vLLM would add (B·A·x, its scaling folded in) is compared with the
trained update from the checkpoint ((alpha/rank)·B·A·x) on the same random inputs; for each
shared key/value layer, the served V-slot update is compared with the trained k_proj update (the
values' update in training). Any relative error above --tolerance (bfloat16 storage needs ~1e-2)
fails the export. The result is written to fidelity.json.
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


def shared_kv_layers(*, base_model: str) -> set[int]:
    """Layers whose values are their keys, by vLLM's rule for Gemma 4 (`_gemma4_layer_weights_mapper`):
    full-attention layers, before any KV-shared layers, when `attention_k_eq_v` is set."""
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(base_model, local_files_only=True)
    except Exception as error:
        raise SystemExit(f"Error: cannot read {base_model}'s config offline to find shared key/value layers: {error!r}")
    text = getattr(config, "text_config", config)
    if not getattr(text, "attention_k_eq_v", False):
        return set()
    first_kv_shared = text.num_hidden_layers - (getattr(text, "num_kv_shared_layers", 0) or 0)
    return {i for i, kind in enumerate(text.layer_types[:first_kv_shared]) if kind == "full_attention"}


def share_kv(*, adapter: dict, layers: set[int]) -> list[str]:
    """Write each shared layer's k_proj adapter as its v_proj adapter too; the modules written."""
    written = []
    for key in sorted(adapter):
        match = re.search(r"layers\.(\d+)\.self_attn\.k_proj\.(lora_[AB])\.weight$", key)
        if not match or int(match[1]) not in layers:
            continue
        v_key = key.replace(".self_attn.k_proj.", ".self_attn.v_proj.")
        if v_key in adapter:
            raise SystemExit(f"Error: shared key/value layer {match[1]} already has a v_proj adapter ({v_key})")
        adapter[v_key] = adapter[key].clone()
        if match[2] == "lora_A":
            written.append(v_key.removesuffix(".lora_A.weight"))
    return written


def read_adapter(*, ckpt: Path, rank: int, dtype="bfloat16") -> dict:
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
        tensor = state[key].float() if dtype is None else state[key].to(torch.bfloat16)
        adapter[f"{module}.{which}.weight"] = tensor.contiguous()
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
@click.option("--tolerance", default=0.03, type=float, help="Largest relative error allowed in the fidelity check.")
def main(ckpt, out_dir, base_model, rank, alpha, dropout, tolerance):
    from safetensors.torch import save_file

    checkpoint = find_checkpoint(path=ckpt)
    adapter = read_adapter(ckpt=checkpoint, rank=rank)
    trained = read_adapter(ckpt=checkpoint, rank=rank, dtype=None)
    shared = share_kv(adapter=adapter, layers=shared_kv_layers(base_model=base_model))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter, str(out_dir / "adapter_model.safetensors"))
    targets = sorted({key.split(".lora_")[0].rsplit(".", 1)[-1] for key in adapter})
    config = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model,
        "r": rank, "lora_alpha": alpha, "lora_dropout": dropout, "bias": "none",
        "target_modules": targets, "modules_to_save": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "source.json").write_text(json.dumps({"checkpoint": str(checkpoint), "shared_kv_v_proj_copied_from_k_proj": shared}, indent=2))
    fidelity = verify_export(out_dir=out_dir, trained=trained, shared=shared, rank=rank, alpha=alpha, targets=targets)
    (out_dir / "fidelity.json").write_text(json.dumps(fidelity, indent=2))
    if fidelity["max_relative_error"] > tolerance:
        raise SystemExit(f"Error: the exported adapter does not reproduce the trained update: worst relative error "
                         f"{fidelity['max_relative_error']:.4f} at {fidelity['worst']} (tolerance {tolerance}); see fidelity.json")
    print(f"exported {len(adapter) // 2} adapted modules ({', '.join(targets)}) from {checkpoint} to {out_dir}; "
          f"k_proj adapter also written as v_proj for {len(shared)} shared key/value layers; vLLM-loaded update matches the "
          f"checkpoint on {fidelity['modules_checked']} modules (worst relative error {fidelity['max_relative_error']:.4f})")


def verify_export(*, out_dir: Path, trained: dict, shared: list[str], rank: int, alpha: float, targets: list[str]) -> dict:
    """Reload the written adapter with vLLM's loader and compare each module's update with the checkpoint's."""
    import torch
    import vllm.lora.lora_model as vllm_lora_model
    from vllm.lora.peft_helper import PEFTHelper

    vllm_lora_model.PIN_MEMORY = False  # pinning needs a GPU; loading does not
    helper = PEFTHelper.from_local_dir(str(out_dir), max_position_embeddings=None)
    served = vllm_lora_model.LoRAModel.from_local_checkpoint(
        str(out_dir), expected_lora_modules=set(targets), peft_helper=helper, device="cpu", dtype=torch.float32)
    generator = torch.Generator().manual_seed(0)
    errors = {}
    checks = [(m, m) for m in {k.rsplit(".lora_", 1)[0] for k in trained}]
    checks += [(v, v.replace(".self_attn.v_proj", ".self_attn.k_proj")) for v in shared]  # served V slot vs trained values
    for served_module, trained_module in checks:
        a, b = trained[f"{trained_module}.lora_A.weight"], trained[f"{trained_module}.lora_B.weight"]
        x = torch.randn(a.shape[1], generator=generator)
        expected = (alpha / rank) * (b @ (a @ x))
        lora = served.get_lora(served_module)
        if lora is None:
            errors[served_module] = float("inf")
            continue
        lora.optimize()
        got = lora.lora_b @ (lora.lora_a @ x)
        errors[served_module] = float((got - expected).norm() / expected.norm().clamp(min=1e-12))
    worst = max(errors, key=errors.get)
    return {"modules_checked": len(errors), "max_relative_error": errors[worst], "worst": worst, "relative_errors": errors}


if __name__ == "__main__":
    main()
