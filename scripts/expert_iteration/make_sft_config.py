"""Write the prime-rl LoRA SFT config for an expert-iteration dataset.

    python scripts/expert_iteration/make_sft_config.py --data_dir D --out sft.toml --output_dir O \
        [--model google/gemma-4-31b-it] [--epochs 1] [--max_steps N] ...

max_steps (prime-rl SFT has no epochs; its loader cycles) is computed from the dataset's token
count (D/stats.json): epochs * train_tokens / (batch_size * seq_len * packing_efficiency).
Each step packs batch_size rows of seq_len tokens; only completion tokens are trained.

Gemma 4 needs: renderer "gemma4" (prime-rl's typed renderer; SFT refuses the default one), and
the fused LM head off (the model softcaps its final logits). The frozen base and the LoRA
weights are held in bfloat16 (as prime-rl's LoRA SFT example does), halving memory against the
float32 default. Checkpoints hold the whole model (about 62 GB in bfloat16), so only the latest
is kept; one is written every --ckpt_interval steps (default: a quarter of the run, at least 25
steps) so an interrupted run resumes from it, and one at the end.
"""

import json
import math
from pathlib import Path

import click

TEMPLATE = """# Written by scripts/expert_iteration/make_sft_config.py
max_steps = {max_steps}
output_dir = "{output_dir}"
dashboard = false

[run]
name = "{run_name}"

[deployment]
num_train_gpus = {num_gpus}
gpus_per_node = {num_gpus}

[model]
name = "{model}"
fused_lm_head_token_chunk_size = "disabled"
optimization_dtype = "bfloat16"
reduce_dtype = "bfloat16"

[model.lora]
rank = {rank}
alpha = {alpha}
dropout = 0.0
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

[renderer]
{renderer}

[data]
type = "sft"
name = "{data_dir}"
seq_len = {seq_len}
batch_size = {batch_size}
micro_batch_size = 1

[data.loss_mask]
assistant = true

[optim]
lr = {lr}
max_norm = 1.0

[scheduler]
{scheduler}

[ckpt]
keep_last = 1
{ckpt_interval}{val}"""


@click.command()
@click.option("--data_dir", required=True, type=click.Path(exists=True, path_type=Path), help="select_sft_data.py's --out_dir.")
@click.option("--out", required=True, type=click.Path(path_type=Path))
@click.option("--output_dir", required=True, type=click.Path(path_type=Path), help="prime-rl output dir (checkpoints, logs).")
@click.option("--run_name", default="ei-sft")
@click.option("--model", default="google/gemma-4-31b-it")
@click.option("--renderer", default="gemma4", help='prime-rl renderer name ("auto" for models in its renderer map).')
@click.option("--num_gpus", default=2)
@click.option("--rank", default=16)
@click.option("--alpha", default=32.0)
@click.option("--lr", default=1e-4)
@click.option("--seq_len", default=8192)
@click.option("--batch_size", default=16, help="Packed rows of seq_len tokens per step.")
@click.option("--epochs", default=1.0)
@click.option("--packing_efficiency", default=0.9)
@click.option("--max_steps", "max_steps_override", default=None, type=int, help="Overrides the computed step count.")
@click.option("--val_interval", default=50)
@click.option("--ckpt_interval", default=0, help="Steps between checkpoints; 0: a quarter of the run, at least 25.")
def main(data_dir, out, output_dir, run_name, model, renderer, num_gpus, rank, alpha, lr, seq_len, batch_size, epochs,
         packing_efficiency, max_steps_override, val_interval, ckpt_interval):
    if batch_size % num_gpus:
        raise click.BadParameter(f"batch_size {batch_size} must be divisible by num_gpus {num_gpus}")
    stats = json.loads((data_dir / "stats.json").read_text())
    if stats["max_example_tokens"] > seq_len:
        raise click.BadParameter(f"an example has {stats['max_example_tokens']} tokens, more than seq_len {seq_len}")
    computed = max(1, math.ceil(epochs * stats["train_tokens"] / (batch_size * seq_len * packing_efficiency)))
    max_steps = max_steps_override or computed
    warmup = min(10, max_steps // 10)
    interval = ckpt_interval or max(25, math.ceil(max_steps / 4))
    ckpt_line = f"interval = {interval}\n" if interval < max_steps else ""
    scheduler = (f'type = "cosine"\nwarmup_steps = {warmup}\nmin_lr = {lr / 10}' if max_steps >= 20 else 'type = "constant"')
    renderer_block = 'name = "gemma4"\nenable_thinking = false' if renderer == "gemma4" else f'name = "{renderer}"'
    val = ""
    if stats["validation_examples"] > 0 and max_steps > val_interval:
        val = (f'\n[val]\ninterval = {val_interval}\n\n[val.data]\ntype = "sft"\nname = "{data_dir / "dataset"}"\nsplits = ["validation"]\n'
               f"seq_len = {seq_len}\nbatch_size = {num_gpus}\nmicro_batch_size = 1\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.format(
        max_steps=max_steps, output_dir=output_dir, run_name=run_name, num_gpus=num_gpus, model=model, rank=rank,
        alpha=alpha, renderer=renderer_block, data_dir=data_dir / "dataset", seq_len=seq_len, batch_size=batch_size, lr=lr,
        scheduler=scheduler, val=val, ckpt_interval=ckpt_line,
    ))
    print(json.dumps({"config": str(out), "max_steps": max_steps, "computed_max_steps": computed,
                      "train_tokens": stats["train_tokens"], "tokens_per_step": batch_size * seq_len}))


if __name__ == "__main__":
    main()
