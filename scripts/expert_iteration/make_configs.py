"""Write the eval configs one expert-iteration stage needs: one per opponent rule.

    python scripts/expert_iteration/make_configs.py --out_dir D --kind gen|eval --model M --rules r1,r2 \
        --games N --rounds R [--copycat_games N] [--skip S] [--temperature T] ...

Prints one line per config: "<rule> <config path>". The rule copy_last is written as the
standard "copycat_rps" game (the same game; its results compare directly with earlier copycat
baselines), every other rule as "hidden_rule_rps". Seeds are skip..skip+N-1: generation uses
seeds from 0, evaluation of the training rules uses fresh seeds (--skip), so no evaluated game
was ever played for training data. Prompts never change: prediction_mode stays off.
"""

from pathlib import Path

import click

from self_play.games.hidden_rule_rps import RULES

TEMPLATE = """# Written by scripts/expert_iteration/make_configs.py ({kind}, rule {rule}).
model = "{model}"
push = false
max_concurrent = {max_concurrent}

[select]
limit = {games}
skip = {skip}

[client]
type = "eval"
base_url = "http://localhost:8000/v1"
api_key_var = "VLLM_API_KEY"

[sampling]
temperature = {temperature}
top_p = {top_p}
top_k = {top_k}
max_tokens = {max_tokens}

[env]
game = "{game}"
prediction_mode = "off"

[env.game_config]
rounds = {rounds}
{rule_line}
[env.taskset]
id = "self-play-verifiers"
"""


@click.command()
@click.option("--out_dir", required=True, type=click.Path(path_type=Path))
@click.option("--kind", required=True, type=click.Choice(["gen", "eval"]))
@click.option("--model", required=True)
@click.option("--rules", required=True, help="Comma-separated rule names (self_play.games.hidden_rule_rps.RULES).")
@click.option("--games", required=True, type=int, help="Games per rule.")
@click.option("--copycat_games", default=None, type=int, help="Games for copy_last, if different from --games.")
@click.option("--rounds", required=True, type=int)
@click.option("--skip", default=0, type=int, help="First seed (copycat_rps always starts at 0).")
@click.option("--temperature", default=1.0, type=float)
@click.option("--top_p", default=0.95, type=float)
@click.option("--top_k", default=64, type=int)
@click.option("--max_tokens", default=2048, type=int)
@click.option("--max_concurrent", default=4, type=int)
@click.option("--tag", default="", help="Appended to config file names (e.g. the model being evaluated).")
def main(out_dir, kind, model, rules, games, copycat_games, rounds, skip, temperature, top_p, top_k, max_tokens, max_concurrent, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    for rule in [r.strip() for r in rules.split(",") if r.strip()]:
        if rule not in RULES:
            raise click.BadParameter(f"unknown rule {rule!r}; known: {sorted(RULES)}")
        copycat = rule == "copy_last"
        text = TEMPLATE.format(
            kind=kind, rule=rule, model=model, max_concurrent=max_concurrent,
            games=copycat_games if copycat and copycat_games else games,
            skip=0 if copycat else skip,
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens,
            game="copycat_rps" if copycat else "hidden_rule_rps", rounds=rounds,
            rule_line="" if copycat else f'rule = "{rule}"\n',
        )
        path = out_dir / f"{kind}_{rule}{'_' + tag if tag else ''}.toml"
        path.write_text(text)
        print(rule, path)


if __name__ == "__main__":
    main()
