"""Compare the base model with its SFT adapter, rule by rule (the last expert-iteration stage).

    python scripts/expert_iteration/summarize.py --runs_dir R --name EXP --rules r1,r2 --heldout copy_last \
        --out_json summary.json --out_md summary.md

For each rule, runs <EXP>-eval-base-<rule> and <EXP>-eval-<sft_arm>-<rule> are paired game by game
on matching seeds, after the strict checks of scripts/analyze_copycat_games.py (both complete,
each its run's own traces.jsonl, provenance recorded, and the same condition except
agents.seat0.model). A pair that fails them is reported, never compared. Rounds won and mean
round payoff are over rounds 2+ (round 1 is chance against every rule).
"""

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_copycat_games import episode_metrics, load_episodes, pair_problems, paired, summarize  # noqa: E402

VARY = ("agents.seat0.model",)


def rule_summary(*, runs_dir: Path, name: str, rule: str, sft_arm: str):
    paths = {"base": runs_dir / f"{name}-eval-base-{rule}" / "traces.jsonl", "sft": runs_dir / f"{name}-eval-{sft_arm}-{rule}" / "traces.jsonl"}
    problems = pair_problems(a=str(paths["base"]), b=str(paths["sft"]), vary=VARY)
    if problems:
        return {"rule": rule, "problems": problems}
    rows = {arm: [episode_metrics(episode=e) for e in load_episodes(path=str(path))[0]] for arm, path in paths.items()}
    return {
        "rule": rule,
        "problems": [],
        "base": summarize(rows=rows["base"]),
        "sft": summarize(rows=rows["sft"]),
        "paired_sft_minus_base": paired(rows_a=rows["base"], rows_b=rows["sft"]),
    }


def pct(value):
    return "–" if value is None else f"{100 * value:.1f}%"


def markdown(*, results, heldout):
    lines = [
        "| Rule | Games | Base: rounds won (2+) | SFT: rounds won (2+) | SFT − base (95% CI) | Base payoff (2+) | SFT payoff (2+) | Base / SFT games W-T-L |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        label = f"**{r['rule']}** (held out)" if r["rule"] in heldout else r["rule"]
        if r["problems"]:
            lines.append(f"| {label} | – | not compared: {'; '.join(r['problems'])[:300]} | | | | | |")
            continue
        diff = r["paired_sft_minus_base"]["round_win_frac_2plus"]
        lines.append(
            f"| {label} | {r['paired_sft_minus_base']['matched_games']} | {pct(r['base']['round_win_frac_2plus'])} | "
            f"{pct(r['sft']['round_win_frac_2plus'])} | {100 * diff['mean_diff']:+.1f} pts "
            f"[{100 * diff['ci95'][0]:+.1f}, {100 * diff['ci95'][1]:+.1f}] | {r['base']['mean_round_payoff_2plus']:.3f} | "
            f"{r['sft']['mean_round_payoff_2plus']:.3f} | {r['base']['match_W/T/L']} / {r['sft']['match_W/T/L']} |"
        )
    return "\n".join(lines) + "\n\nRounds won is over rounds 2+ (random play wins 33%). Payoff: win 1, draw 0.5, loss 0.\n"


@click.command()
@click.option("--runs_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--sft_arm", default="sft", help="The SFT runs' arm label (the pipeline uses sft-<adapter hash>).")
@click.option("--rules", required=True)
@click.option("--heldout", default="copy_last", help="Comma-separated rules that were never trained on.")
@click.option("--out_json", required=True, type=click.Path(path_type=Path))
@click.option("--out_md", required=True, type=click.Path(path_type=Path))
def main(runs_dir, name, sft_arm, rules, heldout, out_json, out_md):
    heldout = {r for r in heldout.split(",") if r}
    results = [rule_summary(runs_dir=runs_dir, name=name, rule=rule, sft_arm=sft_arm) for rule in rules.split(",") if rule]
    out_json.write_text(json.dumps({"name": name, "sft_arm": sft_arm, "heldout": sorted(heldout), "results": results}, indent=1))
    table = markdown(results=results, heldout=heldout)
    out_md.write_text(f"# {name}: base vs SFT ({sft_arm})\n\n{table}")
    print(table)
    if any(r["problems"] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
