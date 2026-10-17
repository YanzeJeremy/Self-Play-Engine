"""Compare the base model with its SFT adapter, rule by rule (the last expert-iteration stage).

    python scripts/expert_iteration/summarize.py --runs_dir R --name EXP --rules r1,r2 --heldout copy_last \
        --out_json summary.json --out_md summary.md

For each rule and horizon, runs <EXP>-eval-<base_arm>-<rule><suffix> and <EXP>-eval-<sft_arm>-<rule><suffix>
(suffix "" for the main evaluation, e.g. "-r20" for --long_suffix) are paired game by game
on matching seeds, after the strict checks of scripts/analyze_copycat_games.py (both complete,
each its run's own traces.jsonl, provenance recorded, and the same condition except
agents.seat0.model). A pair that fails them is reported, never compared. Rounds won and mean
round payoff are over rounds 2+ (round 1 is chance against every rule). Each horizon also gets
the rounds-won rate round by round for both models, to tell faster discovery from better
early moves.

--diag lists diagnostic conditions as name:rule (runs <EXP>-eval-<arm>-<rule>-diag-<name>), scored
from the games themselves: rounds won over rounds 2+ that were not forced (an intervention's
round is the opponent's doing, not the player's), recovery (wins in the two rounds after each
forced round), and for a rule switch the rounds won before and after it and in the first five
rounds after it; base and SFT are paired by seed with 95% bootstrap intervals over games.
"""

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_copycat_games import bootstrap_interval, episode_metrics, load_episodes, pair_problems, paired, summarize  # noqa: E402

VARY = ("agents.seat0.model",)


def per_round_wins(*, episodes):
    """Share of games won at each round index (from 1)."""
    results = [[e["content"] for e in ep["game"]["record"] if e["kind"] == "event"] for ep in episodes]
    rounds = max((len(r) for r in results), default=0)
    return [sum("the player wins" in r[i] for r in results if len(r) > i) / sum(len(r) > i for r in results) for i in range(rounds)]


def rule_summary(*, runs_dir: Path, name: str, rule: str, sft_arm: str, suffix: str = "", base_arm: str = "base"):
    paths = {"base": runs_dir / f"{name}-eval-{base_arm}-{rule}{suffix}" / "traces.jsonl",
             "sft": runs_dir / f"{name}-eval-{sft_arm}-{rule}{suffix}" / "traces.jsonl"}
    problems = pair_problems(a=str(paths["base"]), b=str(paths["sft"]), vary=VARY)
    if problems:
        return {"rule": rule, "problems": problems}
    episodes = {arm: load_episodes(path=str(path))[0] for arm, path in paths.items()}
    rows = {arm: [episode_metrics(episode=e) for e in eps] for arm, eps in episodes.items()}
    return {
        "rule": rule,
        "problems": [],
        "base": summarize(rows=rows["base"]),
        "sft": summarize(rows=rows["sft"]),
        "paired_sft_minus_base": paired(rows_a=rows["base"], rows_b=rows["sft"]),
        "per_round_wins": {arm: per_round_wins(episodes=eps) for arm, eps in episodes.items()},
    }


def game_rounds(*, game):
    """Per round: (won, forced) for the player."""
    rounds, played = [], 0
    forced_rounds = set()
    for entry in game["record"]:
        if entry["kind"] == "action" and entry["accepted"] and entry["action"] != "THINK" and entry["source"] == "forced":
            forced_rounds.add(played)
        elif entry["kind"] == "event":
            rounds.append(("the player wins" in entry["content"], played in forced_rounds))
            played += 1
    return rounds


def diagnostic_metrics(*, game):
    rounds = game_rounds(game=game)
    scored = [won for i, (won, forced) in enumerate(rounds) if i >= 1 and not forced]
    after_forced = [rounds[j][0] for i, (_, forced) in enumerate(rounds) if forced
                    for j in (i + 1, i + 2) if j < len(rounds) and not rounds[j][1]]
    metrics = {"rounds_won": sum(scored) / len(scored) if scored else None,
               "recovery": sum(after_forced) / len(after_forced) if after_forced else None}
    switch = game["game_config"].get("switch_round")
    if switch is not None:
        before = [won for i, (won, forced) in enumerate(rounds[:switch]) if i >= 1 and not forced]
        after = [won for won, forced in rounds[switch:] if not forced]
        metrics |= {"won_before_switch": sum(before) / len(before) if before else None,
                    "won_after_switch": sum(after) / len(after) if after else None,
                    "won_first5_after_switch": sum(w for w, f in rounds[switch:switch + 5] if not f)
                    / max(1, sum(1 for w, f in rounds[switch:switch + 5] if not f))}
    return metrics


def diagnostic_summary(*, runs_dir: Path, name: str, condition: str, rule: str, sft_arm: str, base_arm: str = "base"):
    suffix = f"-diag-{condition}"
    paths = {"base": runs_dir / f"{name}-eval-{base_arm}-{rule}{suffix}" / "traces.jsonl",
             "sft": runs_dir / f"{name}-eval-{sft_arm}-{rule}{suffix}" / "traces.jsonl"}
    problems = pair_problems(a=str(paths["base"]), b=str(paths["sft"]), vary=VARY)
    if problems:
        return {"condition": condition, "rule": rule, "problems": problems}
    per_seed = {arm: {ep["game"]["game_config"]["seed"]: diagnostic_metrics(game=ep["game"])
                      for ep in load_episodes(path=str(path))[0]} for arm, path in paths.items()}
    seeds = sorted(set(per_seed["base"]) & set(per_seed["sft"]))
    result = {"condition": condition, "rule": rule, "problems": [], "games": len(seeds), "metrics": {}}
    for metric in per_seed["base"][seeds[0]]:
        pairs = [(per_seed["base"][s][metric], per_seed["sft"][s][metric]) for s in seeds
                 if per_seed["base"][s][metric] is not None and per_seed["sft"][s][metric] is not None]
        if not pairs:
            continue
        diffs = [b2 - b1 for b1, b2 in pairs]
        result["metrics"][metric] = {"base": sum(p[0] for p in pairs) / len(pairs), "sft": sum(p[1] for p in pairs) / len(pairs),
                                     "diff": sum(diffs) / len(diffs), "ci95": list(bootstrap_interval(diffs=diffs)), "games": len(pairs)}
    episodes = {arm: load_episodes(path=str(path))[0] for arm, path in paths.items()}
    result["per_round_wins"] = {arm: per_round_wins(episodes=eps) for arm, eps in episodes.items()}
    return result


def diagnostics_markdown(*, results):
    lines = ["| Condition | Rule | Games | Metric | Base | SFT | SFT − base (95% CI) |", "|---|---|---|---|---|---|---|"]
    for r in results:
        if r["problems"]:
            lines.append(f"| {r['condition']} | {r['rule']} | – | not compared: {'; '.join(r['problems'])[:300]} | | | |")
            continue
        for metric, m in r["metrics"].items():
            lines.append(f"| {r['condition']} | {r['rule']} | {m['games']} | {metric} | {pct(m['base'])} | {pct(m['sft'])} | "
                         f"{100 * m['diff']:+.1f} pts [{100 * m['ci95'][0]:+.1f}, {100 * m['ci95'][1]:+.1f}] |")
    return ("\n".join(lines) + "\n\nrounds_won leaves out round 1 and forced rounds; recovery is the win rate in the two "
            "rounds after each forced round; switch metrics split the game at the unannounced rule change.\n")


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
    curves = ["", "Rounds won, round by round (base / SFT):", ""]
    for r in results:
        if not r["problems"]:
            pairs = zip(r["per_round_wins"]["base"], r["per_round_wins"]["sft"])
            curves.append(f"- {r['rule']}: " + "  ".join(f"r{i + 1} {100 * b:.0f}/{100 * t:.0f}" for i, (b, t) in enumerate(pairs)))
    return ("\n".join(lines) + "\n\nRounds won is over rounds 2+ (random play wins 33%). Payoff: win 1, draw 0.5, loss 0.\n"
            + "\n".join(curves) + "\n")


@click.command()
@click.option("--runs_dir", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--name", required=True)
@click.option("--sft_arm", default="sft", help="B: the runs' arm label compared (the pipeline uses <arm>-<adapter hash>).")
@click.option("--base_arm", default="base", help="A: the arm B is compared against (default the base model).")
@click.option("--rules", required=True)
@click.option("--heldout", default="copy_last", help="Comma-separated rules that were never trained on.")
@click.option("--out_json", required=True, type=click.Path(path_type=Path))
@click.option("--out_md", required=True, type=click.Path(path_type=Path))
@click.option("--long_suffix", default="", help='Run-name suffix of a second, longer evaluation (e.g. "-r20"); empty: none.')
@click.option("--long_label", default="20 rounds")
@click.option("--main_label", default="10 rounds")
@click.option("--diag", default="", help="Diagnostic conditions, comma-separated name:rule (runs ...-<rule>-diag-<name>).")
def main(runs_dir, name, sft_arm, base_arm, rules, heldout, out_json, out_md, long_suffix, long_label, main_label, diag):
    heldout = {r for r in heldout.split(",") if r}
    rule_list = [rule for rule in rules.split(",") if rule]
    horizons = {main_label: ""} | ({long_label: long_suffix} if long_suffix else {})
    results = {label: [rule_summary(runs_dir=runs_dir, name=name, rule=rule, sft_arm=sft_arm, suffix=suffix, base_arm=base_arm) for rule in rule_list]
               for label, suffix in horizons.items()}
    diagnostics = [diagnostic_summary(runs_dir=runs_dir, name=name, condition=c.split(":")[0], rule=c.split(":")[1], sft_arm=sft_arm, base_arm=base_arm)
                   for c in diag.split(",") if c]
    out_json.write_text(json.dumps({"name": name, "base_arm": base_arm, "sft_arm": sft_arm, "heldout": sorted(heldout), "results": results[main_label],
                                    "by_horizon": results, "diagnostics": diagnostics}, indent=1))
    text = "".join(f"## {label}\n\n{markdown(results=res, heldout=heldout)}\n" for label, res in results.items())
    if diagnostics:
        text += f"## Diagnostics\n\n{diagnostics_markdown(results=diagnostics)}\n"
    legend = f"Columns: \"Base\" = {base_arm}, \"SFT\" = {sft_arm}; differences are {sft_arm} minus {base_arm}, paired by seed.\n\n"
    out_md.write_text(f"# {name}: {sft_arm} vs {base_arm}\n\n{legend}{text}")
    print(text)
    if any(r["problems"] for res in results.values() for r in res) or any(d["problems"] for d in diagnostics):
        sys.exit(1)


if __name__ == "__main__":
    main()
