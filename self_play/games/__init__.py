"""Every game, by the name a run config uses (`[env] game = "..."`)."""

from self_play.game import Game
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.hidden_rule_rps import HiddenRuleRPSGame

GAMES: dict[str, type[Game]] = {game.name: game for game in (CopycatRPSGame, HiddenRuleRPSGame)}
