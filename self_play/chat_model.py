"""The one model interface the engine talks to.

Games and players only ever call `ChatModel.complete`. Real runs back it with a verifiers
interaction (`InteractionChatModel`); unit tests use `ScriptedChatModel`. This module must
not import verifiers: `InteractionChatModel` only calls `.turn()` on what it is given.
"""

from abc import ABC, abstractmethod

STOP_STRING = "[STOP]"
"""Models are told to end every reply with this. Seats also send it as a stop sequence."""


def strip_stop(*, text: str) -> str:
    """The reply up to the first `STOP_STRING` (all of it when absent), stripped."""
    return text.split(STOP_STRING)[0].strip()


class ChatModel(ABC):
    last_call: dict | None = None
    """Provider details of the latest completed `complete` (see `provider_call_info`), when
    the backend reports them; None otherwise. The runner stores them with the call's record
    entry, so they survive checkpoint resume."""

    @abstractmethod
    async def complete(self, *, messages: list[dict]) -> str:
        """Reply to `messages`, which are the whole context for this call (the chat never
        grows between calls). Returns the raw reply text."""


def provider_call_info(*, calls: list) -> dict | None:
    """Token usage, finish reason and timing of the provider exchanges behind one model call
    (normally one), read by attribute from verifiers `ModelCall`s: summed usage, the last
    finish reason, first start to last end. None when there were none."""
    if not calls:
        return None
    usage = [call.usage for call in calls if getattr(call, "usage", None) is not None]
    total = lambda field: sum(getattr(u, field, None) or 0 for u in usage) if usage else None
    times = [call.time for call in calls if getattr(call, "time", None) is not None]
    return {
        "provider_calls": len(calls),
        "prompt_tokens": total("prompt_tokens"),
        "cached_input_tokens": total("cached_input_tokens"),
        "completion_tokens": total("completion_tokens"),
        "finish_reason": getattr(calls[-1], "finish_reason", None),
        "start": getattr(times[0], "start", None) if times else None,
        "end": getattr(times[-1], "end", None) if times else None,
    }


class ModelCallFailed(RuntimeError):
    """A model call did not produce a reply. Never papered over: the game stops here."""


class InteractionChatModel(ChatModel):
    """A `ChatModel` backed by one seat's verifiers interaction.

    Each `complete` is one `interaction.turn`; with `StatelessHarness` the messages sent are
    the call's whole context. If the seat's run ended instead of replying (a token or turn
    limit, a timeout), the call raises `ModelCallFailed`, which fails the episode; with
    checkpointing on, a resumed run continues the game from before this call.
    """

    def __init__(self, *, interaction):
        self._interaction = interaction

    async def complete(self, *, messages: list[dict]) -> str:
        calls = getattr(getattr(self._interaction, "trace", None), "calls", None)
        before = len(calls) if calls is not None else 0
        segment = await self._interaction.turn(messages)
        if segment.terminated:
            raise ModelCallFailed("the seat's run ended (a limit, a timeout) instead of replying")
        # verifiers appends each provider exchange to trace.calls before the turn returns
        self.last_call = provider_call_info(calls=list(calls[before:])) if calls is not None else None
        return segment.last_reply


class ScriptedChatModel(ChatModel):
    """Replies with the given strings in order; records every messages list it was sent."""

    def __init__(self, *, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete(self, *, messages: list[dict]) -> str:
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("ScriptedChatModel ran out of replies")
        return self._replies.pop(0)
