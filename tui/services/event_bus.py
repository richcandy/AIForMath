from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class SearchEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


class SearchEventHandler(Protocol):
    def __call__(self, event: SearchEvent) -> None:
        ...


def coerce_search_event(raw_event: dict[str, Any]) -> SearchEvent:
    event_type = str(raw_event.get("type") or "unknown")
    payload = {key: value for key, value in raw_event.items() if key != "type"}
    return SearchEvent(type=event_type, payload=payload)


def format_event_message(event: SearchEvent) -> str:
    if event.type == "startup_stage":
        return str(event.payload.get("message") or "startup")
    if event.type == "root_state_loaded":
        return "Root goal loaded"
    if event.type == "layer_started":
        depth = event.payload.get("depth")
        beam_size = event.payload.get("beam_size")
        return f"Layer {depth} started (beam={beam_size})"
    if event.type == "layer_generation_completed":
        depth = event.payload.get("depth")
        count = event.payload.get("generated_tactics")
        return f"Layer {depth} generated {count} tactics"
    if event.type == "node_verification_completed":
        depth = event.payload.get("depth")
        results = event.payload.get("results") or []
        return f"Depth {depth} verified {len(results)} tactics"
    if event.type == "layer_completed":
        depth = event.payload.get("depth")
        beam_out = event.payload.get("beam_out")
        return f"Layer {depth} completed (beam_out={beam_out})"
    if event.type == "solution_found":
        return "Solution found"
    if event.type == "search_finished":
        failure = event.payload.get("failure")
        return f"Search finished ({failure})"
    return event.type
