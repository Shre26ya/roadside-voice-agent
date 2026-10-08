from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

REQUIRED = ["caller_name", "location", "vehicle", "issue", "is_safe"]
PROMPTS = {
    "caller_name": "Ask for the caller's name.",
    "location": "Ask for their exact location: street, city, or landmark.",
    "vehicle": "Ask for the vehicle make, model, and color.",
    "issue": "Ask what is wrong with the vehicle.",
    "is_safe": "Ask whether they are in a safe place, away from traffic.",
}


class IntakeState(TypedDict, total=False):
    caller_name: str | None
    location: str | None
    vehicle: str | None
    issue: str | None
    is_safe: bool | None
    confirmed: bool
    dispatched: bool
    handoff_requested: bool
    handoff_failed: bool
    caller_phone: str | None
    created_at: str
    next_action: str
    instruction: str


def validate(state: IntakeState) -> IntakeState:
    """Reject obviously bad values so the LLM re-asks."""
    problems = []
    name = state.get("caller_name")
    if name and len(name.split()) > 4:
        problems.append("caller_name looks like a sentence, not a name")
    loc = state.get("location")
    if loc and len(loc) < 4:
        problems.append("location is too short to dispatch to")
    if problems:
        return {**state, "next_action": "reask", "instruction": "; ".join(problems) + ". Ask again politely."}
    return state


def decide(state: IntakeState) -> IntakeState:
    if state.get("handoff_requested") and not state.get("handoff_failed") and not state.get("dispatched"):
        return {**state, "next_action": "handoff", "instruction": "Call transfer_to_human now."}
    if state.get("is_safe") is False and not state.get("handoff_failed"):
        return {**state, "next_action": "handoff",
                "instruction": "Caller is not safe. Tell them to move away from traffic if possible and call transfer_to_human."}
    missing = [f for f in REQUIRED if state.get(f) is None]
    if missing:
        return {**state, "next_action": "ask", "instruction": PROMPTS[missing[0]]}
    if not state.get("confirmed"):
        return {**state, "next_action": "readback",
                "instruction": "All fields collected. Read them back in one short sentence and ask the caller to confirm. "
                               "When they say yes, call confirm_details. If they also ask for something else in the same breath, handle both."}
    if not state.get("dispatched"):
        return {**state, "next_action": "dispatch", "instruction": "Call confirm_and_dispatch now."}
    return {**state, "next_action": "done", "instruction": "Tell the caller help is on the way and end the call politely."}


def route(state: IntakeState) -> Literal["decide", "__end__"]:
    return END if state.get("next_action") == "reask" else "decide"


builder = StateGraph(IntakeState)
builder.add_node("validate", validate)
builder.add_node("decide", decide)
builder.add_edge(START, "validate")
builder.add_conditional_edges("validate", route)
builder.add_edge("decide", END)
intake_graph = builder.compile()


def advance(state: dict) -> IntakeState:
    """Run the graph once and return the updated state with next_action and instruction."""
    allowed = IntakeState.__annotations__.keys()
    clean = {k: v for k, v in state.items() if k in allowed}
    result = intake_graph.invoke({**clean, "next_action": "", "instruction": ""})
    result.setdefault("next_action", "unknown")
    result.setdefault("instruction", "Continue the intake.")
    return result