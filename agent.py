import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import anthropic, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from intake_graph import advance

load_dotenv()

Path("logs").mkdir(exist_ok=True)
Path("dispatch").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.FileHandler(f"logs/{datetime.now():%Y%m%d_%H%M%S}.log")],
)
log = logging.getLogger("dispatcher")

INSTRUCTIONS = """You are Sam, a roadside assistance dispatcher for QuickTow.
Speak in short, calm sentences. Ask one question at a time.

Every tool returns a line starting with NEXT. Always follow that NEXT instruction.
If the caller says two things in one turn, call a tool for each.

Tools:
- set_caller_name, set_location, set_vehicle, set_issue, set_safety: call as soon as you learn each detail
- confirm_details: call when the caller says the read-back is correct
- confirm_and_dispatch: call only when a NEXT instruction tells you to
- transfer_to_human: call when the caller asks for a person or a NEXT instruction tells you to

Only say you are transferring if transfer_to_human reports success.
Never claim to have done something you did not call a tool for."""


@dataclass
class IntakeRecord:
    caller_name: str | None = None
    location: str | None = None
    vehicle: str | None = None
    issue: str | None = None
    is_safe: bool | None = None
    confirmed: bool = False
    dispatched: bool = False
    handoff_requested: bool = False
    handoff_failed: bool = False
    caller_phone: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class Dispatcher(Agent):
    def __init__(self, caller_phone: str | None = None) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.record = IntakeRecord(caller_phone=caller_phone)

    def _advance(self) -> str:
        state = advance(asdict(self.record))
        log.info("graph next_action=%s instruction=%s", state.get("next_action"), state.get("instruction"))
        return f"NEXT: {state.get('instruction')}"

    @function_tool()
    async def set_caller_name(self, context: RunContext, name: str) -> str:
        """Record the caller's name."""
        self.record.caller_name = name
        log.info("tool set_caller_name %s", name)
        return self._advance()

    @function_tool()
    async def set_location(self, context: RunContext, location: str) -> str:
        """Record the caller's location: street, city, or landmark."""
        self.record.location = location
        log.info("tool set_location %s", location)
        return self._advance()

    @function_tool()
    async def set_vehicle(self, context: RunContext, vehicle: str) -> str:
        """Record the vehicle description: make, model, and color."""
        self.record.vehicle = vehicle
        log.info("tool set_vehicle %s", vehicle)
        return self._advance()

    @function_tool()
    async def set_issue(self, context: RunContext, issue: str) -> str:
        """Record what is wrong with the vehicle."""
        self.record.issue = issue
        log.info("tool set_issue %s", issue)
        return self._advance()

    @function_tool()
    async def set_safety(self, context: RunContext, is_safe: bool) -> str:
        """Record whether the caller is currently in a safe place."""
        self.record.is_safe = is_safe
        log.info("tool set_safety %s", is_safe)
        return self._advance()

    @function_tool()
    async def confirm_details(self, context: RunContext) -> str:
        """Call when the caller confirms the read-back is correct."""
        self.record.confirmed = True
        log.info("tool confirm_details")
        return self._advance()

    @function_tool()
    async def confirm_and_dispatch(self, context: RunContext) -> str:
        """Create the dispatch record. Call only when a NEXT instruction says to."""
        state = advance(asdict(self.record))
        if state.get("next_action") != "dispatch":
            log.info("dispatch refused, next_action=%s", state.get("next_action"))
            return f"NEXT: {state.get('instruction')}"
        self.record.dispatched = True
        path = Path("dispatch") / f"{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(asdict(self.record), indent=2))
        log.info("dispatch record written %s", path)
        return self._advance()

    @function_tool()
    async def transfer_to_human(self, context: RunContext, reason: str) -> str:
        """Transfer the caller to a human dispatcher. Returns whether it succeeded."""
        self.record.handoff_requested = True
        log.info("tool transfer_to_human reason=%s", reason)
        self.record.handoff_failed = True
        return "FAILED: no human dispatcher is connected yet. Apologize briefly, then " + self._advance()


async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    caller_phone = participant.identity.removeprefix("sip_") if participant.identity.startswith("sip_") else None
    log.info("call from %s", caller_phone)

    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=anthropic.LLM(model="claude-sonnet-5"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(room=ctx.room, agent=Dispatcher(caller_phone=caller_phone))
    await session.generate_reply(
        instructions="Greet the caller as Sam from QuickTow roadside assistance and ask for their name."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, agent_name="roadside"))