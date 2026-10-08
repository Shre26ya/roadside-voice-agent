import json
import logging
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession, RunContext, function_tool
from livekit.plugins import anthropic, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

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

Collect these five things, in order, and call the matching tool as soon as you learn each one:
1. Caller's name -> set_caller_name
2. Exact location (street, city, landmark) -> set_location
3. Vehicle make, model, color -> set_vehicle
4. What is wrong with the vehicle -> set_issue
5. Whether the caller is in a safe place -> set_safety

When all five are recorded, read them back and ask the caller to confirm.
When they confirm, call confirm_and_dispatch and tell them help is on the way.

If the caller asks for a human, call transfer_to_human. Only say you are
transferring if that tool reports success. Otherwise, apologize and continue.
Never claim to have done something you did not call a tool for."""


@dataclass
class IntakeRecord:
    caller_name: str | None = None
    location: str | None = None
    vehicle: str | None = None
    issue: str | None = None
    is_safe: bool | None = None
    dispatched: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def missing(self) -> list[str]:
        return [k for k in ("caller_name", "location", "vehicle", "issue", "is_safe")
                if getattr(self, k) is None]


class Dispatcher(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)
        self.record = IntakeRecord()

    @function_tool()
    async def set_caller_name(self, context: RunContext, name: str) -> str:
        """Record the caller's name."""
        self.record.caller_name = name
        log.info("tool set_caller_name %s", name)
        return f"Recorded name: {name}. Still missing: {self.record.missing()}"

    @function_tool()
    async def set_location(self, context: RunContext, location: str) -> str:
        """Record the caller's location: street, city, or landmark."""
        self.record.location = location
        log.info("tool set_location %s", location)
        return f"Recorded location: {location}. Still missing: {self.record.missing()}"

    @function_tool()
    async def set_vehicle(self, context: RunContext, vehicle: str) -> str:
        """Record the vehicle description: make, model, and color."""
        self.record.vehicle = vehicle
        log.info("tool set_vehicle %s", vehicle)
        return f"Recorded vehicle: {vehicle}. Still missing: {self.record.missing()}"

    @function_tool()
    async def set_issue(self, context: RunContext, issue: str) -> str:
        """Record what is wrong with the vehicle."""
        self.record.issue = issue
        log.info("tool set_issue %s", issue)
        return f"Recorded issue: {issue}. Still missing: {self.record.missing()}"

    @function_tool()
    async def set_safety(self, context: RunContext, is_safe: bool) -> str:
        """Record whether the caller is currently in a safe place."""
        self.record.is_safe = is_safe
        log.info("tool set_safety %s", is_safe)
        return f"Recorded safety: {is_safe}. Still missing: {self.record.missing()}"

    @function_tool()
    async def confirm_and_dispatch(self, context: RunContext) -> str:
        """Call only after the caller confirms all details. Creates the dispatch record."""
        missing = self.record.missing()
        if missing:
            return f"Cannot dispatch, missing fields: {missing}. Ask for them."
        self.record.dispatched = True
        path = Path("dispatch") / f"{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(asdict(self.record), indent=2))
        log.info("dispatch record written %s", path)
        return "Dispatch created. Tell the caller help is on the way."

    @function_tool()
    async def transfer_to_human(self, context: RunContext, reason: str) -> str:
        """Transfer the caller to a human dispatcher. Returns whether it succeeded."""
        log.info("tool transfer_to_human reason=%s", reason)
        return "FAILED: no human dispatcher is connected yet. Apologize and continue helping."


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", language="en"),
        llm=anthropic.LLM(model="claude-sonnet-5"),
        tts=cartesia.TTS(),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )
    await session.start(room=ctx.room, agent=Dispatcher())
    await session.generate_reply(
        instructions="Greet the caller as Sam from QuickTow roadside assistance and ask for their name."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))