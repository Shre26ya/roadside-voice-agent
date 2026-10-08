from dotenv import load_dotenv
from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import anthropic, cartesia, deepgram, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

INSTRUCTIONS = """You are a roadside assistance dispatcher for QuickTow.
Speak in short, calm sentences. One question at a time.
Your job is to collect five things from the caller:
1. Their name
2. Their exact location (street, city, landmark)
3. Vehicle make, model, and color
4. What is wrong with the vehicle
5. Whether they are in a safe place
Once you have all five, read them back and ask the caller to confirm.
If the caller asks for a human, say you will transfer them."""


class Dispatcher(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=INSTRUCTIONS)


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
        instructions="Greet the caller, say you are QuickTow roadside assistance, and ask for their name."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))