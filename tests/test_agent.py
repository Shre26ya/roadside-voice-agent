import pytest
from livekit.agents import AgentSession
from livekit.plugins import anthropic

from agent import Dispatcher

LLM = "claude-sonnet-5"
JUDGE = anthropic.LLM(model="claude-haiku-4-5-20251001")


@pytest.mark.asyncio
async def test_happy_path_collects_all_fields_and_dispatches():
    async with AgentSession(llm=anthropic.LLM(model=LLM)) as session:
        agent = Dispatcher()
        await session.start(agent)

        await session.run(user_input="Hi, my name is Shreya")
        await session.run(user_input="I'm at the Target on Route 22 in Union, New Jersey")
        await session.run(user_input="It's a blue Honda Civic")
        await session.run(user_input="Flat tire")
        await session.run(user_input="Yes, I'm safe")
        result = await session.run(user_input="Yes, that's all correct")

        assert agent.record.caller_name == "Shreya"
        assert agent.record.confirmed is True
        assert agent.record.dispatched is True
        await (
            result.expect.contains_message(role="assistant")
            .judge(JUDGE, intent="Tells the caller help is on the way")
        )


@pytest.mark.asyncio
async def test_compound_turn_confirms_and_handles_handoff():
    async with AgentSession(llm=anthropic.LLM(model=LLM)) as session:
        agent = Dispatcher()
        await session.start(agent)
        for turn in ["Shreya", "Journal Square, Jersey City", "blue Honda Civic", "flat tire", "yes I'm safe"]:
            await session.run(user_input=turn)
        result = await session.run(user_input="Yes that's correct, and can I talk to a person?")

        assert agent.record.confirmed is True, "compound turn dropped the confirmation"
        assert agent.record.handoff_requested is True, "compound turn dropped the handoff request"
        await (
            result.expect.contains_message(role="assistant")
            .judge(JUDGE, intent="Does not claim a transfer happened; apologizes that no human is available")
        )


@pytest.mark.asyncio
async def test_off_topic_does_not_create_fields():
    async with AgentSession(llm=anthropic.LLM(model=LLM)) as session:
        agent = Dispatcher()
        await session.start(agent)
        result = await session.run(user_input="Can you help me with a math problem?")
        assert agent.record.caller_name is None
        await (
            result.expect.contains_message(role="assistant")
            .judge(JUDGE, intent="Politely declines and asks for the caller's name")
        )


@pytest.mark.asyncio
async def test_unsafe_caller_triggers_handoff_attempt():
    async with AgentSession(llm=anthropic.LLM(model=LLM)) as session:
        agent = Dispatcher()
        await session.start(agent)
        for turn in ["Shreya", "Route 22 in Union", "blue Honda Civic", "I was rear-ended"]:
            await session.run(user_input=turn)
        await session.run(user_input="No, I'm standing on the shoulder next to traffic")
        assert agent.record.is_safe is False
        assert agent.record.handoff_requested is True, "unsafe caller did not trigger transfer_to_human"