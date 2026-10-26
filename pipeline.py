
"""Graph nodes and the LangGraph pipeline.

    START -> intake -> (missing info? -> clarify -> intake)
                    -> planning -> human_decision
                          approve -> writer -> END
                          revise  -> intake
                          inquire -> END
"""
import datetime
import logging
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from config import MODEL_NAME, require_env
from prompts import INTAKE_PROMPT, TRAVEL_AGENT_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT
from schemas import (
    FlightDetails,
    TravelDetails,
    TravelState,
    as_dict,
    get_missing,
    make_filename,
)
from tools import (
    airports_mcp_search_tool,
    filesystem_write_mcp_tool,
    flights_mcp_search_tool,
    tavily_mcp_search_tool,
)

logger = logging.getLogger("trip_planner")


# ---------------------------------------------------------------- Nodes
def make_intake_node(llm):
    chain = INTAKE_PROMPT | llm.with_structured_output(TravelDetails)

    async def intake(state: TravelState) -> dict:
        query = state.get("reframed_question") or state["question"]

        earlier = [
            m.content for m in (state.get("messages") or [])
            if isinstance(m, HumanMessage)
        ]
        conversation = "\n".join([*earlier, query])

        try:
            details = await chain.ainvoke({
                "today": datetime.date.today().isoformat(),
                "conversation": conversation,
            })
        except Exception as exc:
            logger.warning("Intake failed: %r", exc)
            return {
                "messages": [HumanMessage(content=query)],
                "reframed_question": None,
                "travel_framed_details": None,
                "missing_info": [
                    "exact dates (day, month and year); I could not read them"
                ],
            }

        return {
            "messages": [HumanMessage(content=query)],
            "reframed_question": None,
            "travel_framed_details": details,
            "missing_info": get_missing(details),
        }

    return intake


def clarify_node(state: TravelState) -> Command[Literal["intake"]]:
    payload = {
        "question": "I need a few more details before I can search:",
        "missing": state["missing_info"],
    }
    # Bad input re-asks (with an "error" key) instead of raising: a node that
    # raises after a resume leaves the thread stuck on the stale resume value.
    while True:
        reply = as_dict(interrupt(payload))
        message = (reply.get("message") or "").strip()
        if message:
            return Command(update={"reframed_question": message}, goto="intake")
        payload = {**payload, "error": "Please type your answer."}


def make_planning_node(llm):
    agent = create_agent(
        model=llm,
        tools=[
            airports_mcp_search_tool,
            flights_mcp_search_tool,
            tavily_mcp_search_tool,
        ],
        system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
    )

    async def planning(state: TravelState) -> dict:
        travel_details_json = state["travel_framed_details"].model_dump_json(indent=2)

        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Research and plan the complete trip using the "
                            "structured travel details below.\n\n"
                            "Follow the required tool-routing rules exactly.\n\n"
                            "Travel details:\n"
                            f"{travel_details_json}"
                        ),
                    }
                ]
            }
        )

        final_answer = response["messages"][-1].content

        return {
            "travel_search_result": final_answer,
            "messages": [AIMessage(content=final_answer)],
        }

    return planning


def human_decision_node(
    state: TravelState,
) -> Command[Literal["writer", "intake", "__end__"]]:
    payload = {
        "presentation": state["travel_search_result"],
        "question": "What would you like to do?",
        "options": {
            "inquire": "Keep these options and end search",
            "revise": "Change the trip/search and search something new",
            "approve": "Approve these options, go ahead and create a txt file",
        },
    }
    # Invalid input re-asks (with an "error" key) instead of raising, so the
    # thread never gets stuck on a bad resume value.
    while True:
        response = as_dict(interrupt(payload))
        action = response.get("action")

        if action == "approve":
            return Command(goto="writer")

        if action == "inquire":
            return Command(goto=END)

        if action == "revise":
            message = (response.get("message") or "").strip()
            if message:
                return Command(update={"reframed_question": message}, goto="intake")
            error = "'revise' needs a 'message' describing the change."
        else:
            error = f"Unknown action {action!r}. Use 'approve', 'revise' or 'inquire'."

        payload = {**payload, "error": error}


def make_writer_node(llm):
    agent = create_agent(
        model=llm,
        tools=[filesystem_write_mcp_tool],
        system_prompt=WRITER_SYSTEM_PROMPT,
    )

    async def writer(state: TravelState) -> dict:
        final_trip_plan = state.get("travel_search_result")

        # file name from the real destination, e.g. "Lisbon, Portugal" -> lisbon_portugal.txt
        filename = make_filename(state["travel_framed_details"].flight.destination)

        response = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Save the following completed travel plan using exactly "
                            f"this filename: {filename}\n\n"
                            "Do not change any factual information.\n\n"
                            f"{final_trip_plan}"
                        ),
                    }
                ]
            }
        )

        return {
            "saved_file": response["messages"][-1].content,
            "saved_filename": filename,
        }

    return writer


def route_after_intake(state: TravelState) -> Literal["clarify", "planning"]:
    return "clarify" if state["missing_info"] else "planning"


# ---------------------------------------------------------------- Graph
def build_serde() -> JsonPlusSerializer:
    """Allow our Pydantic models in the checkpoint (avoids the 'unregistered type' warning)."""
    return JsonPlusSerializer(
        allowed_msgpack_modules=[
            (TravelDetails.__module__, TravelDetails.__name__),
            (FlightDetails.__module__, FlightDetails.__name__),
        ]
    )


def build_graph(llm=None, checkpointer=None):
    """Compile the trip-planner graph.

    A checkpointer and a `thread_id` in the run config are required: they hold the
    state while the graph waits at a human-in-the-loop interrupt.
    """
    require_env()
    llm = llm or ChatOpenAI(model=MODEL_NAME, temperature=0)

    workflow = StateGraph(TravelState)
    workflow.add_node("intake", make_intake_node(llm))
    workflow.add_node("clarify", clarify_node)
    workflow.add_node("planning", make_planning_node(llm))
    workflow.add_node("human_decision", human_decision_node)
    workflow.add_node("writer", make_writer_node(llm))

    workflow.add_edge(START, "intake")
    workflow.add_conditional_edges(
        "intake",
        route_after_intake,
        {"clarify": "clarify", "planning": "planning"},
    )
    workflow.add_edge("planning", "human_decision")
    workflow.add_edge("writer", END)
    # clarify and human_decision route themselves with Command(goto=...)

    return workflow.compile(checkpointer=checkpointer or MemorySaver(serde=build_serde()))

