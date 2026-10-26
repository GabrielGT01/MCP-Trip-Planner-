"""
Streamlit front-end for the MCP trip planner.

    streamlit run streamlit_app.py

Streamlit re-runs this script on every click, but the MCP connections and the
LangGraph checkpointer must outlive those re-runs. So one background event loop
(created once, cached) owns the graph and every MCP connection, and the script
hands coroutines to it.
"""
import asyncio
import atexit
import threading
import uuid

import streamlit as st
from langgraph.types import Command

from config import MODEL_NAME, SANDBOX_DIR
from pipeline import build_graph
from schemas import get_interrupt
from tools import close_mcp

st.set_page_config(page_title="MCP Trip Planner", page_icon="✈️", layout="wide")


# -----------------------------------------------------------------------------
# Runtime: one event loop + one compiled graph for the whole server process
# -----------------------------------------------------------------------------
class AsyncRuntime:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, name="trip-planner-loop", daemon=True).start()
        self.graph = build_graph()  # MemorySaver: one thread_id per browser session
        atexit.register(self.close)

    def run(self, coro, timeout: float = 600):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def close(self):
        try:
            self.run(close_mcp(), timeout=10)
        except Exception:
            pass


@st.cache_resource(show_spinner="Starting planner…")
def get_runtime() -> AsyncRuntime:
    return AsyncRuntime()


try:
    rt = get_runtime()
except RuntimeError as exc:  # e.g. missing API keys
    st.error(str(exc))
    st.stop()


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
def new_session():
    st.session_state.update(
        thread_id=uuid.uuid4().hex,
        result=None,      # last graph result
        pending=None,     # pending interrupt payload (None = run finished)
        error=None,
        turn=0,           # bumps on every run so widgets reset
    )


if "thread_id" not in st.session_state:
    new_session()


def advance(payload, spinner_text: str):
    """Run the graph (first message or Command(resume=...)) and store the outcome."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    with st.spinner(spinner_text):
        try:
            result = rt.run(rt.graph.ainvoke(payload, config))
        except Exception as exc:
            st.session_state.error = f"{type(exc).__name__}: {exc}"
        else:
            st.session_state.error = None
            st.session_state.result = result
            st.session_state.pending = get_interrupt(result)
    st.session_state.turn += 1
    st.rerun()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Trip planner")
    if st.button("New trip"):
        new_session()
        st.rerun()
    st.caption(f"Model: {MODEL_NAME}")
    st.caption(f"Sandbox: {SANDBOX_DIR}")
    st.caption(f"Thread: {st.session_state.thread_id[:8]}")


# -----------------------------------------------------------------------------
# Main view
# -----------------------------------------------------------------------------
st.title("✈️ MCP Trip Planner")

if st.session_state.error:
    st.error(st.session_state.error)

result = st.session_state.result
pending = st.session_state.pending
turn = st.session_state.turn

# What the intake step understood (useful while clarifying / reviewing)
if result and result.get("travel_framed_details"):
    with st.expander("What I understood from your request"):
        st.json(result["travel_framed_details"].model_dump())

if result is None:
    # ---- 1. first request ---------------------------------------------------
    with st.form(f"request_{turn}"):
        text = st.text_area(
            "Describe your trip",
            height=120,
            placeholder="Cologne to Lisbon, 2026-11-12 to 2026-11-18, 2 adults, "
                        "budget 1500 EUR, I like food and history",
        )
        submitted = st.form_submit_button("Plan trip", type="primary")
    if submitted:
        if text.strip():
            advance({"question": text.strip()}, "Reading your request…")
        else:
            st.warning("Write a few words about the trip first.")

elif pending is None:
    # ---- 4. finished (approved or ended) ------------------------------------
    st.markdown(result.get("travel_search_result", ""))
    st.divider()

    filename = result.get("saved_filename")
    if filename:
        path = SANDBOX_DIR / filename
        if path.exists():
            st.success(f"Saved to {path}")
            st.download_button(
                "Download .txt",
                path.read_text(encoding="utf-8"),
                file_name=filename,
                mime="text/plain",
            )
        else:
            st.warning(f"The writer finished but {path} does not exist.")
        if result.get("saved_file"):
            st.caption(result["saved_file"])
    else:
        st.info("Search ended. Nothing was saved. Use 'New trip' in the sidebar to start again.")

elif "missing" in pending:
    # ---- 2. clarify ---------------------------------------------------------
    st.subheader(pending["question"])
    if pending.get("error"):
        st.warning(pending["error"])
    for item in pending["missing"]:
        st.markdown(f"- {item}")
    with st.form(f"clarify_{turn}"):
        reply = st.text_input("Your answer")
        sent = st.form_submit_button("Send", type="primary")
    if sent:
        if reply.strip():
            advance(Command(resume={"message": reply.strip()}), "Checking your details…")
        else:
            st.warning("Type your answer first.")

else:
    # ---- 3. review the plan and decide --------------------------------------
    st.markdown(pending["presentation"])
    st.divider()
    st.subheader(pending["question"])
    if pending.get("error"):
        st.warning(pending["error"])
    options = pending["options"]

    revise_msg = st.text_area(
        "Want changes? Describe them, then press Revise",
        key=f"revise_{turn}",
        height=80,
        placeholder="e.g. I want a hotel near the airport",
    )

    col_approve, col_revise, col_end = st.columns(3)
    if col_approve.button("✅ Approve & save .txt", type="primary", help=options["approve"]):
        advance(Command(resume={"action": "approve"}), "Writing the file…")
    if col_revise.button("✏️ Revise", help=options["revise"]):
        if revise_msg.strip():
            advance(
                Command(resume={"action": "revise", "message": revise_msg.strip()}),
                "Searching again…",
            )
        else:
            st.warning("Describe what to change first.")
    if col_end.button("📌 Keep & end", help=options["inquire"]):
        advance(Command(resume={"action": "inquire"}), "Finishing…")
