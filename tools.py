
"""MCP client (keeps connections open) and the LangChain tool wrappers."""
import asyncio

from langchain.mcp import MCPAdapter  # beta API, as used in the original notebook
from langchain_core.tools import tool

from config import (
    SANDBOX_DIR,
    TOOL_AIRPORTS,
    TOOL_FLIGHTS,
    TOOL_FS_WRITE,
    TOOL_TAVILY_SEARCH,
    build_mcp_config,
)


# ---------------------------------------------------------------- MCP connection
class MCPToolbox:
    """Opens the MCP connections once and keeps them open.

    The connections live inside one long-running task, so they are opened and
    closed by the same task (required by anyio). Works in a notebook and in a
    long-lived background event loop (Streamlit).
    """

    def __init__(self, config_factory):
        self._config_factory = config_factory
        self._tools = None
        self._task: asyncio.Task | None = None
        self._ready: asyncio.Event | None = None
        self._stop: asyncio.Event | None = None
        self._error: Exception | None = None
        self._start_lock: asyncio.Lock | None = None

    async def _run(self):
        try:
            async with MCPAdapter(self._config_factory()) as adapter:
                self._tools = await adapter.list_tools()
                self._ready.set()
                await self._stop.wait()   # stay connected until close()
        except Exception as exc:
            self._error = exc
        finally:
            self._ready.set()

    async def get_tools(self):
        if self._tools is not None:
            return self._tools

        if self._start_lock is None:
            self._start_lock = asyncio.Lock()

        async with self._start_lock:
            if self._tools is None:
                if self._task is None:
                    self._ready, self._stop = asyncio.Event(), asyncio.Event()
                    self._error = None
                    self._task = asyncio.create_task(self._run())
                await self._ready.wait()
                if self._tools is None:
                    err, self._task, self._error = self._error, None, None
                    raise RuntimeError(f"Could not start the MCP servers: {err!r}") from err
        return self._tools

    async def get_tool(self, name: str):
        tools = await self.get_tools()
        for mcp_tool in tools:
            if mcp_tool.name == name:
                return mcp_tool
        available = ", ".join(t.name for t in tools)
        raise RuntimeError(f"MCP tool '{name}' was not found. Available: {available}")

    async def close(self):
        """Disconnect every MCP server. Call from the same event loop that opened them."""
        if self._task is None:
            return
        self._stop.set()
        await self._task
        self._task, self._tools = None, None


toolbox = MCPToolbox(build_mcp_config)


async def close_mcp():
    """Run once at the very end of a session."""
    await toolbox.close()


# ---------------------------------------------------------------- Research tools (Tavily, Ignav)
@tool
async def tavily_mcp_search_tool(query: str):
    """
    Search the web for destination and travel research.

    Use ONLY for:
    - hotels
    - neighborhoods
    - attractions
    - food
    - events
    - visa requirements
    - local transport
    - travel tips

    NEVER use this tool for:
    - flights
    - flight prices
    - airlines
    - flight schedules
    - flight booking links

    For any flight-related request, use flights_mcp_search_tool instead.
    """

    mcp_tool = await toolbox.get_tool(TOOL_TAVILY_SEARCH)
    return await mcp_tool.ainvoke({"query": query})


@tool
async def airports_mcp_search_tool(
    query: str,
    limit: int = 5,
):
    """
    Look up airports and their IATA codes.

    Use this before searching flights whenever the correct airport
    code is unknown or when a city has multiple airports.

    Args:
        query:
            City, airport name, metro area, or IATA code.

            Examples:
            "Cologne"
            "Lisbon"
            "London"
            "LHR"

        limit:
            Maximum number of airport matches to return.

    Returns airport information including IATA codes.
    """

    mcp_tool = await toolbox.get_tool(TOOL_AIRPORTS)
    return await mcp_tool.ainvoke({"query": query, "limit": limit})


@tool
async def flights_mcp_search_tool(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str | None = None,
):
    """
    Search live flights and flight prices.

    This is the ONLY tool that may be used for:
    - flight options
    - flight prices
    - airlines
    - flight schedules
    - flight booking links

    Do not use Tavily for flight information.

    Search for one-way or round-trip flights.

    Use IATA airport codes such as CGN, LIS, LHR, JFK.

    If return_date is provided, search for a round trip.
    If return_date is None, search for a one-way flight.

    Dates must use YYYY-MM-DD format.
    """

    mcp_tool = await toolbox.get_tool(TOOL_FLIGHTS)

    payload = {
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
    }
    if return_date is not None:
        payload["return_date"] = return_date

    return await mcp_tool.ainvoke(payload)


# ---------------------------------------------------------------- Write tool (sandbox only)
@tool
async def filesystem_write_mcp_tool(
    filename: str,
    content: str,
):
    """
    Write text content to a .txt file inside the allowed sandbox directory.

    Use this tool to save generated text such as travel plans.

    Args:
        filename:
            Name of the file inside the sandbox, named after the trip
            destination and ending in .txt.

        content:
            The full text content that should be written to the file.

    Important:
    - Files are written only inside the sandbox directory.
    - If the file already exists, its contents may be overwritten.
    """

    mcp_tool = await toolbox.get_tool(TOOL_FS_WRITE)

    sandbox_root = SANDBOX_DIR.resolve()
    file_path = (sandbox_root / filename).resolve()

    # Safety check: prevent ../ or absolute paths from escaping the sandbox
    if not file_path.is_relative_to(sandbox_root):
        raise ValueError("File must be written inside the sandbox directory.")

    return await mcp_tool.ainvoke({"path": str(file_path), "content": content})

