
"""Environment, paths and MCP server configuration."""
import logging
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp.server.providers.proxy").setLevel(logging.WARNING)

MODEL_NAME = os.getenv("TRIP_PLANNER_MODEL", "gpt-4o")

# The only folder the filesystem MCP server may touch.
SANDBOX_DIR = Path(os.getenv("TRIP_PLANNER_SANDBOX", "travel_plan_file")).resolve()

REQUIRED_ENV = ("OPENAI_API_KEY", "TAVILY_API_KEY", "IGNAV_API_KEY")

# MCP tool names as exposed by the servers (server prefix + tool name).
TOOL_TAVILY_SEARCH = "tavily_tavily_search"
TOOL_AIRPORTS = "ignav_search_airports"
TOOL_FLIGHTS = "ignav_search_flights"
TOOL_FS_WRITE = "filesystem_write_file"


def require_env() -> None:
    """Fail fast with a readable message if an API key is missing."""
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing environment variables: " + ", ".join(missing)
            + ". Add them to your .env file."
        )


def build_mcp_config() -> dict:
    """MCP servers: Tavily + Ignav (remote) and the filesystem server (local, via npx)."""
    require_env()
    SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "mcpServers": {
            "tavily": {
                "url": f"https://mcp.tavily.com/mcp/?tavilyApiKey={os.environ['TAVILY_API_KEY']}"
            },
            "ignav": {
                "url": "https://ignav.com/mcp",
                "headers": {"Authorization": f"Bearer {os.environ['IGNAV_API_KEY']}"},
            },
            "filesystem": {
                # shutil.which finds npx.cmd on Windows; falls back to plain "npx"
                "command": shutil.which("npx") or "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", str(SANDBOX_DIR)],
            },
        }
    }
