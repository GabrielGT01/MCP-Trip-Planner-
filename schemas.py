

"""Pydantic models, graph state and small pure helpers."""
import datetime
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------- Pydantic models
class FlightDetails(BaseModel):
    origin: str | None = Field(
        default=None,
        description=(
            "The city, airport, metro area, or country the user wants to depart from. "
            "Do not invent an IATA code if the user did not provide one."
        ),
    )

    destination: str | None = Field(
        default=None,
        description="The city, country, municipality, or travel destination the user wants to visit.",
    )

    departure_date: str | None = Field(
        default=None,
        description="Departure date in YYYY-MM-DD format if known.",
    )

    return_date: str | None = Field(
        default=None,
        description=(
            "Return date in YYYY-MM-DD format. "
            "Use None when this is a one-way trip or the date is not yet known."
        ),
    )

    trip_type: Literal["one_way", "round_trip", "unknown"] = Field(
        default="unknown",
        description="Whether the user wants a one-way or round-trip flight.",
    )

    # fail fast on bad date formats before they reach Ignav
    @field_validator("departure_date", "return_date")
    @classmethod
    def check_iso_date(cls, v):
        if v is not None:
            datetime.date.fromisoformat(v)  # raises on bad format
        return v

    # return must be after departure
    @model_validator(mode="after")
    def check_date_order(self):
        if self.departure_date and self.return_date:
            if self.return_date < self.departure_date:  # ISO strings sort correctly
                raise ValueError("return_date must be after departure_date")
        return self


class TravelDetails(BaseModel):
    flight: FlightDetails = Field(default_factory=FlightDetails)

    adults: int = Field(default=1, ge=1, description="Number of adult travellers.")

    budget_amount: float | None = Field(
        default=None,
        description="Maximum or approximate trip budget if the user gives one.",
    )

    budget_currency: str | None = Field(
        default=None,
        description="Currency of the budget, such as EUR, USD or GBP.",
    )

    nationality: str | None = Field(
        default=None,
        description=(
            "Traveller nationality/passport country when provided. "
            "Useful for visa and entry requirement research."
        ),
    )

    hotel_preferences: str | None = Field(
        default=None,
        description=(
            "Hotel requirements such as luxury level, neighborhood, "
            "family-friendly, hostel, beach access, etc."
        ),
    )

    interests: list[str] = Field(
        default_factory=list,
        description=(
            "Activities or interests such as museums, nightlife, hiking, "
            "food, beaches, history, shopping, concerts or family activities."
        ),
    )


# ---------------------------------------------------------------- Graph state
class TravelState(TypedDict):
    question: str
    reframed_question: str | None
    travel_framed_details: TravelDetails | None
    missing_info: list[str]
    travel_search_result: str
    saved_file: str          # the writer agent's final message
    saved_filename: str      # file name written inside SANDBOX_DIR
    messages: Annotated[list[AnyMessage], add_messages]


# ---------------------------------------------------------------- Helpers
def get_missing(d: TravelDetails) -> list[str]:
    """Fields the planner still needs before it can search."""
    f = d.flight
    missing = []
    if not f.origin:
        missing.append("departure city or airport")
    if not f.destination:
        missing.append("destination")
    if not f.departure_date:
        missing.append("exact departure date (day, month and year)")
    if f.trip_type == "unknown":
        missing.append("whether the trip is one-way or round-trip")
    elif f.trip_type == "round_trip" and not f.return_date:
        missing.append("exact return date (day, month and year)")
    return missing


def make_filename(destination: str | None) -> str:
    """'Lisbon, Portugal' -> 'lisbon_portugal.txt'"""
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", (destination or "trip").strip()).strip("_").lower()
    return f"{slug or 'trip'}.txt"


def as_dict(value) -> dict:
    """Resume values should be dicts; anything else counts as empty input."""
    return value if isinstance(value, dict) else {}


def get_interrupt(result: dict) -> dict | None:
    """Payload of the pending interrupt in a graph result, or None if the run finished."""
    pending = result.get("__interrupt__")
    return pending[0].value if pending else None
