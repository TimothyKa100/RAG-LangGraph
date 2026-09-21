from __future__ import annotations

import json
import hashlib
from typing import Literal

from pydantic import BaseModel, Field


UcaCategory = Literal[
    "Not Provided",
    "Provided Incorrectly",
    "Provided but Not Needed",
    "Provided too Early",
    "Provided too Late",
    "Provided too Long",
    "Stopped Providing too Soon",
]


class UCARequest(BaseModel):
    control_action: str = Field(default="")
    source_controller: str = Field(default="", alias="from")
    target: str = Field(default="", alias="to")
    context: str = ""

    class Config:
        populate_by_name = True


def request_from_jsonl_line(line: str) -> UCARequest:
    """Extract and validate a UCA request from one training-style JSONL line."""
    record = json.loads(line)
    messages = record.get("messages", [])
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        raise ValueError("JSONL record does not contain a user message")

    content = user_messages[-1].get("content", "").strip()
    try:
        request_data = json.loads(content)
    except json.JSONDecodeError:
        request_data = json.loads("{" + content + "}", strict=False)
    normalized_request = {
        "control_action": request_data.get("Control Action", request_data.get("control_action")),
        "from": request_data.get("From", request_data.get("from")),
        "to": request_data.get("To", request_data.get("to")),
        "context": request_data.get("Context", request_data.get("context", "")),
    }
    return UCARequest.model_validate(normalized_request)


def diagnose_jsonl_line(line: str, line_number: int) -> dict[str, object]:
    """Validate one JSONL line and return metadata without exposing its content."""
    result: dict[str, object] = {
        "line": line_number,
        "characters": len(line),
        "utf8_bytes": len(line.encode("utf-8")),
        "sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
        "control_characters": [
            {"position": position, "codepoint": ord(character)}
            for position, character in enumerate(line)
            if ord(character) < 32 and character not in ("\n", "\r")
        ],
    }
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        result.update({"ok": False, "stage": "outer JSONL record", "error": str(error), "position": error.pos})
        return result

    messages = record.get("messages", []) if isinstance(record, dict) else []
    user_messages = [message for message in messages if message.get("role") == "user"]
    if not user_messages:
        result.update({"ok": False, "stage": "message extraction", "error": "no user message"})
        return result

    content = user_messages[-1].get("content", "").strip()
    result.update(
        {
            "user_content_characters": len(content),
            "user_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "user_content_control_characters": [
                {"position": position, "codepoint": ord(character)}
                for position, character in enumerate(content)
                if ord(character) < 32
            ],
        }
    )
    try:
        request_data = json.loads(content)
        parse_stage = "user content JSON"
    except json.JSONDecodeError:
        try:
            request_data = json.loads("{" + content + "}", strict=False)
            parse_stage = "wrapped user content JSON"
        except json.JSONDecodeError as error:
            result.update({"ok": False, "stage": "user content JSON", "error": str(error), "position": error.pos})
            return result

    normalized_request = {
        "control_action": request_data.get("Control Action", request_data.get("control_action")),
        "from": request_data.get("From", request_data.get("from")),
        "to": request_data.get("To", request_data.get("to")),
        "context": request_data.get("Context", request_data.get("context", "")),
    }
    try:
        UCARequest.model_validate(normalized_request)
    except Exception as error:
        result.update({"ok": False, "stage": "request validation", "error": str(error)})
        return result

    result.update({"ok": True, "stage": parse_stage})
    return result


class RetrievedContext(BaseModel):
    text: str
    source: str
    score: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)


class GeneratedUCAResponse(BaseModel):
    not_provided: list[str] = Field(default_factory=list, alias="Not Provided")
    provided_incorrectly: list[str] = Field(default_factory=list, alias="Provided Incorrectly")
    provided_but_not_needed: list[str] = Field(default_factory=list, alias="Provided but Not Needed")
    provided_too_early: list[str] = Field(default_factory=list, alias="Provided too Early")
    provided_too_late: list[str] = Field(default_factory=list, alias="Provided too Late")
    provided_too_long: list[str] = Field(default_factory=list, alias="Provided too Long")
    stopped_providing_too_soon: list[str] = Field(default_factory=list, alias="Stopped Providing too Soon")

    class Config:
        populate_by_name = True


class UCAResponse(GeneratedUCAResponse):
    context: str
