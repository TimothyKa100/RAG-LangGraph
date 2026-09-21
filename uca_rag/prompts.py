import json


GENERATION_SYSTEM_PROMPT = """You are performing STPA step 3. Generate unsafe control action statements for the supplied control action.

Use the exact statement pattern: <Source Controller> <Type> <Control Action> <Context>.
Return valid JSON with exactly these keys:
- Not Provided
- Provided Incorrectly
- Provided but Not Needed
- Provided too Early
- Provided too Late
- Provided too Long
- Stopped Providing too Soon

Each category value must be an array of strings.
"""


def build_generation_prompt(control_action: str, source: str, target: str, context: str) -> str:
    request = json.dumps(
        {"Control Action": control_action, "From": source, "To": target},
        ensure_ascii=False,
    )
    return f"""Control Action: {control_action}
From: {source}
To: {target}

The input JSON below is the only control action to analyze. Do not copy a UCA from another example or infer a different source, action, or target.

Evidence and requested context:
{context}

Use a precise context from the evidence. If a category is not applicable, use [] or the required N/A note. Do not invent facts not supported by the evidence.
Generate the seven UCA categories as JSON.

Input JSON:
{request}"""


def build_search_queries(control_action: str, source: str, target: str) -> list[str]:
    base = f'"{control_action}" "{source}" "{target}"'
    return [
        f"{base} accident report",
        f"{base} collision incident investigation",
        f"{base} safety failure near miss",
        f"{source} {control_action} {target} hazard warning loss of control",
    ]
