"""Prompt constants for the terminology system."""

TERMINOLOGY_SELECTION_SYSTEM_PROMPT = """Select obscure technical terms for terminology research.
Use only Control Action, From, and To. Ignore ordinary terms that do not need explanation.
Existing title matching must consider abbreviations, expanded names, spelling variants, and aliases.
Return JSON only."""