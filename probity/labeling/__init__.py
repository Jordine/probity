"""
Unified LLM tagging for probity data.

Two modes:
- TOKEN_SPAN: Training + validation data - locate deceptive token spans
- STATEMENT: Debate transcripts - categorize statements
"""

from probity.labeling.unified_tagger import (
    UnifiedTagger,
    TaggingMode,
    InputFormat,
    DataFormat,  # backwards compat alias
    TokenSpan,
    StatementLabel,
    TaggedSample,
)

__all__ = [
    "UnifiedTagger",
    "TaggingMode",
    "InputFormat",
    "DataFormat",
    "TokenSpan",
    "StatementLabel",
    "TaggedSample",
]
