"""
Probity Extensions for NTML (N Truths, M Lies) Conversational Datasets

This module extends Probity's core functionality to support statement-level probing
in conversational datasets where individual statements within assistant responses
need to be classified as true or false.

Key Components:
- ConversationalProbingExample: Extends ProbingExample with conversation structure
- ConversationalProbingDataset: Extends ProbingDataset with NTML JSONL loading
- NTML-specific analysis tools for statement-level probing research
"""

from .conversational import (
    ConversationalProbingExample,
    ConversationalProbingDataset,
)

__all__ = [
    "ConversationalProbingExample", 
    "ConversationalProbingDataset",
] 