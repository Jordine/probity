from typing import Callable, List, Union, Optional
import re
from dataclasses import dataclass
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


@dataclass
class Position:
    """Represents a position or span in text."""

    start: int  # Character start position
    end: Optional[int] = None  # Character end position (exclusive)

    def __post_init__(self):
        if self.start < 0:
            raise ValueError("Start position must be non-negative")
        if self.end is not None and self.end < self.start:
            raise ValueError("End position must be greater than start position")


class PositionFinder:
    """Strategies for finding target positions in text."""

    @staticmethod
    def from_template(template: str, marker: str) -> Callable[[str], Position]:
        """Create finder for template-based positions.

        Args:
            template: Template string with marker (e.g. "The movie was {ADJ}")
            marker: The marker to find (e.g. "{ADJ}")

        Returns:
            Function that finds the position in a string matching the template
        """

        def finder(text: str) -> Position:
            # First escape the entire template
            regex = re.escape(template)

            # Then replace the escaped marker with a capture group
            escaped_marker = re.escape(marker)
            regex = regex.replace(escaped_marker, "(.*?)")

            # Replace other template variables with non-capturing wildcards
            for var in re.findall(r"\\\{([^}]+)\\\}", regex):
                var_marker = f"\\{{{var}\\}}"
                regex = regex.replace(var_marker, ".*?")
            # print(f"Template: {template}")
            # print(f"Marker: {marker}")
            # print(f"Regex: {regex}")
            # print(f"Text: {text}")
            match = re.match(regex, text)
            if not match:
                raise ValueError(f"Text does not match template: {text}")

            # Get the position of the matching group
            start = match.start(1)
            end = match.end(1)
            return Position(start, end)

        return finder

    @staticmethod
    def from_regex(pattern: str, group: int = 0) -> Callable[[str], List[Position]]:
        """Create finder for regex-based positions.

        Args:
            pattern: Regex pattern to match
            group: Which capture group to use for position (0 = full match)

        Returns:
            Function that finds all matching positions in a string
        """
        compiled = re.compile(pattern)

        def finder(text: str) -> List[Position]:
            positions = []
            for match in compiled.finditer(text):
                start = match.start(group)
                end = match.end(group)
                positions.append(Position(start, end))
            return positions

        return finder

    @staticmethod
    def from_char_position(pos: int) -> Callable[[str], Position]:
        """Create finder for fixed character positions.

        Args:
            pos: Character position to find

        Returns:
            Function that returns the fixed position
        """

        def finder(text: str) -> Position:
            if pos >= len(text):
                raise ValueError(f"Position {pos} is beyond text length {len(text)}")
            return Position(pos)

        return finder

    @staticmethod
    def convert_to_token_position(
        position: Position,
        text: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        space_precedes_token: bool = True,
        add_special_tokens: bool = True,
        padding_side: Optional[str] = None,
    ) -> int:
        """
        Convert character position to token position, handling special tokens properly.

        Args:
            position: Character position to convert
            text: Input text
            tokenizer: Tokenizer to use
            space_precedes_token: Whether space precedes token (default: True)
            add_special_tokens: Whether to add special tokens (default: True)
            padding_side: Override tokenizer padding side (default: None)

        Returns:
            Token index corresponding to the character position
        """
        # Unused: Get tokenizer padding side.
        # pad_side = padding_side if padding_side is not None \
        #            else getattr(tokenizer, "padding_side", "right")

        # First get clean offsets without special tokens
    
        if position.end is not None:
            print(f"DEBUG [PosConverter]: Converting span {position.start}-{position.end} ({position.end - position.start} chars)")
            print(f"  Text: '{text[position.start:position.end][:50]}...'")
            
        clean_encoding = tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        clean_offsets = clean_encoding["offset_mapping"]
        # clean_tokens = clean_encoding["input_ids"]

        # Find the token index in the clean encoding
        clean_token_idx = None
        assert isinstance(clean_offsets, list)  # Assure type checker it's iterable
        for idx, (start, end) in enumerate(clean_offsets):
            if start <= position.start < end:
                clean_token_idx = idx
                break

        if clean_token_idx is None:
            msg = (
                f"Character position {position.start} not aligned "
                f"with any token offset."
            )
            raise ValueError(msg)

        # If we don't need to account for special tokens, just return the clean index
        if not add_special_tokens:
            return clean_token_idx

        # Get tokens with special tokens to check for actual prefix tokens
        tokens_with_special = tokenizer(text, add_special_tokens=True)["input_ids"]

        # Count special tokens at the beginning IF they actually appear
        prefix_tokens = 0
        if (
            hasattr(tokenizer, "bos_token_id")
            and tokenizer.bos_token_id is not None
            and tokens_with_special  # Ensure list is not empty
            and tokens_with_special[0] == tokenizer.bos_token_id
        ):
            prefix_tokens += 1
        # Add checks for other potential prefix tokens if necessary (e.g., cls_token)
        # elif (
        #     hasattr(tokenizer, "cls_token_id")
        #     and tokenizer.cls_token_id is not None
        #     and tokens_with_special
        #     and tokens_with_special[0] == tokenizer.cls_token_id
        # ):
        #    prefix_tokens += 1

        # Calculate the token position relative to the sequence *with* special tokens
        position_with_special = clean_token_idx + prefix_tokens

        # Note: At this point, the position only accounts for special tokens like BOS,
        # but not for padding. Padding is dynamic and handled at batch preparation time
        # in get_batch_tensors.

        return position_with_special

    @staticmethod
    def validate_token_position(token_position: int, tokens: List[int]) -> bool:
        """Validate that a token position is valid for a sequence.

        Args:
            token_position: Position to validate
            tokens: Token sequence

        Returns:
            True if position is valid, False otherwise
        """
        return 0 <= token_position < len(tokens)

    
    
    @staticmethod
    def convert_char_span_to_token_positions(
        position: Position,
        text: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        add_special_tokens: bool = True,
    ) -> List[int]:
        """
        Convert character span (start, end) to list of ALL token positions in that span.
        
        Args:
            position: Character position with start and end
            text: Input text
            tokenizer: Tokenizer to use
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token indices covering the character span
        """
        # Get token offsets without special tokens first
        clean_encoding = tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        clean_offsets = clean_encoding["offset_mapping"]
        
        # Find ALL tokens that overlap with the character span
        token_positions = []
        assert isinstance(clean_offsets, list)
        
        for idx, (token_start, token_end) in enumerate(clean_offsets):
            # Include token if it has ANY overlap with the character span
            if token_end > position.start and token_start < position.end:
                token_positions.append(idx)
        
        if not token_positions:
            # Fallback: find the nearest token if no exact overlap
            for idx, (token_start, token_end) in enumerate(clean_offsets):
                if token_start <= position.start < token_end:
                    token_positions.append(idx)
                    break
        
        if not token_positions:
            raise ValueError(
                f"Character span {position.start}-{position.end} does not align with any tokens"
            )
        
        # Adjust for special tokens if needed
        if add_special_tokens:
            tokens_with_special = tokenizer(text, add_special_tokens=True)["input_ids"]
            
            # Count prefix special tokens
            prefix_tokens = 0
            if (
                hasattr(tokenizer, "bos_token_id")
                and tokenizer.bos_token_id is not None
                and tokens_with_special
                and tokens_with_special[0] == tokenizer.bos_token_id
            ):
                prefix_tokens += 1
            
            # Adjust all positions by prefix offset
            token_positions = [pos + prefix_tokens for pos in token_positions]
        
        return token_positions
