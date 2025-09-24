from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum


class DebateType(Enum):
    """Types of debates supported"""
    BASELINE = "baseline"
    DEBATER_ACCESS = "debater_access"
    FULL_ACCESS = "full_access"
    JUDGE_ONLY = "judge_only"


class ProviderType(Enum):
    """Supported model providers"""
    LOCAL = "local"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class DebateRole(Enum):
    """Roles in a debate"""
    HONEST = "honest"
    DISHONEST = "dishonest"
    JUDGE = "judge"


@dataclass
class ModelConfig:
    """Configuration for a model"""
    provider: ProviderType
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    supports_probes: bool = False

    def __post_init__(self):
        self.supports_probes = (self.provider == ProviderType.LOCAL)


@dataclass
class ProbeConfig:
    """Configuration for probe scoring"""
    honest_probe_dir: str  # Probe directory for honest debater
    dishonest_probe_dir: str  # Probe directory for dishonest debater
    probe_types: List[str]
    layer: int
    honest_model_name: str  # Model name for honest debater probes
    dishonest_model_name: str  # Model name for dishonest debater probes
    device: str = "cuda"
    enabled: bool = True
    score_threshold: float = 0.7
    cache_dir: Optional[str] = None
    format_for_debaters: bool = True

@dataclass
class DebateConfig:
    """Configuration for a debate"""
    honest_debater: ModelConfig
    dishonest_debater: ModelConfig
    judge: ModelConfig
    probe_config: Optional[ProbeConfig]
    debate_type: DebateType
    max_rounds: int = 3
    topic: Optional[str] = None
    problem_id: Optional[str] = None
    save_dir: str = "./debate_results"
    probe_display_mode: str = "per_statement"
    judge_probe_display_mode: str = "per_statement"


@dataclass
class ProbeScore:
    """Holds the results of a single probe's scoring."""
    probe_type: str
    layer: int
    tokens: List[str]
    token_scores: List[float]
    mean_score: float
    metadata: Dict[str, Any]


@dataclass
class DebateTurn:
    """A single turn in the debate"""
    round: int
    speaker: DebateRole
    content: str
    tokens: Optional[List[str]] = None
    probe_scores: Optional[Dict[str, List[float]]] = None
    mean_probe_score: Optional[float] = None
    flagged_claims: Optional[List[str]] = None
    timestamp: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DebateResult:
    """Complete debate result"""
    debate_id: str
    config: DebateConfig
    turns: List[DebateTurn]
    winner: Optional[DebateRole]
    judge_reasoning: str
    probe_analysis: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]
    start_time: float
    end_time: float
    success: bool = True
    error: Optional[str] = None


@dataclass
class APPSProblem:
    """APPS dataset problem"""
    problem_id: str
    question: str
    solutions: List[str]
    input_output: List[Tuple[str, str]]
    difficulty: str
    tags: List[str]

@dataclass
class QuALITYProblem:
    """QuALITY dataset problem"""
    question_id: str
    story: str
    story_title: str
    question: str
    correct_answer: str
    incorrect_answer: str
    answers: List[str]  # All answer options
    gold_label: int  # Index of correct answer
    difficulty: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeAccessConfig:
    """Configuration for probe access in debates"""
    honest_can_see_own: bool = False
    honest_can_see_opponent: bool = False
    dishonest_can_see_own: bool = False
    dishonest_can_see_opponent: bool = False
    judge_can_see_honest: bool = False
    judge_can_see_dishonest: bool = False

    @classmethod
    def for_debate_type(cls, debate_type: DebateType) -> "ProbeAccessConfig":
        if debate_type == DebateType.BASELINE:
            return cls()
        elif debate_type == DebateType.DEBATER_ACCESS:
            return cls(honest_can_see_opponent=True, dishonest_can_see_opponent=True)
        elif debate_type == DebateType.FULL_ACCESS:
            return cls(
                honest_can_see_opponent=True,
                dishonest_can_see_opponent=True,
                judge_can_see_honest=True,
                judge_can_see_dishonest=True,
            )
        elif debate_type == DebateType.JUDGE_ONLY:
            return cls(judge_can_see_honest=True, judge_can_see_dishonest=True)
        else:
            raise ValueError(f"Unknown debate type: {debate_type}")