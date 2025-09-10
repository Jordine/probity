from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Literal, Any, Tuple
from enum import Enum


class DebateType(Enum):
    """Types of debates supported"""
    BASELINE = "baseline"  # No probes
    DEBATER_ACCESS = "debater_access"  # Debaters see probes, judge doesn't
    FULL_ACCESS = "full_access"  # Everyone sees probes
    JUDGE_ONLY = "judge_only"  # Only judge sees probes


class ProviderType(Enum):
    """Supported model providers"""
    LOCAL = "local"  # HuggingFace/TransformerLens
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
    supports_probes: bool = False  # Only local models support probes
    
    def __post_init__(self):
        # Only local models can have probes run on them
        self.supports_probes = (self.provider == ProviderType.LOCAL)


@dataclass
class ProbeConfig:
    """Configuration for probe scoring"""
    probe_dir: str
    probe_types: List[str]
    layer: int
    model_name: str  # Model being probed
    device: str = "cuda"
    enabled: bool = True
    score_threshold: float = 0.7  # Threshold for flagging deception


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


@dataclass
class DebateTurn:
    """A single turn in the debate"""
    round: int
    speaker: DebateRole
    content: str
    probe_scores: Optional[Dict[str, List[float]]] = None  # token-level scores
    mean_probe_score: Optional[float] = None
    flagged_claims: Optional[List[str]] = None  # Claims flagged as deceptive
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
        """Create probe access config for a debate type"""
        if debate_type == DebateType.BASELINE:
            return cls()  # No one sees probes
        elif debate_type == DebateType.DEBATER_ACCESS:
            return cls(
                honest_can_see_opponent=True,
                dishonest_can_see_opponent=True
            )
        elif debate_type == DebateType.FULL_ACCESS:
            return cls(
                honest_can_see_opponent=True,
                dishonest_can_see_opponent=True,
                judge_can_see_honest=True,
                judge_can_see_dishonest=True
            )
        elif debate_type == DebateType.JUDGE_ONLY:
            return cls(
                judge_can_see_honest=True,
                judge_can_see_dishonest=True
            )
        else:
            raise ValueError(f"Unknown debate type: {debate_type}")