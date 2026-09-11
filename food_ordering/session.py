from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from food_ordering.order import ClarificationContext, OrderLine
from food_ordering.proposals import Proposal


OrderStatus = Literal["draft", "submitted", "rejected", "application_error", "uncertain"]


@dataclass(frozen=True)
class PendingChange:
    original_message: str
    proposal: Proposal
    clarification: ClarificationContext

    def snapshot(self) -> dict[str, Any]:
        return {
            "original_message": self.original_message,
            "proposal": self.proposal.model_dump(),
            "reason": self.clarification.reason,
            "question": self.clarification.fallback_question,
            "subject": self.clarification.subject,
            "field": self.clarification.field,
            "choices": list(self.clarification.choices),
        }


@dataclass
class Session:
    """Persistent state for one food-ordering conversation."""

    lines: list[OrderLine] = field(default_factory=list)
    instructions: str = ""
    next_line_number: int = 1
    history: deque[dict[str, str]] = field(default_factory=lambda: deque(maxlen=12))
    session_id: str = field(default_factory=lambda: uuid4().hex)
    turn_id: int = 0
    revision: int = 0
    reviewed_revision: int | None = None
    status: OrderStatus = "draft"
    receipt_message: str = ""
    rejected_payload: dict[str, Any] | None = None
    application_error_payload: dict[str, Any] | None = None
    retry_requires_review: bool = False
    pending_change: PendingChange | None = None
