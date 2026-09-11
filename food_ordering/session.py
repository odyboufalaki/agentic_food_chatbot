from collections import deque
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

from food_ordering.model_adapter import CustomerMessage, ModelMessage, TranscriptTurn
from food_ordering.order import OrderLine
from food_ordering.tool_protocol import DraftState, ReviewSnapshot, SubmissionResult


OrderStatus = Literal[
    "draft", "submitted", "rejected", "application_error", "not_sent", "uncertain",
]
TRANSCRIPT_TURN_LIMIT = 12


@dataclass
class Session:
    """Persistent state for one food-ordering conversation."""

    lines: list[OrderLine] = field(default_factory=list)
    instructions: str = ""
    next_line_number: int = 1
    transcript: deque[TranscriptTurn] = field(
        default_factory=lambda: deque(maxlen=TRANSCRIPT_TURN_LIMIT),
    )
    session_id: str = field(default_factory=lambda: uuid4().hex)
    turn_id: int = 0
    revision: int = 0
    reviewed_revision: int | None = None
    review_snapshot: ReviewSnapshot | None = None
    last_submission_attempt_turn: int | None = None
    submission_outcome: SubmissionResult | None = None
    status: OrderStatus = "draft"
    receipt_message: str = ""

    def latest_turn_messages(self, customer_message: str) -> tuple[ModelMessage, ...]:
        """Return the completed transcript turn for the supplied customer input."""

        if not self.transcript:
            return ()
        messages = self.transcript[-1].messages
        if not messages or not isinstance(messages[0], CustomerMessage):
            return ()
        return messages if messages[0].content == customer_message else ()

    def commit_draft(self, candidate: DraftState) -> None:
        """Commit one changed Draft candidate and invalidate stale checkout state."""

        self.lines = list(candidate.lines)
        self.instructions = candidate.general_instructions
        self.next_line_number = candidate.next_line_number
        self.revision += 1
        self._clear_checkout_state()

    def invalidate_review(self) -> None:
        """Clear confirmation eligibility unless the Session is permanently locked."""

        if self.status in {"submitted", "uncertain"}:
            return
        self.reviewed_revision = None
        self.review_snapshot = None

    def start_new_order(self) -> tuple[str, ...]:
        """Start a clean Draft while preserving never-reused line identity."""

        removed_line_ids = tuple(line.line_id for line in self.lines)
        self.lines = []
        self.instructions = ""
        self.revision += 1
        self._clear_checkout_state()
        return removed_line_ids

    def _clear_checkout_state(self) -> None:
        self.reviewed_revision = None
        self.review_snapshot = None
        self.last_submission_attempt_turn = None
        self.submission_outcome = None
        self.status = "draft"
        self.receipt_message = ""
