"""Version 1 model-tool and outcome contract.

This module defines the production protocol shared by schemas, parsing,
deterministic validators, orchestration, and logs.
"""

from dataclasses import dataclass, field
import json
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from food_ordering.models import StrictModel
from food_ordering.order import OrderLine


TOOL_PROTOCOL_VERSION = 1


class LineTarget(StrictModel):
    type: Literal["line"]
    line_id: str


class MatchTarget(StrictModel):
    type: Literal["match"]
    item_id: str
    options: dict[str, str] = Field(default_factory=dict)
    extras: list[str] = Field(default_factory=list)
    instructions: str | None = None


Target: TypeAlias = Annotated[LineTarget | MatchTarget, Field(discriminator="type")]


class ShowMenu(StrictModel):
    item_ids: list[str] = Field(default_factory=list)


class ShowDraft(StrictModel):
    pass


class AddItem(StrictModel):
    item_id: str
    quantity: int = Field(ge=1)
    options: dict[str, str] = Field(default_factory=dict)
    extras: list[str] = Field(default_factory=list)
    instructions: str = ""


class CustomizeServings(StrictModel):
    type: Literal["customize"]
    options: dict[str, str] = Field(default_factory=dict)
    add_extras: list[str] = Field(default_factory=list)
    remove_extras: list[str] = Field(default_factory=list)
    instructions: str | None = None

    @model_validator(mode="after")
    def require_a_patch(self) -> Self:
        if (
            not self.options
            and not self.add_extras
            and not self.remove_extras
            and self.instructions is None
        ):
            raise ValueError("customization must include at least one change")
        return self


class ReplaceServings(StrictModel):
    type: Literal["replace"]
    item_id: str
    options: dict[str, str] = Field(default_factory=dict)
    extras: list[str] = Field(default_factory=list)
    instructions: str = ""


ServingChange: TypeAlias = Annotated[
    CustomizeServings | ReplaceServings,
    Field(discriminator="type"),
]
PositiveServingCount: TypeAlias = Annotated[int, Field(ge=1)]


class UpdateItem(StrictModel):
    target: Target
    servings: PositiveServingCount | Literal["all"] | None = None
    change: ServingChange


class ChangeQuantity(StrictModel):
    target: Target
    mode: Literal["set", "increase", "remove"]
    quantity: int = Field(ge=1)


class RemoveItem(StrictModel):
    target: Target


class ClearDraft(StrictModel):
    pass


class SetOrderInstructions(StrictModel):
    instructions: str


class StartNewOrder(StrictModel):
    pass


class ProposeSubmission(StrictModel):
    pass


class SubmitOrder(StrictModel):
    pass


Operation: TypeAlias = (
    ShowMenu
    | ShowDraft
    | AddItem
    | UpdateItem
    | ChangeQuantity
    | RemoveItem
    | ClearDraft
    | SetOrderInstructions
    | StartNewOrder
    | ProposeSubmission
    | SubmitOrder
)

TOOL_ARGUMENT_MODELS: dict[str, type[StrictModel]] = {
    "show_menu": ShowMenu,
    "show_draft": ShowDraft,
    "add_item": AddItem,
    "update_item": UpdateItem,
    "change_quantity": ChangeQuantity,
    "remove_item": RemoveItem,
    "clear_draft": ClearDraft,
    "set_order_instructions": SetOrderInstructions,
    "start_new_order": StartNewOrder,
    "propose_submission": ProposeSubmission,
    "submit_order": SubmitOrder,
}


class StoredLineSnapshot(StrictModel):
    line_id: str
    item_id: str
    name: str
    quantity: int
    options: dict[str, str]
    extras: list[str]
    instructions: str
    unit_cents: int
    total_cents: int


class DisplayGroupSnapshot(StrictModel):
    line_ids: list[str]
    item_id: str
    name: str
    quantity: int
    options: dict[str, str]
    extras: list[str]
    instructions: str
    unit_cents: int
    total_cents: int


class DraftSnapshot(StrictModel):
    revision: int
    lines: list[StoredLineSnapshot]
    display_groups: list[DisplayGroupSnapshot]
    general_instructions: str
    total_cents: int
    checkout_state: Literal[
        "draft",
        "reviewed",
        "rejected",
        "application_error",
        "not_sent",
        "uncertain",
        "submitted",
    ]


class MenuChoiceSnapshot(StrictModel):
    value: str
    price_delta_cents: int


class MenuOptionSnapshot(StrictModel):
    name: str
    required: bool
    default: str | None
    choices: list[MenuChoiceSnapshot]


class MenuExtraSnapshot(StrictModel):
    value: str
    price_cents: int


class MenuItemSnapshot(StrictModel):
    item_id: str
    name: str
    base_price_cents: int
    options: list[MenuOptionSnapshot]
    extras: list[MenuExtraSnapshot]


class MenuSnapshot(StrictModel):
    items: list[MenuItemSnapshot]


AppliedOperation: TypeAlias = Literal[
    "add_item",
    "update_item",
    "change_quantity",
    "remove_item",
    "clear_draft",
    "set_order_instructions",
    "start_new_order",
]


class AppliedEffect(StrictModel):
    operation: AppliedOperation
    affected_line_ids: list[str] = Field(default_factory=list)
    created_line_ids: list[str] = Field(default_factory=list)
    removed_line_ids: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class DraftState:
    lines: tuple[OrderLine, ...]
    general_instructions: str
    next_line_number: int


@dataclass(frozen=True)
class OperationEffect:
    operation: AppliedOperation
    affected_line_ids: tuple[str, ...] = ()
    created_line_ids: tuple[str, ...] = ()
    removed_line_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Valid:
    candidate: DraftState
    effect: OperationEffect
    changed: bool


class Alternative(StrictModel):
    value: str
    label: str


class ValueConstraint(StrictModel):
    minimum: int | None = None
    maximum: int | None = None


class IncompletePayload(StrictModel):
    outcome: Literal["INCOMPLETE"]
    remedy: Literal["ask_customer"]
    reason: str
    resolution: str
    subject: str | None = None
    key: str | None = None
    alternatives: list[Alternative] | None = None
    constraint: ValueConstraint | None = None


class UnsatisfiablePayload(StrictModel):
    outcome: Literal["UNSATISFIABLE"]
    remedy: Literal["change_request"]
    reason: str
    resolution: str
    subject: str | None = None
    key: str | None = None
    alternatives: list[Alternative] | None = None
    note: str | None = None


class CartInvalidPayload(StrictModel):
    outcome: Literal["CART_INVALID"]
    remedy: Literal["edit_draft"]
    reason: str
    resolution: str
    rule: str
    current_cents: int | None = None
    limit_cents: int | None = None
    excess_cents: int | None = None


class SchemaIssue(StrictModel):
    path: list[str | int]
    message: str


class MalformedPayload(StrictModel):
    outcome: Literal["MALFORMED"]
    remedy: Literal["correct_tool"]
    reason: str
    resolution: str
    tool_name: str
    issues: list[SchemaIssue]


Incomplete = IncompletePayload
Unsatisfiable = UnsatisfiablePayload
CartInvalid = CartInvalidPayload
Malformed = MalformedPayload
ValidationOutcome: TypeAlias = Valid | Incomplete | Unsatisfiable | CartInvalid | Malformed


class AppliedPayload(StrictModel):
    outcome: Literal["APPLIED"]
    effect: AppliedEffect
    draft: DraftSnapshot


class AlreadyAppliedPayload(StrictModel):
    outcome: Literal["ALREADY_APPLIED"]
    operation: str
    draft: DraftSnapshot


class ResultPayload(StrictModel):
    outcome: Literal["RESULT"]
    result: MenuSnapshot | DraftSnapshot


class ReviewedPayload(StrictModel):
    outcome: Literal["REVIEWED"]
    review_id: str
    review: str


class RestaurantItemPayload(StrictModel):
    item_id: str
    quantity: int
    options: dict[str, str]
    extras: list[str]


class RestaurantPayload(StrictModel):
    items: list[RestaurantItemPayload]
    special_instructions: str | None = None


@dataclass(frozen=True)
class ReviewSnapshot:
    review_id: str
    revision: int
    reviewed_at_turn: int
    rendered_order: str
    payload: RestaurantPayload
    _frozen_payload: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_frozen_payload", serialize_payload(self.payload))

    def restaurant_payload(self) -> dict[str, Any]:
        """Return a fresh wire copy of the payload frozen when the review was built."""

        payload = json.loads(self._frozen_payload)
        assert isinstance(payload, dict)
        return cast(dict[str, Any], payload)


ModelValidationPayload: TypeAlias = Annotated[
    Incomplete | Unsatisfiable | CartInvalid | Malformed,
    Field(discriminator="outcome"),
]
ModelResult: TypeAlias = Annotated[
    AppliedPayload
    | AlreadyAppliedPayload
    | ResultPayload
    | ReviewedPayload
    | Incomplete
    | Unsatisfiable
    | CartInvalid
    | Malformed,
    Field(discriminator="outcome"),
]
_MODEL_RESULT_ADAPTER: TypeAdapter[ModelResult] = TypeAdapter(ModelResult)


class SubmittedPayload(StrictModel):
    outcome: Literal["SUBMITTED"]
    receipt: str
    order_id: str | None = None
    reviewed_total_cents: int
    restaurant_total_cents: int | None = None
    estimated_time: str | None = None


class RejectedSubmissionPayload(StrictModel):
    outcome: Literal["REJECTED"]
    remedy: Literal["ask_customer_before_retry"]
    reason: str
    resolution: str


class ApplicationErrorPayload(StrictModel):
    outcome: Literal["APPLICATION_ERROR"]
    remedy: Literal["require_changed_draft"]
    reason: str
    resolution: str


class NotSentPayload(StrictModel):
    outcome: Literal["NOT_SENT"]
    remedy: Literal["require_new_review"]
    reason: str
    resolution: str


class UncertainSubmissionPayload(StrictModel):
    outcome: Literal["UNCERTAIN"]
    remedy: Literal["block_resubmission"]
    reason: str
    resolution: str


SubmissionResult: TypeAlias = Annotated[
    SubmittedPayload
    | RejectedSubmissionPayload
    | ApplicationErrorPayload
    | NotSentPayload
    | UncertainSubmissionPayload,
    Field(discriminator="outcome"),
]
_SUBMISSION_RESULT_ADAPTER: TypeAdapter[SubmissionResult] = TypeAdapter(SubmissionResult)


def _schema_issues(error: ValidationError) -> list[SchemaIssue]:
    return [
        SchemaIssue(path=list(issue["loc"]), message=issue["msg"])
        for issue in error.errors(include_url=False, include_context=False, include_input=False)
    ]


def parse_tool_call(tool_name: str, arguments: object) -> Operation | Malformed:
    """Strictly parse one model tool call without invoking a domain handler."""

    model = TOOL_ARGUMENT_MODELS.get(tool_name)
    if model is None:
        return Malformed(
            outcome="MALFORMED",
            remedy="correct_tool",
            reason="Unknown tool name.",
            resolution="Use a tool defined by protocol version 1.",
            tool_name=tool_name,
            issues=[SchemaIssue(path=[], message=f"Unknown tool: {tool_name}")],
        )
    try:
        return cast(Operation, model.model_validate(arguments))
    except ValidationError as error:
        return Malformed(
            outcome="MALFORMED",
            remedy="correct_tool",
            reason="Arguments do not match the tool schema.",
            resolution="Correct the tool call using the protocol version 1 schema.",
            tool_name=tool_name,
            issues=_schema_issues(error),
        )


def validate_model_result(payload: object) -> ModelResult:
    """Reject any model-result shape outside the version 1 union."""

    return _MODEL_RESULT_ADAPTER.validate_python(payload, strict=True)


def validate_submission_result(payload: object) -> SubmissionResult:
    """Reject any submission-result shape outside the separate version 1 union."""

    return _SUBMISSION_RESULT_ADAPTER.validate_python(payload, strict=True)


def serialize_payload(payload: StrictModel) -> str:
    """Serialize a model-facing payload stably, omitting irrelevant null fields."""

    return json.dumps(
        payload.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )


def protocol_schema() -> dict[str, Any]:
    """Return versioned JSON schemas derived from the parsing models."""

    return {
        "protocol_version": TOOL_PROTOCOL_VERSION,
        "tools": {
            name: model.model_json_schema()
            for name, model in TOOL_ARGUMENT_MODELS.items()
        },
        "model_results": _MODEL_RESULT_ADAPTER.json_schema(),
        "submission_results": _SUBMISSION_RESULT_ADAPTER.json_schema(),
    }
