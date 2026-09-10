from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, revalidate_instances="always")


class Add(StrictModel):
    type: Literal["add"]
    item_id: str
    quantity: int = Field(gt=0)
    options: dict[str, str] = Field(default_factory=dict)
    extras: list[str] = Field(default_factory=list)
    instructions: str = ""


class Summary(StrictModel):
    type: Literal["summary"]


class Target(StrictModel):
    line_id: str | None = None
    item_id: str | None = None
    options: dict[str, str] = Field(default_factory=dict)
    extras: list[str] = Field(default_factory=list)
    instructions: str | None = None


class Edit(StrictModel):
    type: Literal["edit"]
    target: Target
    options: dict[str, str] = Field(default_factory=dict)
    add_extras: list[str] = Field(default_factory=list)
    remove_extras: list[str] = Field(default_factory=list)
    quantity: int | Literal["all"] | None = Field(default=None)
    instructions: str | None = None
    replacement_item_id: str | None = None
    replacement_notes: Literal["keep", "discard"] | None = None


class RemoveLine(StrictModel):
    type: Literal["remove_line"]
    target: Target


class ChangeQuantity(StrictModel):
    type: Literal["set_quantity", "increase_quantity", "remove_units"]
    target: Target
    quantity: int = Field(gt=0)


class ClearDraft(StrictModel):
    type: Literal["clear_draft"]


class SetInstructions(StrictModel):
    type: Literal["set_instructions"]
    instructions: str


class Submit(StrictModel):
    type: Literal["submit", "confirm"]


class Review(StrictModel):
    type: Literal["review"]


class NewOrder(StrictModel):
    type: Literal["new_order"]


class RetrySubmission(StrictModel):
    type: Literal["retry_submission"]


class Clarify(StrictModel):
    type: Literal["clarify"]
    reason: Literal["required_option", "target", "quantity", "replacement_notes"]
    item_id: str | None = None
    field: str | None = None


class CancelPending(StrictModel):
    type: Literal["cancel_pending"]


class AbandonPending(StrictModel):
    type: Literal["abandon_pending"]


class MenuQuestion(StrictModel):
    type: Literal["menu"]
    item_ids: list[str] = Field(default_factory=list)


class Unsupported(StrictModel):
    type: Literal["unsupported"]
    reason: Literal["not_available", "unclear", "dietary_guarantee"]


Operation = Annotated[
    Add | Edit | RemoveLine | ChangeQuantity | ClearDraft | Submit | Review | NewOrder | RetrySubmission |
    Clarify | CancelPending | AbandonPending | Summary | MenuQuestion | Unsupported | SetInstructions,
    Field(discriminator="type"),
]


class Proposal(StrictModel):
    operations: list[Operation] = Field(min_length=1)
    corrected_fields: list[Annotated[str, Field(pattern=(
        r"^\d+\.(quantity|item_id|instructions|extras|add_extras|remove_extras|"
        r"replacement_item_id|replacement_notes|options\.[a-z_]+|"
        r"target\.(line_id|item_id|instructions|extras|options\.[a-z_]+))$"
    ))]] = Field(default_factory=list)
