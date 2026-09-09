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


class Summary(StrictModel):
    type: Literal["summary"]


class Target(StrictModel):
    line_id: str | None = None
    item_id: str | None = None
    options: dict[str, str] = Field(default_factory=dict)
    extras: list[str] = Field(default_factory=list)


class Edit(StrictModel):
    type: Literal["edit"]
    target: Target
    options: dict[str, str] = Field(default_factory=dict)
    add_extras: list[str] = Field(default_factory=list)
    remove_extras: list[str] = Field(default_factory=list)


class RemoveLine(StrictModel):
    type: Literal["remove_line"]
    target: Target


class ChangeQuantity(StrictModel):
    type: Literal["set_quantity", "increase_quantity", "remove_units"]
    target: Target
    quantity: int = Field(gt=0)


class ClearDraft(StrictModel):
    type: Literal["clear_draft"]


class MenuQuestion(StrictModel):
    type: Literal["menu"]
    item_ids: list[str] = Field(default_factory=list)


class Unsupported(StrictModel):
    type: Literal["unsupported"]
    reason: Literal["not_available", "unclear", "dietary_guarantee"]


Operation = Annotated[
    Add | Edit | RemoveLine | ChangeQuantity | ClearDraft | Summary | MenuQuestion | Unsupported,
    Field(discriminator="type"),
]


class Proposal(StrictModel):
    operations: list[Operation] = Field(min_length=1)
