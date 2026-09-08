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


class MenuQuestion(StrictModel):
    type: Literal["menu"]
    item_ids: list[str] = Field(default_factory=list)


class Unsupported(StrictModel):
    type: Literal["unsupported"]
    reason: Literal["not_available", "unclear", "dietary_guarantee"]


Operation = Annotated[Add | Summary | MenuQuestion | Unsupported, Field(discriminator="type")]


class Proposal(StrictModel):
    operations: list[Operation] = Field(min_length=1)
