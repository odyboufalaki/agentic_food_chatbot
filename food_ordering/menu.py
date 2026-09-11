from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import BeforeValidator, Field, model_validator

from food_ordering.models import StrictModel


def cents(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)):
        raise ValueError("Menu prices must be numbers")
    amount = Decimal(value) * 100
    if not amount.is_finite() or amount != amount.to_integral_value():
        raise ValueError("Menu prices must have whole cents")
    return int(amount)


Money = Annotated[int, BeforeValidator(cents)]


class Option(StrictModel):
    type: Literal["single_choice"]
    required: bool
    choices: list[str] = Field(min_length=1)
    default: str | None = None
    price_modifier: dict[str, Money] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_choices(self) -> Self:
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("Duplicate option choices")
        if self.default is not None and self.default not in self.choices:
            raise ValueError("Default must be a supported choice")
        if not self.price_modifier.keys() <= set(self.choices):
            raise ValueError("Price modifier must name a supported choice")
        return self


class Extra(StrictModel):
    id: str = Field(min_length=1)
    price: Money = Field(ge=0)


class Extras(StrictModel):
    type: Literal["multi_choice"]
    choices: list[Extra]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({extra.id for extra in self.choices}) != len(self.choices):
            raise ValueError("Duplicate extra IDs")
        return self


class MenuItem(StrictModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    base_price: Money = Field(ge=0)
    options: dict[str, Option]
    extras: Extras = Field(default_factory=lambda: Extras(type="multi_choice", choices=[]))

    @model_validator(mode="after")
    def nonnegative_unit_price(self) -> Self:
        minimum = self.base_price + sum(
            min(0, *(option.price_modifier.get(choice, 0) for choice in option.choices))
            for option in self.options.values()
        )
        if minimum < 0:
            raise ValueError("Options must not create negative prices")
        return self


class Menu(StrictModel):
    menu: list[MenuItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len({item.id for item in self.menu}) != len(self.menu):
            raise ValueError("Duplicate menu IDs")
        return self


class DecimalLoader(yaml.SafeLoader):
    """Read YAML monetary literals without first rounding through a float."""


def decimal_scalar(loader: DecimalLoader, node: yaml.ScalarNode) -> Decimal:
    return Decimal(loader.construct_scalar(node))


DecimalLoader.add_constructor("tag:yaml.org,2002:float", decimal_scalar)


@lru_cache(maxsize=1)
def load_menu() -> Menu:
    source = Path(__file__).with_name("menu.yaml").read_text(encoding="utf-8")
    return Menu.model_validate(yaml.load(source, Loader=DecimalLoader))


ALIASES = {"burger": "classic_burger", "cola": "soda", "pizza": "margherita"}
def money(value: int) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value) // 100}.{abs(value) % 100:02d}"
