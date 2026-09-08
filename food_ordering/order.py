from dataclasses import dataclass, field
from uuid import uuid4
from typing import Any

from food_ordering.menu import ALIASES, Menu, money
from food_ordering.proposals import Add


class InvalidSelection(ValueError):
    pass


@dataclass(frozen=True)
class OrderLine:
    item_id: str
    name: str
    quantity: int
    options: tuple[tuple[str, str], ...]
    extras: tuple[str, ...]
    unit_cents: int
    line_id: str = field(default_factory=lambda: uuid4().hex)
    instructions: str = ""

    @property
    def total_cents(self) -> int:
        return self.quantity * self.unit_cents

    def snapshot(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id, "item_id": self.item_id,
            "quantity": self.quantity, "options": dict(self.options),
            "extras": list(self.extras), "instructions": self.instructions,
            "unit_cents": self.unit_cents, "total_cents": self.total_cents,
        }


def normalize(selection: Add, menu: Menu) -> OrderLine:
    item_id = ALIASES.get(selection.item_id, selection.item_id)
    item = next((item for item in menu.menu if item.id == item_id), None)
    if item is None:
        raise InvalidSelection("That item is not on the menu. Please choose a listed item.")
    if not selection.options.keys() <= item.options.keys():
        raise InvalidSelection(f"Unsupported option for {item.name}. Please check the menu.")
    options = {}
    price = item.base_price
    for name, option in item.options.items():
        value = selection.options.get(name, option.default)
        if value is None:
            if option.required:
                raise InvalidSelection(f"Choose {name} for {item.name}: {', '.join(option.choices)}.")
            continue
        if value not in option.choices:
            raise InvalidSelection(f"Choose a supported {name} for {item.name}: {', '.join(option.choices)}.")
        options[name] = value
        price += option.price_modifier.get(value, 0)
    extra_prices = {extra.id: extra.price for extra in item.extras.choices}
    extras = tuple(sorted(set(selection.extras)))
    if not set(extras) <= extra_prices.keys():
        raise InvalidSelection(f"Unsupported extra for {item.name}. Please choose a listed extra.")
    price += sum(extra_prices[extra] for extra in extras)
    return OrderLine(item.id, item.name, selection.quantity, tuple(options.items()), extras, price)


def render_draft(lines: list[OrderLine]) -> str:
    descriptions = []
    for line in lines:
        choices = ", ".join(f"{key}: {value}" for key, value in line.options)
        extras = f"; extras: {', '.join(line.extras)}" if line.extras else ""
        descriptions.append(f"{line.quantity} × {line.name} ({choices}{extras}) — {money(line.total_cents)}")
    return "Draft order:\n" + "\n".join(descriptions) + f"\nTotal: {money(sum(line.total_cents for line in lines))}"


def render_menu(menu: Menu, item_ids: list[str]) -> str:
    requested = {ALIASES.get(item_id, item_id) for item_id in item_ids}
    if not requested <= {item.id for item in menu.menu}:
        raise InvalidSelection("That item is not on the menu. Please choose a listed item.")
    descriptions = []
    for item in menu.menu:
        if requested and item.id not in requested:
            continue
        parts = [f"{item.name}: base {money(item.base_price)}"]
        for name, option in item.options.items():
            choices = ", ".join(
                f"{choice} ({money(option.price_modifier.get(choice, 0))})"
                for choice in option.choices
            )
            requirement = "required" if option.required else "optional"
            default = f"default: {option.default}" if option.default else "no default"
            parts.append(f"  {name} ({requirement}, {default}): {choices}")
        if item.extras.choices:
            parts.append("  Extras: " + ", ".join(f"{extra.id} ({money(extra.price)})" for extra in item.extras.choices))
        descriptions.append("\n".join(parts))
    return "Menu (option amounts adjust the base price):\n" + "\n".join(descriptions)
