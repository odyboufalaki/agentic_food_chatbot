from dataclasses import dataclass, field, replace
from uuid import uuid4
from typing import Any

from food_ordering.menu import ALIASES, Menu, money
from food_ordering.proposals import Add, ChangeQuantity, Edit, Target


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


def resolve_target(target: Target, lines: list[OrderLine]) -> OrderLine:
    if target.line_id is None and target.item_id is None:
        raise InvalidSelection("Identify the item or order line you want to change.")
    item_id = ALIASES.get(target.item_id, target.item_id) if target.item_id is not None else None
    matches = [line for line in lines
               if (target.line_id is None or line.line_id == target.line_id)
               and (item_id is None or line.item_id == item_id)
               and target.options.items() <= dict(line.options).items()
               and set(target.extras) <= set(line.extras)]
    if not matches:
        raise InvalidSelection("No order line matches that selection in your draft.")
    if len(matches) != 1:
        raise InvalidSelection("More than one order line matches. Please identify which selection to change.")
    return matches[0]


def edit_line(operation: Edit, line: OrderLine, menu: Menu) -> OrderLine:
    if not operation.options and not operation.add_extras and not operation.remove_extras:
        raise InvalidSelection("Specify the options or extras you want to change.")
    if set(operation.add_extras) & set(operation.remove_extras):
        raise InvalidSelection("Specify whether to add or remove each extra.")
    if not set(operation.remove_extras) <= set(line.extras):
        raise InvalidSelection("That extra is not selected on this order line.")
    updated = normalize(Add(
        type="add", item_id=line.item_id, quantity=line.quantity,
        options={**dict(line.options), **operation.options},
        extras=sorted((set(line.extras) - set(operation.remove_extras)) | set(operation.add_extras)),
    ), menu)
    return replace(updated, line_id=line.line_id, instructions=line.instructions)


def change_quantity(operation: ChangeQuantity, line: OrderLine) -> OrderLine | None:
    if operation.type == "set_quantity":
        quantity = operation.quantity
    elif operation.type == "increase_quantity":
        quantity = line.quantity + operation.quantity
    else:
        if operation.quantity > line.quantity:
            raise InvalidSelection(f"Only {line.quantity} servings are selected on that order line.")
        quantity = line.quantity - operation.quantity
    return replace(line, quantity=quantity) if quantity else None


def render_draft(lines: list[OrderLine]) -> str:
    descriptions = []
    for line in lines:
        choices = ", ".join(f"{key}: {value}" for key, value in line.options)
        extras = f"; extras: {', '.join(line.extras)}" if line.extras else ""
        descriptions.append(f"{line.quantity} × {line.name} ({choices}{extras}) — {money(line.total_cents)}")
    return "Draft order:\n" + "\n".join(descriptions) + f"\nTotal: {money(sum(line.total_cents for line in lines))}"


def submission_payload(lines: list[OrderLine], menu: Menu) -> dict[str, Any]:
    if not lines:
        raise InvalidSelection("An empty order cannot be submitted. Add an item first.")
    validated = [normalize(Add(
        type="add", item_id=line.item_id, quantity=line.quantity,
        options=dict(line.options), extras=list(line.extras),
    ), menu) for line in lines]
    total = sum(line.total_cents for line in validated)
    if total > 5000:
        raise InvalidSelection(f"The order total is {money(total)}. Reduce it to $50.00 or less before submission.")
    return {"items": [{"item_id": line.item_id, "quantity": line.quantity,
                       "options": dict(line.options), "extras": list(line.extras)} for line in validated]}


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
