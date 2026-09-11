"""Reusable deterministic Menu, Draft order, pricing, and display behavior."""

from dataclasses import dataclass, replace
import re
from typing import Any, Protocol, TypeAlias

from food_ordering.menu import ALIASES, Menu, money


class InvalidSelection(ValueError):
    pass


class AddSelection(Protocol):
    item_id: str
    quantity: int
    options: dict[str, str]
    extras: list[str]
    instructions: str


@dataclass(frozen=True)
class UnsupportedItem:
    item_id: str


@dataclass(frozen=True)
class UnsupportedOption:
    item_id: str
    item_name: str
    key: str
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class MissingRequiredOption:
    item_id: str
    item_name: str
    key: str
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedOptionValue:
    item_id: str
    item_name: str
    key: str
    value: str
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedExtra:
    item_id: str
    item_name: str
    value: str
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class UnsupportedInstructionAddition:
    item_id: str
    item_name: str
    value: str
    alternatives: tuple[str, ...]


AddSelectionIssue: TypeAlias = (
    UnsupportedItem
    | UnsupportedOption
    | MissingRequiredOption
    | UnsupportedOptionValue
    | UnsupportedExtra
    | UnsupportedInstructionAddition
)


_ADDITION_DIRECTIVE = re.compile(
    r"\b(?:add|with)\s+(?!no\b|without\b)([^,;.]+)",
    re.IGNORECASE,
)
_ADDITION_SEPARATOR = re.compile(r"\s+(?:and|&)\s+", re.IGNORECASE)


def _unsupported_instruction_addition(
    instructions: str,
    selected_extras: tuple[str, ...],
) -> str | None:
    """Return an explicit free-text addition not represented by selected Extras."""

    selected_labels = {extra.casefold().replace("_", " ") for extra in selected_extras}
    for directive in _ADDITION_DIRECTIVE.finditer(instructions):
        for raw_addition in _ADDITION_SEPARATOR.split(directive.group(1)):
            addition = raw_addition.strip().casefold().replace("_", " ")
            addition = re.sub(r"^(?:a|an|the|some)\s+", "", addition)
            addition = re.sub(r"\s+please$", "", addition)
            if addition.startswith(("no ", "without ")):
                continue
            if addition not in selected_labels:
                return raw_addition.strip()
    return None


@dataclass(frozen=True)
class OrderLine:
    item_id: str
    name: str
    quantity: int
    options: tuple[tuple[str, str], ...]
    extras: tuple[str, ...]
    unit_cents: int
    line_id: str
    instructions: str = ""

    @property
    def total_cents(self) -> int:
        return self.quantity * self.unit_cents

    def snapshot(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "item_id": self.item_id,
            "quantity": self.quantity,
            "options": dict(self.options),
            "extras": list(self.extras),
            "instructions": self.instructions,
            "unit_cents": self.unit_cents,
            "total_cents": self.total_cents,
        }


def normalize_add_selection(
    selection: AddSelection,
    menu: Menu,
    *,
    line_id: str,
) -> OrderLine | AddSelectionIssue:
    item_id = ALIASES.get(selection.item_id, selection.item_id)
    item = next((item for item in menu.menu if item.id == item_id), None)
    if item is None:
        return UnsupportedItem(item_id=selection.item_id)
    unsupported_option = next(
        (name for name in selection.options if name not in item.options),
        None,
    )
    if unsupported_option is not None:
        return UnsupportedOption(
            item_id=item.id,
            item_name=item.name,
            key=unsupported_option,
            alternatives=tuple(item.options),
        )
    options = {}
    price = item.base_price
    for name, option in item.options.items():
        value = selection.options.get(name, option.default)
        if value is None:
            if option.required:
                return MissingRequiredOption(
                    item_id=item.id,
                    item_name=item.name,
                    key=name,
                    alternatives=tuple(option.choices),
                )
            continue
        if value not in option.choices:
            return UnsupportedOptionValue(
                item_id=item.id,
                item_name=item.name,
                key=name,
                value=value,
                alternatives=tuple(option.choices),
            )
        options[name] = value
        price += option.price_modifier.get(value, 0)
    extra_prices = {extra.id: extra.price for extra in item.extras.choices}
    extras = tuple(sorted(set(selection.extras)))
    if not set(extras) <= extra_prices.keys():
        unsupported_extra = next(extra for extra in extras if extra not in extra_prices)
        return UnsupportedExtra(
            item_id=item.id,
            item_name=item.name,
            value=unsupported_extra,
            alternatives=tuple(extra_prices),
        )
    price += sum(extra_prices[extra] for extra in extras)
    unsupported_addition = _unsupported_instruction_addition(selection.instructions, extras)
    if unsupported_addition is not None:
        return UnsupportedInstructionAddition(
            item_id=item.id,
            item_name=item.name,
            value=unsupported_addition,
            alternatives=tuple(extra_prices),
        )
    return OrderLine(
        item.id,
        item.name,
        selection.quantity,
        tuple(options.items()),
        extras,
        price,
        line_id,
        selection.instructions.strip(),
    )


def configuration(line: OrderLine) -> tuple[Any, ...]:
    return line.item_id, line.options, line.extras, line.instructions


def display_groups(lines: list[OrderLine]) -> list[list[OrderLine]]:
    groups: dict[tuple[Any, ...], list[OrderLine]] = {}
    for line in lines:
        groups.setdefault(configuration(line), []).append(line)
    return list(groups.values())


def describe_line(line: OrderLine) -> str:
    choices = ", ".join(f"{key}: {value}" for key, value in line.options)
    extras = f"; extras: {', '.join(line.extras)}" if line.extras else ""
    return f"{line.quantity} × {line.name} ({choices}{extras})"


def render_draft(lines: list[OrderLine], instructions: str = "") -> str:
    descriptions = []
    for group in display_groups(lines):
        line = replace(group[0], quantity=sum(part.quantity for part in group))
        notes = f"; instructions: {line.instructions}" if line.instructions else ""
        descriptions.append(f"{describe_line(line)}{notes} — {money(line.total_cents)}")
    if instructions:
        descriptions.append(f"General instructions: {instructions}")
    return (
        "Draft order:\n"
        + "\n".join(descriptions)
        + f"\nTotal: {money(sum(line.total_cents for line in lines))}"
    )


@dataclass
class _StoredSelection:
    item_id: str
    quantity: int
    options: dict[str, str]
    extras: list[str]
    instructions: str


def submission_payload(
    lines: list[OrderLine],
    menu: Menu,
    instructions: str = "",
) -> dict[str, Any]:
    if not lines:
        raise InvalidSelection("An empty order cannot be submitted. Add an item first.")
    validated = []
    for line in lines:
        normalized = normalize_add_selection(
            _StoredSelection(
                item_id=line.item_id,
                quantity=line.quantity,
                options=dict(line.options),
                extras=list(line.extras),
                instructions=line.instructions,
            ),
            menu,
            line_id=line.line_id,
        )
        if not isinstance(normalized, OrderLine):
            raise InvalidSelection("The Draft order contains an invalid selection.")
        validated.append(normalized)
    total = sum(line.total_cents for line in validated)
    if total > 5000:
        raise InvalidSelection(
            f"The order total is {money(total)}. Reduce it to $50.00 or less before submission."
        )
    payload: dict[str, Any] = {
        "items": [{
            "item_id": line.item_id,
            "quantity": line.quantity,
            "options": dict(line.options),
            "extras": list(line.extras),
        } for line in validated],
    }
    notes = [
        f"Item {index}: {describe_line(line)}: {line.instructions}"
        for index, line in enumerate(lines, 1)
        if line.instructions
    ]
    if instructions:
        notes.append(f"General: {instructions}")
    if notes:
        payload["special_instructions"] = "\n".join(notes)
    return payload


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
            parts.append(
                "  Extras: "
                + ", ".join(
                    f"{extra.id} ({money(extra.price)})"
                    for extra in item.extras.choices
                )
            )
        descriptions.append("\n".join(parts))
    return "Menu (option amounts adjust the base price):\n" + "\n".join(descriptions)
