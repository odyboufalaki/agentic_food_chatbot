from dataclasses import dataclass, replace
from typing import Any, Literal

from food_ordering.menu import ALIASES, Menu, money
from food_ordering.proposals import Add, ChangeQuantity, Edit, Target


class InvalidSelection(ValueError):
    pass


@dataclass(frozen=True)
class ClarificationContext:
    reason: str
    fallback_question: str
    subject: str | None = None
    field: str | None = None
    choices: tuple[str, ...] = ()

    def snapshot(self) -> dict[str, Any]:
        return {
            "reason": self.reason, "subject": self.subject, "field": self.field,
            "choices": list(self.choices), "draft_changed": False,
        }


@dataclass(frozen=True)
class ResponseContext:
    kind: Literal["acknowledgement", "rejection"]
    request: str
    operations: tuple[dict[str, Any], ...] = ()
    reason: str | None = None

    def snapshot(self) -> dict[str, Any]:
        def public(value: Any) -> Any:
            if isinstance(value, dict):
                return {
                    key: public(part) for key, part in value.items()
                    if key not in {"line_id", "line_ids", "source_line_id"}
                    and not key.endswith("_cents")
                }
            if isinstance(value, list):
                return [public(part) for part in value]
            return value

        return {
            "kind": self.kind, "request": self.request, "reason": self.reason,
            "operations": [public(operation) for operation in self.operations],
            "draft_changed": self.kind == "acknowledgement",
        }


class ClarificationNeeded(InvalidSelection):
    def __init__(self, context: ClarificationContext) -> None:
        self.context = context
        super().__init__(context.fallback_question)


def build_clarification_context(
    reason: str, menu: Menu, *, item_id: str | None = None, field: str | None = None,
) -> ClarificationContext:
    questions = {
        "required_option": "Which required option would you like?",
        "target": "Which selection do you mean?",
        "quantity": "What exact quantity do you mean?",
        "replacement_notes": "Should I keep or discard the existing preparation instructions?",
    }
    resolved_id = ALIASES.get(item_id, item_id) if item_id is not None else None
    item = next((candidate for candidate in menu.menu if candidate.id == resolved_id), None)
    if reason == "replacement_notes":
        return ClarificationContext(
            reason=reason, fallback_question=questions[reason],
            subject=item.name if item else None, field="replacement_notes", choices=("keep", "discard"),
        )
    if item is None:
        return ClarificationContext(reason=reason, fallback_question=questions[reason], field=field or reason)
    if reason == "quantity":
        return ClarificationContext(
            reason=reason,
            fallback_question=f"How many servings of {item.name} would you like?",
            subject=item.name, field="quantity",
        )
    if reason == "target":
        return ClarificationContext(
            reason=reason, fallback_question=f"Which {item.name} selection do you mean?",
            subject=item.name, field="target",
        )
    option = item.options.get(field) if field is not None else None
    if option is not None:
        return ClarificationContext(
            reason=reason,
            fallback_question=f"Choose {field} for {item.name}: {', '.join(option.choices)}.",
            subject=item.name, field=field, choices=tuple(option.choices),
        )
    return ClarificationContext(
        reason=reason, fallback_question=questions[reason], subject=item.name, field=field,
    )


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
            "line_id": self.line_id, "item_id": self.item_id,
            "quantity": self.quantity, "options": dict(self.options),
            "extras": list(self.extras), "instructions": self.instructions,
            "unit_cents": self.unit_cents, "total_cents": self.total_cents,
        }


def normalize(selection: Add, menu: Menu, *, line_id: str) -> OrderLine:
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
                raise ClarificationNeeded(ClarificationContext(
                    reason="required_option",
                    fallback_question=f"Choose {name} for {item.name}: {', '.join(option.choices)}.",
                    subject=item.name, field=name, choices=tuple(option.choices),
                ))
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
    return OrderLine(item.id, item.name, selection.quantity, tuple(options.items()), extras, price,
                     line_id, selection.instructions.strip())


def matching_lines(target: Target, lines: list[OrderLine]) -> list[OrderLine]:
    if target.line_id is None and target.item_id is None:
        raise ClarificationNeeded(ClarificationContext(
            reason="target", fallback_question="Which item or order line do you want to change?",
            subject="order selection", field="target",
        ))
    item_id = ALIASES.get(target.item_id, target.item_id) if target.item_id is not None else None
    matches = [line for line in lines
               if (target.line_id is None or line.line_id == target.line_id)
               and (item_id is None or line.item_id == item_id)
               and target.options.items() <= dict(line.options).items()
               and set(target.extras) <= set(line.extras)
               and (target.instructions is None or line.instructions == target.instructions)]
    if not matches:
        raise InvalidSelection("No order line matches that selection in your draft.")
    return matches


def resolve_target(target: Target, lines: list[OrderLine]) -> OrderLine:
    matches = matching_lines(target, lines)
    if len(matches) != 1:
        choices = tuple(f"{index}: {describe_line(line)}"
                        + (f"; instructions: {line.instructions}" if line.instructions else "")
                        for index, line in enumerate(matches, start=1))
        raise ClarificationNeeded(ClarificationContext(
            reason="target", fallback_question=f"Which matching selection do you mean? {'; '.join(choices)}.",
            subject=matches[0].name, field="target", choices=choices,
        ))
    return matches[0]


def edit_line(operation: Edit, line: OrderLine, menu: Menu) -> OrderLine:
    if (not operation.options and not operation.add_extras and not operation.remove_extras
            and operation.instructions is None and operation.replacement_item_id is None):
        raise InvalidSelection("Specify the options or extras you want to change.")
    if set(operation.add_extras) & set(operation.remove_extras):
        raise InvalidSelection("Specify whether to add or remove each extra.")
    if not set(operation.remove_extras) <= set(line.extras):
        raise InvalidSelection("That extra is not selected on this order line.")
    replacing = (operation.replacement_item_id is not None
                 and ALIASES.get(operation.replacement_item_id, operation.replacement_item_id) != line.item_id)
    if replacing and line.instructions and operation.replacement_notes is None:
        raise ClarificationNeeded(ClarificationContext(
            reason="replacement_notes",
            fallback_question=f"Keep or discard the instructions '{line.instructions}' when replacing {line.name}?",
            subject=line.name, field="replacement_notes", choices=("keep", "discard"),
        ))
    updated = normalize(Add(
        type="add", item_id=operation.replacement_item_id or line.item_id, quantity=line.quantity,
        options=operation.options if replacing else {**dict(line.options), **operation.options},
        extras=(operation.add_extras if replacing else
                sorted((set(line.extras) - set(operation.remove_extras)) | set(operation.add_extras))),
    ), menu, line_id=line.line_id)
    previous_notes = "" if replacing and operation.replacement_notes == "discard" else line.instructions
    return replace(updated, instructions=(previous_notes if operation.instructions is None
                                          else operation.instructions.strip()))


def edit_servings(
    operation: Edit, lines: list[OrderLine], menu: Menu, next_line_number: int,
) -> tuple[list[OrderLine], int, list[dict[str, Any]]]:
    matches = matching_lines(operation.target, lines)
    if operation.servings is None:
        matches = [resolve_target(operation.target, lines)]
    available = sum(line.quantity for line in matches)
    quantity = available if operation.servings in (None, "all") else operation.servings
    if not isinstance(quantity, int) or quantity <= 0 or quantity > available:
        raise InvalidSelection(f"Choose a positive quantity up to {available} matching servings.")
    candidate = list(lines)
    changes = []
    for line in matches:
        if quantity == 0:
            break
        taken = min(quantity, line.quantity)
        updated = edit_line(operation, replace(line, quantity=taken), menu)
        index = candidate.index(line)
        if taken < line.quantity and configuration(updated) != configuration(line):
            updated = replace(updated, line_id=f"L{next_line_number}")
            next_line_number += 1
            portions = [replace(line, quantity=line.quantity - taken), updated]
            candidate[index:index + 1] = portions
            changes.append({"type": "split", "source_line_id": line.line_id,
                            "lines": [portion.snapshot() for portion in portions]})
        else:
            updated = replace(updated, quantity=line.quantity)
            candidate[index] = updated
        changes.append({"type": "edit", **updated.snapshot()})
        quantity -= taken
    return candidate, next_line_number, changes


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


def render_draft(lines: list[OrderLine], instructions: str = "") -> str:
    descriptions = []
    for group in display_groups(lines):
        line = replace(group[0], quantity=sum(part.quantity for part in group))
        notes = f"; instructions: {line.instructions}" if line.instructions else ""
        descriptions.append(f"{describe_line(line)}{notes} — {money(line.total_cents)}")
    if instructions:
        descriptions.append(f"General instructions: {instructions}")
    return "Draft order:\n" + "\n".join(descriptions) + f"\nTotal: {money(sum(line.total_cents for line in lines))}"


def submission_payload(lines: list[OrderLine], menu: Menu, instructions: str = "") -> dict[str, Any]:
    if not lines:
        raise InvalidSelection("An empty order cannot be submitted. Add an item first.")
    validated = [normalize(Add(
        type="add", item_id=line.item_id, quantity=line.quantity,
        options=dict(line.options), extras=list(line.extras),
    ), menu, line_id=line.line_id) for line in lines]
    total = sum(line.total_cents for line in validated)
    if total > 5000:
        raise InvalidSelection(f"The order total is {money(total)}. Reduce it to $50.00 or less before submission.")
    payload: dict[str, Any] = {"items": [{"item_id": line.item_id, "quantity": line.quantity,
                       "options": dict(line.options), "extras": list(line.extras)} for line in validated]}
    notes = [f"Item {index}: {describe_line(line)}: {line.instructions}"
             for index, line in enumerate(lines, 1) if line.instructions]
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
            parts.append("  Extras: " + ", ".join(f"{extra.id} ({money(extra.price)})" for extra in item.extras.choices))
        descriptions.append("\n".join(parts))
    return "Menu (option amounts adjust the base price):\n" + "\n".join(descriptions)
