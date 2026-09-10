from functools import cache
from typing import Any

from food_ordering.menu import ALIASES, Menu
from food_ordering.order import InvalidSelection
from food_ordering.proposals import Add, ChangeQuantity, Clarify, Edit, Operation, Proposal, RemoveLine, Review, Submit


def validate_pending_resolution(previous: Proposal, proposed: Proposal, menu: Menu) -> None:
    """Preserve resolved fields; additions may partition an established quantity.

    corrected_fields names fields the customer explicitly revises (e.g.
    '0.quantity'). The interpreter supplies that intent, just as it supplies
    selections; otherwise Python requires the previous values to survive.
    """
    def with_defaults(operation: Operation) -> Operation:
        if not isinstance(operation, Add):
            return operation
        item = next((item for item in menu.menu if item.id == canonical(operation.item_id)), None)
        defaults = ({name: option.default for name, option in item.options.items()
                     if option.default is not None} if item is not None else {})
        return operation.model_copy(update={"options": {**defaults, **operation.options}})

    established = [with_defaults(operation) for operation in previous.operations]
    following = [with_defaults(operation) for operation in proposed.operations
                 if not isinstance(operation, (Review, Submit))]

    @cache
    def preserves_from(index: int, cursor: int) -> bool:
        if index == len(established):
            return cursor == len(following)
        old = established[index]
        if isinstance(old, (Review, Submit)):
            return preserves_from(index + 1, cursor)
        if cursor == len(following):
            return False
        new = following[cursor]
        if isinstance(old, Clarify):
            if isinstance(new, Clarify):
                return preserves_from(index + 1, cursor + 1)
            item_id = (new.item_id if isinstance(new, Add) else new.target.item_id
                       if isinstance(new, (Edit, ChangeQuantity, RemoveLine)) else None)
            if old.item_id is not None and (item_id is None or canonical(item_id) != canonical(old.item_id)):
                return False
            # One unresolved selection can become several flavor allocations.
            for end in range(cursor + 1, len(following) + 1):
                if end > cursor + 1:
                    part = following[end - 1]
                    if not isinstance(new, Add) or not isinstance(part, Add) or canonical(part.item_id) != canonical(new.item_id):
                        break
                if preserves_from(index + 1, end):
                    return True
            return False
        if isinstance(old, Add):
            corrected_quantity = f"{index}.quantity" in proposed.corrected_fields
            quantity = 0
            for end in range(cursor, len(following)):
                part = following[end]
                if not isinstance(part, Add) or not fields_preserved(old, part, index, proposed.corrected_fields, split=True):
                    break
                quantity += part.quantity
                if (corrected_quantity or quantity == old.quantity) and preserves_from(index + 1, end + 1):
                    return True
                if not corrected_quantity and quantity >= old.quantity:
                    break
            return False
        return fields_preserved(old, new, index, proposed.corrected_fields) and preserves_from(index + 1, cursor + 1)

    if not preserves_from(0, 0):
        raise InvalidSelection("Please preserve the resolved quantity, target, options, extras, and instructions unless explicitly changing them.")


def canonical(item_id: str) -> str:
    return ALIASES.get(item_id, item_id)


def fields_preserved(
    old: Operation, new: Operation, index: int, corrections: list[str], *, split: bool = False,
) -> bool:
    before, after = old.model_dump(), new.model_dump()
    if split:
        before.pop("quantity")
        after.pop("quantity")

    def matches(value: Any, replacement: Any, path: str) -> bool:
        if path in corrections or value is None:
            return True
        if isinstance(value, dict):
            return isinstance(replacement, dict) and all(
                key in replacement and matches(part, replacement[key], f"{path}.{key}")
                for key, part in value.items()
            )
        if path.endswith("item_id") and isinstance(value, str) and isinstance(replacement, str):
            return canonical(value) == canonical(replacement)
        if isinstance(value, list) and isinstance(replacement, list):
            if path.endswith(".target.extras"):
                return set(value) <= set(replacement)
            return set(value) == set(replacement)
        return bool(value == replacement)

    return matches(before, after, str(index))
