from importlib.util import find_spec

import pytest

from agent import FoodOrderAgent
from food_ordering.session import Session


@pytest.mark.parametrize(
    "module_name",
    [
        "food_ordering.proposals",
        "food_ordering.interpretation",
        "food_ordering.clarification",
    ],
)
def test_legacy_architecture_modules_are_not_importable(module_name: str) -> None:
    assert find_spec(module_name) is None


def test_public_agent_and_session_expose_only_the_outcome_driven_path() -> None:
    agent = FoodOrderAgent.__new__(FoodOrderAgent)
    session = Session()

    assert not hasattr(agent, "_interpreter")
    assert not hasattr(session, "pending_change")
    assert not hasattr(session, "history")
