import json
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from food_ordering.tool_protocol import (
    AddItem,
    AlreadyAppliedPayload,
    ApplicationErrorPayload,
    AppliedEffect,
    AppliedPayload,
    CartInvalid,
    ChangeQuantity,
    ClearDraft,
    CustomizeServings,
    DisplayGroupSnapshot,
    DraftSnapshot,
    DraftState,
    Incomplete,
    LineTarget,
    Malformed,
    MatchTarget,
    MenuSnapshot,
    NotSentPayload,
    OperationEffect,
    ProposeSubmission,
    RejectedSubmissionPayload,
    RemoveItem,
    ReplaceServings,
    RestaurantItemPayload,
    RestaurantPayload,
    ResultPayload,
    ReviewedPayload,
    ReviewSnapshot,
    SetOrderInstructions,
    SchemaIssue,
    ShowDraft,
    ShowMenu,
    StartNewOrder,
    StoredLineSnapshot,
    StrictModel,
    SubmitOrder,
    SubmittedPayload,
    TOOL_PROTOCOL_VERSION,
    UncertainSubmissionPayload,
    Unsatisfiable,
    UpdateItem,
    Valid,
    parse_tool_call,
    protocol_schema,
    serialize_payload,
    validate_model_result,
    validate_submission_result,
)
from food_ordering.order import OrderLine


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_type"),
    [
        ("show_menu", {}, ShowMenu),
        ("show_menu", {"item_ids": ["fries"]}, ShowMenu),
        ("show_draft", {}, ShowDraft),
        ("add_item", {"item_id": "fries", "quantity": 1}, AddItem),
        (
            "update_item",
            {
                "target": {"type": "line", "line_id": "L1"},
                "change": {"type": "customize", "add_extras": ["cheese"]},
            },
            UpdateItem,
        ),
        (
            "update_item",
            {
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": 2,
                "change": {"type": "customize", "instructions": "well done"},
            },
            UpdateItem,
        ),
        (
            "update_item",
            {
                "target": {"type": "match", "item_id": "classic_burger"},
                "servings": "all",
                "change": {"type": "replace", "item_id": "fries"},
            },
            UpdateItem,
        ),
        (
            "change_quantity",
            {"target": {"type": "line", "line_id": "L1"}, "mode": "set", "quantity": 2},
            ChangeQuantity,
        ),
        ("remove_item", {"target": {"type": "line", "line_id": "L1"}}, RemoveItem),
        ("clear_draft", {}, ClearDraft),
        ("set_order_instructions", {"instructions": "no cutlery"}, SetOrderInstructions),
        ("start_new_order", {}, StartNewOrder),
        ("propose_submission", {}, ProposeSubmission),
        ("submit_order", {}, SubmitOrder),
    ],
)
def test_every_version_one_tool_parses_strict_arguments(
    tool_name: str, arguments: object, expected_type: type[object],
) -> None:
    parsed = parse_tool_call(tool_name, arguments)

    assert isinstance(parsed, expected_type)


@pytest.mark.parametrize(
    ("tool_name", "arguments", "path"),
    [
        ("unknown", {}, []),
        ("show_draft", {"extra": True}, ["extra"]),
        ("add_item", {"item_id": "fries"}, ["quantity"]),
        ("add_item", {"item_id": "fries", "quantity": True}, ["quantity"]),
        ("add_item", {"item_id": "fries", "quantity": 0}, ["quantity"]),
        (
            "update_item",
            {"target": {"type": "recent", "line_id": "L1"}, "change": {"type": "customize", "instructions": ""}},
            ["target"],
        ),
        (
            "update_item",
            {"target": {"type": "line", "line_id": "L1"}, "servings": 0, "change": {"type": "customize", "instructions": ""}},
            ["servings", "constrained-int"],
        ),
        (
            "update_item",
            {"target": {"type": "line", "line_id": "L1"}, "change": {"type": "customize"}},
            [],
        ),
    ],
)
def test_malformed_and_unknown_tool_calls_are_returned_as_structured_malformed(
    tool_name: str, arguments: object, path: list[str | int],
) -> None:
    result = parse_tool_call(tool_name, arguments)

    assert isinstance(result, Malformed)
    assert result.outcome == "MALFORMED"
    assert result.remedy == "correct_tool"
    assert result.tool_name == tool_name
    assert result.issues
    if path:
        assert any(issue.path[: len(path)] == path for issue in result.issues)


def test_targets_and_serving_changes_are_discriminated_and_keep_quantity_separate() -> None:
    line = LineTarget(type="line", line_id="L8")
    match = MatchTarget(type="match", item_id="milkshake", options={"flavor": "vanilla"})
    customize = CustomizeServings(type="customize", options={"size": "large"}, instructions=None)
    replacement = ReplaceServings(type="replace", item_id="fries")

    assert line.model_dump() == {"type": "line", "line_id": "L8"}
    assert match.model_dump() == {
        "type": "match",
        "item_id": "milkshake",
        "options": {"flavor": "vanilla"},
        "extras": [],
        "instructions": None,
    }
    assert customize.instructions is None
    assert replacement.instructions == ""
    assert "quantity" not in UpdateItem.model_fields
    assert "servings" not in ChangeQuantity.model_fields


def test_internal_validation_outcomes_have_the_approved_discriminators() -> None:
    candidate = DraftState(
        lines=(
            OrderLine(
                item_id="fries", name="French Fries", quantity=1,
                options=(("size", "regular"),), extras=(), unit_cents=350,
                line_id="L1",
            ),
        ),
        general_instructions="",
        next_line_number=2,
    )
    valid = Valid(
        candidate=candidate,
        effect=OperationEffect(operation="add_item", created_line_ids=("L1",)),
        changed=True,
    )
    incomplete = Incomplete(
        outcome="INCOMPLETE", remedy="ask_customer",
        reason="Choose a flavor.", resolution="Ask which flavor.",
    )
    unsatisfiable = Unsatisfiable(
        outcome="UNSATISFIABLE", remedy="change_request",
        reason="Not on menu.", resolution="Choose a listed item.",
    )
    invalid = CartInvalid(
        outcome="CART_INVALID",
        remedy="edit_draft",
        reason="The draft exceeds the checkout limit.",
        resolution="Reduce the draft total.",
        rule="maximum_total",
        current_cents=5100,
        limit_cents=5000,
        excess_cents=100,
    )

    assert valid.changed is True
    assert valid.candidate.lines[0].line_id == "L1"
    assert incomplete.model_dump(exclude_none=True)["outcome"] == "INCOMPLETE"
    assert unsatisfiable.model_dump(exclude_none=True)["outcome"] == "UNSATISFIABLE"
    assert invalid.model_dump(exclude_none=True)["outcome"] == "CART_INVALID"


def _draft() -> DraftSnapshot:
    line = StoredLineSnapshot(
        line_id="L1",
        item_id="fries",
        name="French Fries",
        quantity=1,
        options={"size": "regular"},
        extras=[],
        instructions="",
        unit_cents=350,
        total_cents=350,
    )
    group = DisplayGroupSnapshot(
        line_ids=["L1"],
        item_id="fries",
        name="French Fries",
        quantity=1,
        options={"size": "regular"},
        extras=[],
        instructions="",
        unit_cents=350,
        total_cents=350,
    )
    return DraftSnapshot(
        revision=1,
        lines=[line],
        display_groups=[group],
        general_instructions="",
        total_cents=350,
        checkout_state="draft",
    )


@pytest.mark.parametrize(
    "payload",
    [
        AppliedPayload(
            outcome="APPLIED",
            effect=AppliedEffect(operation="add_item", created_line_ids=["L1"]),
            draft=_draft(),
        ),
        AlreadyAppliedPayload(
            outcome="ALREADY_APPLIED", operation="set_order_instructions", draft=_draft(),
        ),
        ResultPayload(outcome="RESULT", result=MenuSnapshot(items=[])),
        ReviewedPayload(outcome="REVIEWED", review_id="R1", review="Order review"),
        Incomplete(
            outcome="INCOMPLETE", remedy="ask_customer",
            reason="Choose one.", resolution="Ask the customer.",
        ),
        Unsatisfiable(
            outcome="UNSATISFIABLE", remedy="change_request",
            reason="Unavailable.", resolution="Change the request.",
        ),
        CartInvalid(
            outcome="CART_INVALID", remedy="edit_draft",
            reason="Too expensive.", resolution="Edit the draft.", rule="maximum_total",
        ),
        Malformed(
            outcome="MALFORMED",
            remedy="correct_tool",
            reason="Arguments do not match the tool schema.",
            resolution="Correct the tool call.",
            tool_name="add_item",
            issues=[SchemaIssue(path=["quantity"], message="Field required")],
        ),
    ],
)
def test_every_model_result_discriminator_round_trips_with_stable_serialization(
    payload: StrictModel,
) -> None:
    serialized = serialize_payload(payload)

    assert validate_model_result(json.loads(serialized)).outcome == payload.model_dump()["outcome"]
    assert serialize_payload(validate_model_result(json.loads(serialized))) == serialized
    assert ":null" not in serialized


@pytest.mark.parametrize(
    "payload",
    [
        SubmittedPayload(outcome="SUBMITTED", receipt="Accepted", reviewed_total_cents=350),
        RejectedSubmissionPayload(
            outcome="REJECTED", remedy="ask_customer_before_retry",
            reason="Kitchen declined.", resolution="Ask before retrying.",
        ),
        ApplicationErrorPayload(
            outcome="APPLICATION_ERROR", remedy="require_changed_draft",
            reason="Invalid request.", resolution="Change the draft.",
        ),
        NotSentPayload(
            outcome="NOT_SENT", remedy="require_new_review",
            reason="Connection failed before sending.", resolution="Review again.",
        ),
        UncertainSubmissionPayload(
            outcome="UNCERTAIN", remedy="block_resubmission",
            reason="Acceptance is unknown.", resolution="Do not resubmit.",
        ),
    ],
)
def test_submission_outcomes_are_a_separate_strict_union(payload: StrictModel) -> None:
    dumped = payload.model_dump(mode="json", exclude_none=True)

    assert validate_submission_result(dumped).outcome == dumped["outcome"]
    with pytest.raises(ValidationError):
        validate_model_result(dumped)


@pytest.mark.parametrize("validator", [validate_model_result, validate_submission_result])
def test_unknown_outcome_shapes_are_rejected(
    validator: Callable[[object], object],
) -> None:
    with pytest.raises(ValidationError):
        validator({"outcome": "FUTURE", "data": {}})


def test_protocol_schema_is_explicitly_versioned_and_derived_from_strict_models() -> None:
    schema = protocol_schema()

    assert TOOL_PROTOCOL_VERSION == 1
    assert schema["protocol_version"] == 1
    assert set(schema["tools"]) == {
        "show_menu",
        "show_draft",
        "add_item",
        "update_item",
        "change_quantity",
        "remove_item",
        "clear_draft",
        "set_order_instructions",
        "start_new_order",
        "propose_submission",
        "submit_order",
    }
    assert schema["tools"]["add_item"]["additionalProperties"] is False
    assert set(schema["tools"]["add_item"]["required"]) == {"item_id", "quantity"}
    assert schema["tools"]["show_draft"].get("required", []) == []


def test_tool_schemas_have_exact_required_and_omittable_fields() -> None:
    required = {
        "show_menu": set(),
        "show_draft": set(),
        "add_item": {"item_id", "quantity"},
        "update_item": {"target", "change"},
        "change_quantity": {"target", "mode", "quantity"},
        "remove_item": {"target"},
        "clear_draft": set(),
        "set_order_instructions": {"instructions"},
        "start_new_order": set(),
        "propose_submission": set(),
        "submit_order": set(),
    }

    assert {
        name: set(schema.get("required", []))
        for name, schema in protocol_schema()["tools"].items()
    } == required


def test_target_change_and_serving_variants_have_exact_required_fields() -> None:
    assert {name for name, field in LineTarget.model_fields.items() if field.is_required()} == {
        "type", "line_id",
    }
    assert {name for name, field in MatchTarget.model_fields.items() if field.is_required()} == {
        "type", "item_id",
    }
    assert {
        name for name, field in CustomizeServings.model_fields.items() if field.is_required()
    } == {"type"}
    assert {
        name for name, field in ReplaceServings.model_fields.items() if field.is_required()
    } == {"type", "item_id"}

    omitted = parse_tool_call("update_item", {
        "target": {"type": "line", "line_id": "L1"},
        "change": {"type": "customize", "instructions": ""},
    })
    positive = parse_tool_call("update_item", {
        "target": {"type": "match", "item_id": "classic_burger"},
        "servings": 2,
        "change": {"type": "customize", "instructions": "well done"},
    })
    all_servings = parse_tool_call("update_item", {
        "target": {"type": "match", "item_id": "classic_burger"},
        "servings": "all",
        "change": {"type": "customize", "instructions": "well done"},
    })

    assert isinstance(omitted, UpdateItem) and omitted.servings is None
    assert isinstance(positive, UpdateItem) and positive.servings == 2
    assert isinstance(all_servings, UpdateItem) and all_servings.servings == "all"


def test_model_outcome_discriminators_and_remedies_are_required() -> None:
    model_types = (
        AppliedPayload,
        AlreadyAppliedPayload,
        ResultPayload,
        ReviewedPayload,
        Incomplete,
        Unsatisfiable,
        CartInvalid,
        Malformed,
        SubmittedPayload,
        RejectedSubmissionPayload,
        ApplicationErrorPayload,
        NotSentPayload,
        UncertainSubmissionPayload,
    )

    for model in model_types:
        assert model.model_fields["outcome"].is_required()
    for model in (
        Incomplete,
        Unsatisfiable,
        CartInvalid,
        Malformed,
        RejectedSubmissionPayload,
        ApplicationErrorPayload,
        NotSentPayload,
        UncertainSubmissionPayload,
    ):
        assert model.model_fields["remedy"].is_required()


def test_review_snapshot_freezes_python_owned_review_and_restaurant_payload() -> None:
    review = ReviewSnapshot(
        review_id="R1",
        revision=3,
        reviewed_at_turn=4,
        rendered_order="Order review",
        payload=RestaurantPayload(items=[RestaurantItemPayload(
            item_id="fries", quantity=1, options={"size": "regular"}, extras=[],
        )]),
    )

    assert review.payload.items[0].item_id == "fries"
    review.payload.items[0].quantity = 9
    assert review.restaurant_payload() == {
        "items": [{
            "item_id": "fries",
            "quantity": 1,
            "options": {"size": "regular"},
            "extras": [],
        }],
    }


def test_forbidden_migration_fields_are_absent_from_the_parallel_contract() -> None:
    schema_text = json.dumps(protocol_schema(), sort_keys=True)

    for forbidden in (
        "pending",
        "corrected_fields",
        "resume_token",
        "suggested_operation",
        "remainder",
        "deferred_tool_calls",
    ):
        assert forbidden not in schema_text
