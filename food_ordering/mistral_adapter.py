"""Mistral translation boundary for the provider-neutral turn model."""

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import httpx
from mistralai.client import Mistral, models
from mistralai.client.errors import MistralError, ResponseValidationError

from food_ordering.mistral_support import (
    ModelFailure,
    Settings,
    client_context,
    error_category,
    validate_configuration,
)
from food_ordering.model_adapter import (
    AssistantMessage,
    CustomerMessage,
    ModelMessage,
    ToolCall,
    ToolResultMessage,
    ToolSpec,
)


SYSTEM_PROMPT = """You help a customer build a food order using the supplied tools.
The tool schemas are authoritative. Use tool calls for all Menu and Draft facts
and for every requested operation. Never invent an argument that the customer
must choose. Read each tool result before continuing. Only return a final plain
text response when no more tool calls are needed."""


class MistralToolModel:
    """Translate application messages and tools to one Mistral completion."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Mistral | None = None,
    ) -> None:
        self._settings = settings if settings is not None else Settings.from_env()
        self._client = client

    def complete(
        self,
        *,
        messages: Sequence[ModelMessage],
        tools: Sequence[ToolSpec],
    ) -> AssistantMessage:
        self._validate_configuration()
        provider_messages = self._provider_messages(messages)
        provider_tools: list[models.ChatCompletionRequestTool] = [
            models.Tool(function=models.Function(
                name=tool.name,
                description=tool.description,
                parameters=dict(tool.parameters),
            ))
            for tool in tools
        ]
        with client_context(self._settings, self._client) as client:
            for attempt in range(2):
                try:
                    response = client.chat.complete(
                        model=self._settings.model,
                        messages=provider_messages,
                        tools=provider_tools,
                        tool_choice="auto",
                        parallel_tool_calls=True,
                        temperature=0,
                        retries=None,
                        timeout_ms=self._settings.timeout_ms,
                        max_tokens=4096,
                    )
                    return self._assistant_message(response)
                except httpx.TimeoutException:
                    category = "model_timeout"
                except httpx.TransportError:
                    category = "model_unavailable"
                except MistralError as error:
                    category = error_category(error)
                    if category in {"authentication", "configuration", "model_error"}:
                        raise ModelFailure(category) from None
                except (ResponseValidationError, ValueError, TypeError):
                    raise ModelFailure("invalid_structured_output") from None
                if attempt == 1:
                    raise ModelFailure(category) from None
        raise ModelFailure("model_error")

    def _validate_configuration(self) -> None:
        validate_configuration(self._settings, self._client)

    @staticmethod
    def _provider_messages(
        messages: Sequence[ModelMessage],
    ) -> list[models.ChatCompletionRequestMessage]:
        translated: list[models.ChatCompletionRequestMessage] = [
            models.SystemMessage(content=SYSTEM_PROMPT),
        ]
        for message in messages:
            if isinstance(message, CustomerMessage):
                translated.append(models.UserMessage(content=message.content))
            elif isinstance(message, AssistantMessage):
                tool_calls = [
                    models.ToolCall(
                        id=call.call_id,
                        function=models.FunctionCall(
                            name=call.name,
                            arguments=json.dumps(
                                call.arguments,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        ),
                    )
                    for call in message.tool_calls
                ]
                translated.append(models.AssistantMessage(
                    content=message.content or None,
                    tool_calls=tool_calls or None,
                ))
            else:
                translated.append(models.ToolMessage(
                    name=message.name,
                    tool_call_id=message.call_id,
                    content=json.dumps(
                        message.payload.model_dump(mode="json", exclude_none=True),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ))
        return translated

    @staticmethod
    def _assistant_message(response: models.ChatCompletionResponse) -> AssistantMessage:
        if not response.choices:
            raise ValueError("Mistral returned no assistant choice")
        choice = response.choices[0]
        provider_message = choice.message
        if provider_message is None:
            raise ValueError("Mistral returned no assistant message")
        content = provider_message.content
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            raise ValueError("Mistral returned unsupported assistant content")
        calls = tuple(
            ToolCall(
                call_id="" if call.id in {None, "null"} else call.id,
                name=call.function.name,
                arguments=MistralToolModel._decode_arguments(call.function.arguments),
            )
            for call in (provider_message.tool_calls or [])
        )
        completion_status: Literal["complete", "truncated"] = (
            "truncated"
            if choice.finish_reason in {"length", "model_length", "error"}
            else "complete"
        )
        return AssistantMessage(
            content=text,
            tool_calls=calls,
            completion_status=completion_status,
        )

    @staticmethod
    def _decode_arguments(arguments: str | Mapping[str, Any]) -> object:
        if not isinstance(arguments, str):
            return dict(arguments)
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
