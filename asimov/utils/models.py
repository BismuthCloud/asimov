from asimov.constants import (
    ModelFamily,
    OAI_REASONING_MODELS,
    OAI_GPT_MODELS,
    ANTHROPIC_MODELS,
    LLAMA_MODELS,
)
from asimov.services.inference_clients import InferenceClient, ChatMessage


def get_model_family(client: InferenceClient):
    if client.model in OAI_GPT_MODELS:
        return ModelFamily.OAI_GPT
    elif client.model in OAI_REASONING_MODELS:
        return ModelFamily.OAI_REASONING
    elif client.model in ANTHROPIC_MODELS:
        return ModelFamily.ANTHROPIC
    elif client.model in LLAMA_MODELS:
        return ModelFamily.LLAMA
    else:
        return None


def is_model_family(client: InferenceClient, families: list[ModelFamily]) -> bool:
    model_family = get_model_family(client)

    return model_family in families


def prepare_model_generation_input(
    messages: list[ChatMessage], client: InferenceClient
):
    final_output = []
    client_model_family = get_model_family(client)

    for message in messages:
        if not message.model_families or client_model_family in message.model_families:
            final_output.append(message)

    return final_output
