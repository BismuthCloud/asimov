import enum

ANTHROPIC_MODELS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
]


OAI_REASONING_MODELS = ["o1-mini", "o1-preview"]

OAI_GPT_MODELS = ["gpt-4o", "gpt-4o-turbo"]


LLAMA_MODELS = ["hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"]


class ModelFamily(enum.Enum):
    OAI_REASONING = "OAI_REASONING"
    OAI_GPT = "OAI_GPT"
    ANTHROPIC = "ANTHROPIC"
    LLAMA = "LLAMA"
