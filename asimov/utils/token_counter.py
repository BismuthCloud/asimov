# utils/approx_token_count.py
from typing import Any, Dict, List
import math

AVG_CHARS_PER_TOKEN = 4        # heuristic—you can tweak if your data skews long/short
TOKENS_PER_MSG      = 4        # ChatML fixed overhead  (role, separators, etc.)
TOKENS_PER_NAME     = -1       # spec quirk: “name” field shaves one token
END_OF_REQ_TOKENS   = 2        # every request implicitly ends with: <assistant|ANSWER>

def approx_tokens_from_serialized_messages(
    serialized_messages: List[Dict[str, Any]],
    avg_chars_per_token: int = AVG_CHARS_PER_TOKEN,
) -> int:
    """
    Fast, model-agnostic token estimate for a ChatML message array.

    Parameters
    ----------
    serialized_messages : list[dict]
        Your [{role, content:[{type,text}]}] structure.
    avg_chars_per_token : int, optional
        How many characters you assume map to one token (default 4).

    Returns
    -------
    int
        Estimated prompt token count.
    """
    total_tokens = 0

    for msg in serialized_messages:
        total_tokens += TOKENS_PER_MSG

        # role string itself
        total_tokens += math.ceil(len(msg["role"]) / avg_chars_per_token)

        if "name" in msg:
            total_tokens += TOKENS_PER_NAME

        for part in msg["content"]:
            if part["type"] == "text":
                total_tokens += math.ceil(len(part["text"]) / avg_chars_per_token)
            else:
                # non-text parts: fall back to raw length heuristic
                total_tokens += math.ceil(len(str(part)) / avg_chars_per_token)

    total_tokens += END_OF_REQ_TOKENS
    return max(total_tokens, 0)
