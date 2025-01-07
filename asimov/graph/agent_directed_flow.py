import textwrap
import logging
from typing import List, Dict, Any, Sequence, Optional
from pydantic import Field, PrivateAttr, model_validator
import json
import asyncio

from asimov.caches.cache import Cache
from asimov.graph import (
    FlowControlModule,
    FlowDecision,
    FlowControlConfig,
    CompositeModule,  # noqa
    Node,  # noqa
)
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

MAKE_DECISION_SCHEMA = {
    "name": "make_decision",
    "description": "Make decision about what choice to make from the provided options in the prompt.",
    "input_schema": {
        "type": "object",
        "properties": {
            "thoughts": {
                "type": "string",
                "description": "Your thoughts about why you are making the decsiion you are mkaing regarding the possible choices available.",
            },
            "decision": {
                "type": "string",
                "description": "A phrase to search for exact case sensitive matches over the codebase.",
            },
        },
        "required": ["decision"],
    },
}


class Example:
    def __init__(
        self,
        message: str,
        choices: List[Dict[str, str]],
        choice: str,
        reasoning: str = "",
    ):
        self.message = message
        self.choices = choices
        self.choice = choice
        self.reasoning = reasoning


class AgentDrivenFlowDecision(FlowDecision):
    examples: List[Example] = Field(default_factory=list)


class AgentDrivenFlowControlConfig(FlowControlConfig):  # type: ignore[override]
    decisions: Sequence[AgentDrivenFlowDecision]
    default: Optional[str] = None


class AgentDirectedFlowControl(FlowControlModule):  # type: ignore[override]
    inference_client: InferenceClient
    system_description: str
    flow_config: AgentDrivenFlowControlConfig
    input_var: str = Field(default="input_message")
    use_historicals: bool = Field(default=True)
    voters: int = 1
    _prompt: str = PrivateAttr()
    _logger: logging.Logger = PrivateAttr()

    @model_validator(mode="after")
    def set_attrs(self):
        examples = []

        for decision in self.flow_config.decisions:
            prompt = ""
            for _, example in enumerate(decision.examples, 1):
                prompt += textwrap.dedent(
                    f"""
                    Message: {example.message}
                    Choices:
                    {json.dumps(example.choices, indent=2)}
                    Choice: {example.choice}
                    """
                ).strip()

                if example.reasoning != "":
                    prompt += f"\nReasoning: {example.reasoning}"

                examples.append(prompt)

        self._prompt = f"""You are going to be provided with some input for a user, you need to decide based on the available options which will also be provided what option best fits the message the user sent.
Users may ask for tasks to be completed and not every task may be possible given the available options. You'll be provided examples between <EXAMPLES> and </EXAMPLES> and the choices between <CHOICES> and </CHOICES>
You may also be provided with context, such as past messages, to help you make your decision. If this context is available it will be between <CONTEXT> and </CONTEXT> Your response must be valid json.

Description of the system: {self.system_description}

<EXAMPLES>
{"\n".join(examples)}
</EXAMPLES>"""

        return self

    def most_common_string(self, string_list):
        return max(set(string_list), key=string_list.count)

    async def gen(self, generation_input, cache):
        async def make_decision(resp):
            print(dict(resp))
            async with cache.with_suffix(self.name):
                votes = await cache.get("votes", [])
                votes.append(resp["decision"].lower())

                await cache.set("votes", votes)

            return dict(resp)

        await self.inference_client.tool_chain(
            messages=generation_input,
            top_p=0.9,
            tool_choice="any",
            temperature=0,
            max_iterations=1,
            tools=[
                (make_decision, MAKE_DECISION_SCHEMA),
            ],
        )

    async def run(self, cache: Cache, semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        self._logger = logging.getLogger(__name__).getChild(
            await cache.get("request_id")
        )

        message = await cache.get(self.input_var)

        choices = []
        for decision in self.flow_config.decisions:
            choice = {
                "choice": decision.next_node,
                "description": decision.metadata.get("description"),
            }
            choices.append(choice)

        prompt = self._prompt

        if self.use_historicals:
            context = await cache.get("formatted_chat_messages", [])
            context_string = "\n".join([ctx.model_dump_json() for ctx in context])
            prompt = (
                self._prompt
                + textwrap.dedent(
                    f"""
                    <CONTEXT>
                    {context_string}
                    </CONTEXT>
                    """
                ).strip()
            )

        prompt = (
            prompt
            + "\n"
            + textwrap.dedent(
                f"""
                <CHOICES>
                {json.dumps(choices, indent=2)}
                </CHOICES>
                """
            ).strip()
        )

        generation_input = [
            ChatMessage(
                role=ChatRole.SYSTEM,
                content=prompt,
            ),
            ChatMessage(
                role=ChatRole.USER,
                content=message,
            ),
        ]

        tasks = []
        for _ in range(0, self.voters):
            tasks.append(self.gen(generation_input, cache))

        await asyncio.gather(*tasks)

        async with cache.with_suffix(self.name):
            votes = await cache.get("votes", [])

        print(votes)
        if votes:
            voted_choice = self.most_common_string(votes)

            for decision in self.flow_config.decisions:
                if decision.next_node.lower() == voted_choice.lower():
                    self._logger.info(f"agent directed flow decision: {decision.next_node}")
                    return {
                        "status": "success",
                        "cleanup": decision.cleanup_on_jump,
                        "decision": decision.next_node,
                        "metadata": decision.metadata,
                    }

        if self.flow_config.default:
            self._logger.info(f"Using default decision: {self.flow_config.default}")
            return {
                "status": "success",
                "cleanup": True,
                "decision": self.flow_config.default,
                "metadata": {},
            }

        # If no decisions were met, fall through
        self._logger.warning("No agent directed flow decision was met, falling through")
        return {
            "status": "success",
            "decision": None,  # Indicates fall-through
            "cleanup": True,
            "metadata": {},
        }
