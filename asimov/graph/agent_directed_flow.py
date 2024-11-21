import textwrap
import logging
from typing import List, Dict, Any
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

from asimov.constants import ModelFamily
from asimov.utils.models import is_model_family, prepare_model_generation_input


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


class AgentDrivenFlowControlConfig(FlowControlConfig):
    decisions: List[AgentDrivenFlowDecision]


class AgentDirectedFlowControl(FlowControlModule):
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

    def most_common_string(self, string_list):
        return max(set(string_list), key=string_list.count)

    async def gen_loop(self, x, generation_input, generation_pretext):
        """A function to handle regeneration for bad json, all because the LLM would randomly not generate correct JSON and simply regenerating did not get there."""
        await asyncio.sleep(x * 0.1)
        while True:
            try:
                generation = await self.inference_client.get_generation(
                    generation_input,
                )
                if is_model_family(self.inference_client, [ModelFamily.ANTHROPIC]):
                    generation = generation_pretext + generation

                generation = generation[: generation.rfind("}") + 1]
                self._logger.info(f"DECISION GENERATION: {generation}")
                decision_choice = json.loads(generation)
                decision_choice = decision_choice["choice"].strip()

                return decision_choice
            except json.JSONDecodeError as e:
                print(generation)
                if is_model_family(self.inference_client, [ModelFamily.ANTHROPIC]):
                    generation_input = generation_input[:-1]

                generation_input.append(
                    ChatMessage(role=ChatRole.ASSISTANT, content=generation)
                )
                generation_input.append(
                    ChatMessage(
                        role=ChatRole.USER,
                        content="Sorry that was not valid JSON try again. Please only generate JSON. Remember you need the escape sequence newline characters in json and not linebreaks.",
                    )
                )

                if generation_pretext:
                    generation_input.append(
                        ChatMessage(
                            role=ChatRole.ASSISTANT,
                            content=generation_pretext,
                            model_families=[ModelFamily.ANTHROPIC],
                        )
                    )

                generation_input = prepare_model_generation_input(
                    generation_input, self.inference_client
                )

                print(generation_input)

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

        generation_pretext = None

        if is_model_family(self.inference_client, [ModelFamily.ANTHROPIC]):
            generation_pretext = '{"choice": "'
            generation_input.append(
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content=generation_pretext,
                ),
            )

        votes = []

        # Occasionally the llm messes up the JSON so just retry.
        while True:
            try:
                tasks = []
                for x in range(0, self.voters):
                    tasks.append(self.gen_loop(x, generation_input, generation_pretext))

                votes = await asyncio.gather(*tasks)

                break
            except json.JSONDecodeError as e:
                self._logger.warning(f"JSON DECODE ERROR: {e}")
                pass

        votes = list(map(lambda x: x.lower(), votes))

        print(votes)
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

        # If no decisions were met, fall through
        self._logger.warning("No agent directed flow decision was met, falling through")
        return {
            "status": "success",
            "decision": None,  # Indicates fall-through
            "cleanup": True,
            "metadata": {},
        }
