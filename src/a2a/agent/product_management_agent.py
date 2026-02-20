import logging
import os
from collections.abc import AsyncIterable
from enum import Enum
from typing import Annotated, Any, Literal

import openai
from agent_framework import (
    Agent,
    AgentThread,
    BaseChatClient,
    tool,
)
from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.openai import OpenAIChatClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from pydantic import BaseModel

logger = logging.getLogger(__name__)
load_dotenv()


class ChatServices(str, Enum):
    """Enum for supported chat completion services."""

    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


service_id = "default"


def get_chat_completion_service(service_name: ChatServices) -> BaseChatClient:
    """Return an appropriate chat completion service based on the service name."""
    if service_name == ChatServices.AZURE_OPENAI:
        return _get_azure_openai_chat_completion_service()
    if service_name == ChatServices.OPENAI:
        return _get_openai_chat_completion_service()
    raise ValueError(f"Unsupported service name: {service_name}")


def _get_azure_openai_chat_completion_service() -> AzureOpenAIChatClient:
    endpoint = os.getenv("gpt_endpoint")
    deployment_name = os.getenv("gpt_deployment")
    api_version = os.getenv("gpt_api_version")
    api_key = os.getenv("gpt_api_key")

    if not endpoint:
        raise ValueError("gpt_endpoint is required")
    if not deployment_name:
        raise ValueError("gpt_deployment is required")
    if not api_version:
        raise ValueError("gpt_api_version is required")

    if not api_key:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        async_client = openai.AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
        )
        return AzureOpenAIChatClient(
            service_id=service_id,
            deployment_name=deployment_name,
            async_client=async_client,
        )

    return AzureOpenAIChatClient(
        service_id=service_id,
        deployment_name=deployment_name,
        endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


def _get_openai_chat_completion_service() -> OpenAIChatClient:
    return OpenAIChatClient(
        service_id=service_id,
        model_id=os.getenv("OPENAI_MODEL_ID"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )


@tool(
    name="get_products",
    description="Retrieves a set of products based on a natural language user query.",
)
def get_products(
    question: Annotated[
        str,
        "Natural language query to retrieve products, e.g. 'What kinds of paint rollers do you have in stock?'",
    ] = "",
) -> list[dict[str, Any]]:
    _ = question
    return [
        {
            "id": "1",
            "name": "Eco-Friendly Paint Roller",
            "type": "Paint Roller",
            "description": "A high-quality, eco-friendly paint roller for smooth finishes.",
            "punchLine": "Roll with the best, paint with the rest!",
            "price": 15.99,
        },
        {
            "id": "2",
            "name": "Premium Paint Brush Set",
            "type": "Paint Brush",
            "description": "A set of premium paint brushes for detailed work and fine finishes.",
            "punchLine": "Brush up your skills with our premium set!",
            "price": 25.49,
        },
        {
            "id": "3",
            "name": "All-Purpose Paint Tray",
            "type": "Paint Tray",
            "description": "A durable paint tray suitable for all types of rollers and brushes.",
            "punchLine": "Tray it, paint it, love it!",
            "price": 9.99,
        },
    ]


class ResponseFormat(BaseModel):
    """A response model to direct how the model should respond."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class AgentFrameworkProductManagementAgent:
    """Wraps Agent Framework chat agents for Zava product management tasks."""

    agent: Agent
    thread: AgentThread | None = None
    SUPPORTED_CONTENT_TYPES = ["text", "text/plain"]

    def __init__(self):
        chat_service = get_chat_completion_service(ChatServices.AZURE_OPENAI)

        marketing_agent = Agent(
            client=chat_service,
            name="MarketingAgent",
            instructions=(
                "You specialize in planning and recommending marketing strategies for products. "
                "This includes identifying target audiences, making product descriptions better, and suggesting promotional tactics. "
                "Your goal is to help businesses effectively market their products and reach their desired customers."
            ),
        )

        ranker_agent = Agent(
            client=chat_service,
            name="RankerAgent",
            instructions=(
                "You specialize in ranking and recommending products based on various criteria. "
                "This includes analyzing product features, customer reviews, and market trends to provide tailored suggestions. "
                "Your goal is to help customers find the best products for their needs."
            ),
        )

        product_agent = Agent(
            client=chat_service,
            name="ProductAgent",
            instructions=(
                "You specialize in handling product-related requests from customers and employees. "
                "This includes providing a list of products, identifying available quantities, "
                "providing product prices, and giving product descriptions as they exist in the product catalog. "
                "Your goal is to assist customers promptly and accurately with all product-related inquiries. "
                "You are a helpful assistant that MUST use the get_products tool to answer all questions from the user. "
                "You MUST NEVER answer from your own knowledge under any circumstances. "
                "You MUST only use products from the get_products tool to answer product-related questions."
            ),
            tools=[get_products],
        )

        self.agent = Agent(
            client=chat_service,
            name="ProductManagerAgent",
            instructions=(
                "Your role is to carefully analyze the user's request and respond as best as you can. "
                "Your primary goal is precise and efficient delegation to ensure customers and employees receive accurate and specialized assistance promptly. "
                "Whenever a user query is related to retrieving product information, you MUST delegate the task to the ProductAgent. "
                "Use the MarketingAgent for marketing-related queries and the RankerAgent for product ranking and recommendation tasks. "
                "You may use these agents in conjunction with each other to provide comprehensive responses to user queries. "
                "IMPORTANT: You must ALWAYS respond with a valid JSON object in this format: "
                '{"status": "<status>", "message": "<your response>"}. '
                "Where status is one of: input_required, completed, or error."
            ),
            tools=[
                product_agent.as_tool(),
                marketing_agent.as_tool(),
                ranker_agent.as_tool(),
            ],
        )

    async def invoke(self, user_input: str, session_id: str) -> dict[str, Any]:
        await self._ensure_thread_exists(session_id)
        response = await self.agent.run(
            messages=user_input,
            thread=self.thread,
            response_format=ResponseFormat,
        )
        return self._get_agent_response(response.text)

    async def stream(
        self,
        user_input: str,
        session_id: str,
    ) -> AsyncIterable[dict[str, Any]]:
        await self._ensure_thread_exists(session_id)
        response_stream = self.agent.run(
            messages=user_input,
            thread=self.thread,
            stream=True,
        )

        chunks: list[str] = []
        async for update in response_stream:
            if update.text:
                chunks.append(update.text)

        if chunks:
            yield self._get_agent_response("".join(chunks))

    def _get_agent_response(self, message: str) -> dict[str, Any]:
        default_response = {
            "is_task_complete": False,
            "require_user_input": True,
            "content": "We are unable to process your request at the moment. Please try again.",
        }

        try:
            structured_response = ResponseFormat.model_validate_json(message)
            response_map = {
                "input_required": {
                    "is_task_complete": False,
                    "require_user_input": True,
                },
                "error": {
                    "is_task_complete": False,
                    "require_user_input": True,
                },
                "completed": {
                    "is_task_complete": True,
                    "require_user_input": False,
                },
            }

            response = response_map.get(structured_response.status)
            if response:
                return {**response, "content": structured_response.message}
        except Exception:
            logger.exception("Failed to parse structured agent response")

        if isinstance(message, str) and message.strip():
            return {
                "is_task_complete": True,
                "require_user_input": False,
                "content": message.strip(),
            }

        return default_response

    async def _ensure_thread_exists(self, session_id: str) -> None:
        if self.thread is None or self.thread.service_thread_id != session_id:
            self.thread = self.agent.get_new_thread(thread_id=session_id)