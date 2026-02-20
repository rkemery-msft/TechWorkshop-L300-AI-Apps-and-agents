import asyncio
import logging
import uuid
from typing import Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.product_management_agent import AgentFrameworkProductManagementAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

product_management_agent = AgentFrameworkProductManagementAgent()
active_sessions: Dict[str, str] = {}


class ChatMessage(BaseModel):
    """Chat message model."""

    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    """Chat response model."""

    response: str
    session_id: str
    is_complete: bool
    requires_input: bool


def _is_rate_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return "429" in message or "ratelimit" in message or "rate limit" in message


async def _invoke_agent_with_timeout(message: str, session_id: str, timeout_seconds: int = 15):
    def _run_in_thread():
        return asyncio.run(product_management_agent.invoke(message, session_id))

    return await asyncio.wait_for(
        asyncio.to_thread(_run_in_thread),
        timeout=timeout_seconds,
    )


@router.post("/message", response_model=ChatResponse)
async def send_message(chat_message: ChatMessage):
    """Send a message to the product management agent and get a response."""
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        active_sessions[session_id] = session_id

        response = await _invoke_agent_with_timeout(chat_message.message, session_id)

        return ChatResponse(
            response=response.get("content", "No response available"),
            session_id=session_id,
            is_complete=response.get("is_task_complete", False),
            requires_input=response.get("require_user_input", True),
        )
    except asyncio.TimeoutError:
        session_id = chat_message.session_id or str(uuid.uuid4())
        return ChatResponse(
            response=(
                "The agent is taking longer than expected right now. "
                "Please try your message again in a few seconds."
            ),
            session_id=session_id,
            is_complete=False,
            requires_input=True,
        )
    except Exception as error:
        if _is_rate_limit_error(error):
            logger.warning("Rate limit reached while processing chat message")
            session_id = chat_message.session_id or str(uuid.uuid4())
            return ChatResponse(
                response=(
                    "I'm getting a high volume of requests right now. "
                    "Please wait a few seconds and try again."
                ),
                session_id=session_id,
                is_complete=False,
                requires_input=True,
            )
        logger.error("Error processing chat message: %s", error)
        session_id = chat_message.session_id or str(uuid.uuid4())
        return ChatResponse(
            response="Something went wrong while contacting the agent. Please try again.",
            session_id=session_id,
            is_complete=False,
            requires_input=True,
        )


@router.post("/stream")
async def stream_message(chat_message: ChatMessage):
    """Stream a response from the product management agent."""
    try:
        session_id = chat_message.session_id or str(uuid.uuid4())
        active_sessions[session_id] = session_id

        async def generate_response():
            try:
                async for partial in product_management_agent.stream(
                    chat_message.message,
                    session_id,
                ):
                    content = partial.get("content", "")
                    is_complete = partial.get("is_task_complete", False)
                    requires_input = partial.get("require_user_input", False)

                    response_data = {
                        "content": content,
                        "session_id": session_id,
                        "is_complete": is_complete,
                        "requires_input": requires_input,
                    }

                    yield f"data: {response_data}\n\n"

                    if is_complete:
                        break
            except Exception as error:
                if _is_rate_limit_error(error):
                    yield "data: {'error': 'rate_limit', 'message': 'High request volume. Please try again in a few seconds.'}\n\n"
                    return
                logger.error("Error in streaming response: %s", error)
                yield f"data: {{'error': '{str(error)}'}}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            },
        )
    except Exception as error:
        logger.error("Error setting up streaming: %s", error)
        raise HTTPException(status_code=500, detail=str(error)) from error


@router.get("/sessions")
async def get_active_sessions():
    """Get list of active chat sessions."""
    return {"active_sessions": list(active_sessions.keys())}


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific chat session."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    raise HTTPException(status_code=404, detail="Session not found")