import os
from fastapi import (
    FastAPI,
    Path,
    Query,
    Body,
    Cookie,
    Depends,
    Header,
    Request,
    Response,
    Form,
    File,
    UploadFile,
    HTTPException,
)
from enum import Enum
from pydantic import BaseModel, Required, Field, HttpUrl
from typing import Annotated, Union, Any, Optional
from fastapi.responses import JSONResponse, RedirectResponse
from conversation import Question, Conversation
from errors import PolicyViolationError

# OpenAI Configuration
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # Read the local .env file
openai.api_key = os.environ["OPENAI_API_KEY"]
# OpenAI Configuration

app = FastAPI()


class ResponseContent(BaseModel):
    code: int = 0
    message: Any = None


async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != os.environ["ACCESS_TOKEN"]:
        raise HTTPException(status_code=401, detail="X-Token header invalid")


@app.exception_handler(PolicyViolationError)
async def policy_violation_error_handler(request: Request, exc: PolicyViolationError):
    return JSONResponse(
        status_code=200,
        content=ResponseContent(code=exc.code, message=exc.message).dict(),
    )


@app.post("/chatbot", dependencies=[Depends(verify_token)])
async def chat(question: Question):
    return ResponseContent(
        message=await Conversation.chat_with_moderation(question=question)
    )
