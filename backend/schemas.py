"""Pydantic request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, StringConstraints


class AuthRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=6, max_length=128)


class RegisterRequest(AuthRequest):
    username: str = Field(min_length=1, max_length=80)


class GenerateRequest(BaseModel):
    theme: str = Field(min_length=1, max_length=120)
    category: str = Field(default="想象故事", max_length=80)
    character: str = Field(default="小主角", max_length=80)
    length: str = Field(default="medium", pattern="^(short|medium|long)$")
    extra: str = Field(default="", max_length=500)
    generate_images: bool = False
    image_count: int = Field(default=3, ge=1, le=4)


class ImagesRequest(BaseModel):
    story_id: int = Field(gt=0)
    count: int = Field(default=3, ge=1, le=8)


class VoiceRequest(BaseModel):
    story_id: int = Field(gt=0)
    voice_id: str = Field(default="browser", max_length=40)


class SessionCreateRequest(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True, max_length=120)] = "新会话"


class AssistantChatRequest(BaseModel):
    message: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    email: EmailStr
