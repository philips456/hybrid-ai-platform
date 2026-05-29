"""
src/api/routers/auth.py
Authentication endpoints.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from src.api.auth.jwt import create_access_token, Token

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Demo users — replaced by DB in production
DEMO_USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "analyst": {"password": "analyst123", "role": "analyst"},
    "viewer": {"password": "viewer123", "role": "viewer"},
}


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/token", response_model=Token)
async def login(request: LoginRequest):
    """
    Authenticates user and returns JWT token.

    Demo credentials:
    - admin / admin123 (full access)
    - analyst / analyst123 (can validate HITL)
    - viewer / viewer123 (read only)
    """
    user = DEMO_USERS.get(request.username)
    if not user or user["password"] != request.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = create_access_token(
        data={"sub": request.username, "role": user["role"]}
    )
    return Token(access_token=token)
