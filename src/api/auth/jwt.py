"""
src/api/auth/jwt.py
JWT authentication for FastAPI.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from pydantic import BaseModel

from configs.settings import settings

security = HTTPBearer()

SECRET_KEY = settings.api_secret_key
ALGORITHM = settings.api_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.api_access_token_expire_minutes


class TokenData(BaseModel):
    username: str
    role: str = "viewer"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role", "viewer")
        if username is None:
            raise credentials_exception
        return TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception


def require_role(required_role: str):
    """Role-based access control dependency."""
    role_hierarchy = {"viewer": 0, "analyst": 1, "admin": 2}

    def check_role(token_data: TokenData = Depends(verify_token)) -> TokenData:
        user_level = role_hierarchy.get(token_data.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return token_data

    return check_role
