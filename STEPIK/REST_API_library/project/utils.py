import aiofiles
import json
import os
from uuid import UUID
from .schemas import Book, UserInDB, LoginRequest, LoginResponse
import logging
from passlib.context import CryptContext
from typing import Optional
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import Depends, HTTPException, Body
import secrets

logging.getLogger('passlib').setLevel(logging.ERROR)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

users = [
    UserInDB(
        login="admin",
        is_admin=True,
        hashed_password=pwd_context.hash("adminpass"),
    ),
    UserInDB(
        login="user",
        is_admin=False,
        hashed_password=pwd_context.hash("userpass"),
    ),
]

DATA_FILE = 'books.json'

sessions = {}

async def load_books() -> list[Book]:
    books = []
    if not os.path.exists(DATA_FILE):
        return books
    async with aiofiles.open(DATA_FILE, 'r', encoding='utf-8') as f:
        content = await f.read()
        raw_list = json.loads(content)
    for raw_data in raw_list:
        book_id = UUID(raw_data['id'])
        updated_raw_data = {**raw_data, "id": book_id}
        books.append(Book(**updated_raw_data))
    return books

async def save_books(books: list[Book]):
    data = [{**book.model_dump(exclude={'id'}), 'id': str(book.id)} for book in books]
    async with aiofiles.open(DATA_FILE, 'w', encoding='utf-8') as f:
        content = json.dumps(data, indent=4)
        await f.write(content)

async def get_user_by_login(login: str) -> Optional[UserInDB]:
    for user in users:
        if user.login == login:
            return user
    return None

async def get_user_by_token(token: str) -> Optional[UserInDB]:
    user_login = sessions.get(token)
    if not user_login:
        return None
    return await get_user_by_login(user_login)


async def handle_login(data: LoginRequest = Body()):
    user = await get_user_by_login(data.login)
    if not user or not pwd_context.verify(data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail = "Incorrect login or password")
    token = secrets.token_hex(32)
    sessions[token] = user.login
    return LoginResponse(token=token)

async def authenticate_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))):
    if not credentials:
        raise HTTPException(status_code=401, detail="No token")
    token = credentials.credentials
    user = await get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Not enough permissions")
