from uuid import uuid4, UUID
from pydantic import BaseModel, Field

class BaseBook(BaseModel):
    name: str = Field(examples=["Война и мир"])
    author: str = Field(examples=["Л. Н. Толстой"])
    year: int = Field(examples=[1868])
    annotation: str = Field(examples=["Классический роман-эпопея рассказывает о сложном, бурном периоде в истории России и всей Европы с 1805 по 1812 год. "])

class Book (BaseBook):
    id: UUID = Field(examples=[uuid4()])
    ...

class CreateBook(BaseBook):
    ...

class UpdateBook(BaseBook):
    ...

class UserInDB(BaseModel):
    login: str
    is_admin: bool
    hashed_password: str

class LoginRequest(BaseModel):
    login: str
    password: str

class LoginResponse(BaseModel):
    token: str
class SuccessMessage(BaseModel):
    message: str = Field(examples=["Операция успешно выполнена"])
