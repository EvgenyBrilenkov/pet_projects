import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables.history import RunnableWithMessageHistory, BaseChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from chainlit.input_widget import TextInput, Select, Slider
from typing import Optional, Dict, Optional, List
from chainlit import PersistedUser, User
from chainlit.data import BaseDataLayer
from chainlit.element import ElementDict
from chainlit.step import StepDict
from chainlit.types import Feedback, ThreadDict, Pagination, ThreadFilter, PaginatedResponse, PageInfo
import json
from datetime import datetime


store = {}

def get_history_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


users = [
    cl.User(identifier="1", display_name="Admin", metadata={"username": "admin", "password": "admin"}),
    cl.User(identifier="2", display_name="Nick", metadata={"username": "nick", "password": "super"}),
    cl.User(identifier="3", display_name="Dan", metadata={"username": "dan", "password": "ultra"}),
]

@cl.password_auth_callback
def on_login(username: str, password: str) -> Optional[cl.User]:
    for user in users:
        current_username, current_password = user.metadata["username"], user.metadata["password"]
        if current_username == username and current_password == password:
            return user
    return None

@cl.set_chat_profiles
async def chat_profile():
    return [
        cl.ChatProfile(
            name="LangChain Helper",
            icon="https://static.wikitide.net/italianbrainrotwiki/5/50/Chimpanzini_Bananino.png",
            markdown_description="Please confirm the settings, even if you have left some fields empty.",
            starters=[
                cl.Starter(
                    label="What is a LangChain?",
                    message="What is a LangChain?",
                    icon="https://upload.wikimedia.org/wikipedia/commons/b/be/Twisted_link_chain.jpg",
                ),
                cl.Starter(
                    label="Why is the parrot a mascot?",
                    message="Why is the parrot a mascot?",
                    icon="https://cdn.britannica.com/35/3635-050-96241EC1/Scarlet-macaw-ara-macao.jpg",
                ),
            ],
        )
    ]

@cl.on_chat_start
async def start():
    settings = cl.ChatSettings(
        [
            Select(
                id="model",
                label="Model",
                values=["open-mistral-nemo", "mistral-small"],
                initial_index=0,
            ),
            TextInput(
                id="token",
                label="API Token",
                initial="",
                placeholder="Type token here",
                multiline=False
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=1,
                min=0,
                max=2,
                step=0.1,
            ),
            TextInput(
                id="domain",
                label="Expert in",
                initial="Biology",
                placeholder="Type domain here",
                multiline=False
            ),
            TextInput(
                id="name",
                label="Your name",
                initial="",
                placeholder="Type your name here",
                multiline=False
            ),
        ]
    )
    await settings.send()

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("model", settings["model"])
    cl.user_session.set("token", settings["token"])
    cl.user_session.set("temperature", settings["temperature"])
    cl.user_session.set("domain", settings["domain"])
    cl.user_session.set("name", settings["name"])
    if cl.user_session.get("name", settings["name"]) is None:
        cl.user_session.set("name", "stranger")
    if cl.user_session.get("token", settings["token"]) is None:
        cl.user_session.set("token", '8TPqMUmAcGPrGMqkHKvBbJXzCMRIRvMn')

@cl.on_message
async def handle_message(message: cl.Message):
    model = cl.user_session.get("model")
    token = cl.user_session.get("token")
    temperature = cl.user_session.get("temperature")
    domain = cl.user_session.get("domain")
    name = cl.user_session.get("name")
    messages = [
        ("system", "You are an expert in {domain}. Your task is answer the question. Do not answer too long"),
        ("human", "{question}"),
    ]
    prompt = ChatPromptTemplate(messages)
    llm = ChatMistralAI(
        model=model,
        temperature=temperature,
        mistral_api_key=token,
    )
    final_chain = RunnableWithMessageHistory(
    prompt | llm, get_history_by_session_id,
    input_messages_key="question", history_messages_key="history") | StrOutputParser()
    user_question = message.content
    user_session_id = cl.user_session.get("id")
    thanks_action = cl.Action(
        label="❤",
        name="thanks_action",
        payload={"user_session_id": user_session_id},
        tooltip="Send thanks for the helpful reply"
    )
    msg = cl.Message(content=f"Thank you for your request, {name}!\n", actions=[thanks_action])
    async for chunk in final_chain.astream(
        {"domain": domain, "question": user_question},
        config=RunnableConfig(configurable={"session_id": user_session_id})
    ):
        await msg.stream_token(chunk)
    await msg.send()

@cl.action_callback("thanks_action")
async def on_action(action: cl.Action):
    print("message id:", action.forId, "action payload:", action.payload)
    await action.remove()
    await cl.Message(content="Thank you too!").send()

@cl.data_layer
def get_data_layer():
    return CustomDataLayer()

class CustomDataLayer(BaseDataLayer):
    async def build_debug_url(self) -> str:
        return ""

    def __init__(self):
        self.users: Dict[str, PersistedUser] = {}
        self.threads: Dict[str, ThreadDict] = {}
        self.elements: Dict[str, ElementDict] = {}
        self.steps: Dict[str, StepDict] = {}
        self.feedback: Dict[str, Feedback] = {}

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        return self.users.get(identifier)

    async def create_user(self, user: User) -> PersistedUser:
        persisted_user = PersistedUser(**user.__dict__, id=user.identifier, createdAt=datetime.now().date().strftime("%Y-%m-%d"))
        self.users[user.identifier] = persisted_user
        return persisted_user

    async def upsert_feedback(self, feedback: Feedback) -> str:
        self.feedback[feedback.id] = feedback
        return feedback.id

    async def delete_feedback(self, feedback_id: str) -> bool:
        if feedback_id in self.feedback:
            del self.feedback[feedback_id]
            return True
        return False

    async def create_element(self, element_dict: ElementDict) -> None:
        self.elements[element_dict["id"]] = element_dict

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        return self.elements.get(element_id)

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None) -> None:
        if element_id in self.elements:
            del self.elements[element_id]

    async def create_step(self, step_dict: StepDict) -> None:
        self.steps[step_dict["id"]] = step_dict

    async def update_step(self, step_dict: StepDict) -> None:
        self.steps[step_dict["id"]] = step_dict

    async def delete_step(self, step_id: str) -> None:
        if step_id in self.steps:
            del self.steps[step_id]

    async def get_thread_author(self, thread_id: str) -> str:
        if thread_id in self.threads:
            author = self.threads[thread_id]["userId"]
        else:
            author = "Unknown"
        return author

    async def delete_thread(self, thread_id: str) -> None:
        if thread_id in self.threads:
            del self.threads[thread_id]

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        if not filters.userId:
            raise ValueError("userId is required")

        threads = [t for t in list(self.threads.values()) if t["userId"] == filters.userId]
        start = 0
        if pagination.cursor:
            for i, thread in enumerate(threads):
                if thread["id"] == pagination.cursor:
                    start = i + 1
                    break
        end = start + pagination.first
        paginated_threads = threads[start:end] or []

        has_next_page = len(threads) > end
        start_cursor = paginated_threads[0]["id"] if paginated_threads else None
        end_cursor = paginated_threads[-1]["id"] if paginated_threads else None

        result = PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next_page,
                startCursor=start_cursor,
                endCursor=end_cursor,
            ),
            data=paginated_threads,
        )
        return result

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        thread = self.threads.get(thread_id)
        thread["steps"] = [st for st in self.steps.values() if st["threadId"] == thread_id]
        thread["elements"] = [el for el in self.elements.values() if el["threadId"] == thread_id]
        return thread

    async def update_thread(self, thread_id: str, name: Optional[str] = None, user_id: Optional[str] = None, metadata: Optional[Dict] = None, tags: Optional[List[str]] = None):
        if thread_id in self.threads:
            if name:
                self.threads[thread_id]["name"] = name
            if user_id:
                self.threads[thread_id]["userId"] = user_id
            if metadata:
                self.threads[thread_id]["metadata"] = metadata
            if tags:
                self.threads[thread_id]["tags"] = tags
        else:
            data = {
                "id": thread_id,
                "createdAt": (
                    datetime.now().isoformat() + "Z" if metadata is None else None
                ),
                "name": (
                    name
                    if name is not None
                    else (metadata.get("name") if metadata and "name" in metadata else None)
                ),
                "userId": user_id,
                "userIdentifier": user_id,
                "tags": tags,
                "metadata": json.dumps(metadata) if metadata else None,
            }
            self.threads[thread_id] = data
