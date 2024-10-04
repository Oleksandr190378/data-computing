from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import json
from dotenv import load_dotenv
import re


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API is not found")


template = """
You are a helpful assistant. Help the user with their requests.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000, memory_key="history", return_messages=True)
store = {}


def get_session_history(session_id: str):

    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


session_id = "bcd"
chain = prompt_template | llm | memory
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

config = {"configurable": {"session_id": session_id}}


def chat_with_history(query):
    try:
        response = chain_with_history.invoke({"query": query}, config=config)
        content = response.content if hasattr(response, 'content') else str(response)
        history = get_session_history(session_id).messages
        history_text = "\n".join([
          f"{msg.type}: {msg.content}" if hasattr(msg, 'type') else f"{msg.role}: {msg.content}"
          for msg in history
        ])
        return content, history_text
    except Exception as e:
        return f"Error processing query: {e}", ""


def chat_with_history_context(query, context=None):
    try:
        if context:
            query = f"{query}\nContext: {context}"
        response = chain_with_history.invoke({"query": query}, config=config)
        content = response.content if hasattr(response, 'content') else str(response)
        history = get_session_history(session_id).messages
        history_text = "\n".join([
            f"{msg.type}: {msg.content}" if hasattr(msg, 'type') else f"{msg.role}: {msg.content}"
            for msg in history
        ])
        history_text = re.sub(r'(Context:.*?)ai:', r'ai:', history_text, flags=re.DOTALL)
        return content, history_text
    except Exception as e:
        return f"Error processing query: {e}", ""


def save_history_to_file(history_text):
    with open("app/chat_history.json", "w") as f:
        json.dump({"history": history_text}, f)


def load_history_from_file():
    if os.path.exists("app/chat_history.json"):
        with open("app/chat_history.json", "r") as f:
            data = json.load(f)
            return data.get("history", "")
    return ""


def delete_history_file():
    global store
    if os.path.exists("app/chat_history.json"):
        os.remove("app/chat_history.json")
    store = {}

