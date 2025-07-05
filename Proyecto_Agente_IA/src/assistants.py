# src/assistants.py
import os
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from .prompts import sales_rep_prompt, support_prompt
from .state import State
from .tools import (
    DEFAULT_USER_ID,
    EscalateToHuman,
    RouteToCustomerSupport,
    cart_tool,
    search_tool,
    set_thread_id,
    set_user_id,
    structured_search_tool,
    view_cart,
)

load_dotenv()
import pandas as pd

# Setup LLM
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or "Empty"
llm = ChatOpenAI(api_key=OPENAI_API_KEY)

# Tool registration
sales_tools = [
    RouteToCustomerSupport,
    search_tool,
    structured_search_tool,
    cart_tool,
    view_cart,
]
support_tools = [EscalateToHuman]

# Runnable pipelines
sales_runnable = sales_rep_prompt.partial(time=datetime.now) | llm.bind_tools(
    sales_tools
)
support_runnable = support_prompt.partial(time=datetime.now) | llm.bind_tools(
    support_tools
)


# TODO
def sales_assistant(
    state: State, config: RunnableConfig, runnable=sales_runnable
) -> dict:
    """
    LangGraph node function for running the sales assistant LLM agent.

    This function binds a chat prompt (`sales_rep_prompt`) with tools and invokes
    the LangChain Runnable pipeline. It sets the thread and user IDs and runs the
    agent with the given state and config.

    ---
    Arguments:
    - state (State): LangGraph state with current dialog history.
    - config (RunnableConfig): Config object that contains the `thread_id`.
    - runnable: (optional) The runnable to use; defaults to global `sales_runnable`.

    ---
    Behavior:
    - Extract thread ID from config and set it using `set_thread_id(...)`.
    - Set default user ID via `set_user_id(...)`.
    - Use the given `runnable` to run the assistant logic.

    ---
    Returns:
    - A dictionary with a `"messages"` key containing the new AI message(s).
    Example: `{"messages": [AIMessage(...)]}`
    """
    """
    Runs the Sales Representative agent inside a LangGraph node.

    Workflow
    --------
    1. Grabs the **`thread_id`** from *config* and registers it via
   `set_thread_id(...)` so that all tools (cart, etc.) reference the
   correct shopping cart.
   2. Sets a safe **`user_id`** (the first user in the dataset) via
   `set_user_id(...)`; some tools rely on this to filter purchase
   history.
   3. Invokes the `sales_runnable` pipeline (prompt + tools + LLM) with the
   given *state* and *config*.
    4. Wraps the LLM’s response in a dictionary  
   `{"messages": [...]}` — the structure LangGraph expects.

    The *runnable* parameter can be overridden in tests to inject a mock and
    speed up execution.
   """
    # Manage IDs to make the tools work
    thread_id = config.get("configurable", {}).get("thread_id")
    if thread_id:
        set_thread_id(thread_id)

    # For these examples we always use the default user
    set_user_id(DEFAULT_USER_ID)

    # Run the LLM-tools pipeline
    ai_messages = runnable.invoke(state, config=config)

    # Normalize return
    #    (If invoke already returns a list, we pack it as is.)
    return {"messages": ai_messages}


def support_assistant(state: State, config: RunnableConfig) -> dict:
    set_thread_id(config["configurable"]["thread_id"])
    return {"messages": support_runnable.invoke(state, config=config)}
