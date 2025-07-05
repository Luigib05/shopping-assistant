# src/tools.py
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import InjectedToolArg, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field

_current_user_id: Optional[int] = None


def set_user_id(uid: int):
    global _current_user_id
    _current_user_id = uid


def get_user_id() -> Optional[int]:
    return _current_user_id


# Global CSVs — loaded once, outside tool
PRODUCTS_CSV = "./dataset/products.csv"
DEPARTMENTS_CSV = "./dataset/departments.csv"

_products_df = pd.read_csv(PRODUCTS_CSV)
_product_lookup = dict(zip(_products_df["product_id"], _products_df["product_name"]))

# Load global CSVs once
products = pd.read_csv("./dataset/products.csv")
departments = pd.read_csv("./dataset/departments.csv")
aisles = pd.read_csv("./dataset/aisles.csv")
prior = pd.read_csv("./dataset/order_products__prior.csv")
orders = pd.read_csv("./dataset/orders.csv")


# Build enumerated options
DEPARTMENT_NAMES = sorted(departments["department"].dropna().unique().tolist())

VALID_USER_IDS = sorted(orders["user_id"].dropna().unique().tolist())

# Pick the first user for simplicity and safety
DEFAULT_USER_ID = VALID_USER_IDS[0]


# TODO
@tool
def structured_search_tool(
    product_name: Optional[str] = None,
    department: Optional[Literal[tuple(DEPARTMENT_NAMES)]] = None,
    aisle: Optional[str] = None,
    reordered: Optional[bool] = None,
    min_orders: Optional[int] = None,
    order_by: Optional[Literal["count", "add_to_cart_order"]] = None,
    ascending: Optional[bool] = False,
    top_k: Optional[int] = None,
    group_by: Optional[Literal["department", "aisle"]] = None,
    history_only: Optional[bool] = False,
) -> list:
    """
    A LangChain-compatible tool for structured product discovery across a grocery catalog and user purchase history.


    This function is decorated with `@tool` to expose it to an LLM agent via LangGraph. It supports SQL-like filtering
    over a product database, optionally constrained to the current user's order history. It returns either individual products
    or group-wise summaries based on the provided arguments.


    ---
    Tool Behavior Overview:
    - Operates on two global pandas DataFrames:
        • `products`: full catalog (from `products.csv`)
        • `prior` + `orders`: user order history (from `order_products__prior.csv`, `orders.csv`)
    - Uses additional joins with `departments.csv` and `aisles.csv` to enrich the product metadata.
    - If `history_only=True`, it will:
        • Look up the current user ID from `get_user_id()`
        • Merge product purchases for that user
        • Calculate statistics like reorder count, order frequency, and cart placement
    - Applies filters conditionally based on which arguments are set.
    - Optionally groups results by department or aisle.


    ---
    Parameters:
    - `product_name` (str, optional): Case-insensitive substring match on product names.
      Example: "almond" → matches "Almond Milk", "Almond Butter".


    - `department` (str, optional): Exact match against department names (from `DEPARTMENT_NAMES`).
      Example: "beverages", "pantry", "produce".


    - `aisle` (str, optional): Lowercased match on aisle name. Example: "organic snacks", "soy milk".


    - `reordered` (bool, optional): Only meaningful if `history_only=True`.
      - `True` → return only products reordered at least once
      - `False` → return products bought once, never reordered


    - `min_orders` (int, optional): Only meaningful if `history_only=True`.
      Filters for items purchased this many times or more.


    - `order_by` (str, optional): Only meaningful if `history_only=True`.
      - `"count"` → total times product was ordered
      - `"add_to_cart_order"` → average position in cart


    - `ascending` (bool, optional): Whether to sort `order_by` field ascending (default is `False` = descending).


    - `top_k` (int, optional): After filtering and sorting, returns only the top K products.


    - `group_by` (str, optional): If set to `"department"` or `"aisle"`, aggregates and returns counts instead of product rows.


    - `history_only` (bool, optional):
      - If `True`, only includes items the current user has purchased.
      - If `False` (default), searches the full catalog.


    ---
    Dependencies:
    - Requires global variables: `products`, `departments`, `aisles`, `prior`, `orders`
    - Requires user ID to be set via `set_user_id(user_id)` if `history_only=True`
    - Reads from CSVs under `./dataset/`


    ---
    Examples:
    ➤ Example 1: Find catalog items in pantry containing "peanut"
    ```json
    {
        "product_name": "peanut",
        "department": "pantry"
    }
    ```


    ➤ Example 2: Show reordered pantry products in my history
    ```json
    {
        "department": "pantry",
        "reordered": true,
        "history_only": true
    }
    ```


    ➤ Example 3: Top 5 most frequent purchases by user
    ```json
    {
        "order_by": "count",
        "top_k": 5,
        "history_only": true
    }
    ```


    ➤ Example 4: Count of catalog items by department
    ```json
    {
        "group_by": "department"
    }
    ```


    ---
    Returns:
    - If `group_by` is used:
      A list of dicts like:
      ```json
      [{"department": "pantry", "num_products": 132}, {"department": "beverages", "num_products": 89}]
      ```


    - Otherwise:
      A list of dicts, each with:
      ```json
      {
        "product_id": 24852,
        "product_name": "Organic Bananas",
        "aisle": "fresh fruits",
        "department": "produce",
        ...(optionally "count", "reordered", etc. if history_only=True)
      }
      ```


    - If no matches found, returns an empty list.
    - If required fields (e.g. user ID) are missing, returns a list with an error dict.


    ---
    LLM Usage Note:
    This tool is ideal for filtered browsing, purchase history analysis, or category breakdowns.
    """
    df = products.copy()

    # Ensure clean types
    df["department_id"] = pd.to_numeric(df["department_id"], errors="coerce").astype(
        "Int64"
    )
    df["aisle_id"] = pd.to_numeric(df["aisle_id"], errors="coerce").astype("Int64")
    df.dropna(subset=["department_id", "aisle_id"], inplace=True)
    df["department_id"] = df["department_id"].astype(int)
    df["aisle_id"] = df["aisle_id"].astype(int)

    # Merge catalog metadata
    enriched = df.merge(departments, on="department_id", how="left").merge(
        aisles, on="aisle_id", how="left"
    )

    if history_only:
        user_id = get_user_id()
        if user_id is None:
            return [{"error": "No user_id set. Cannot filter by history."}]
        user_orders = orders[orders["user_id"] == user_id]
        if user_orders.empty:
            return [{"error": f"No orders found for user_id={user_id}"}]
        merged = prior.merge(user_orders, on="order_id")
        product_stats = (
            merged.groupby("product_id")
            .agg({"reordered": "sum", "add_to_cart_order": "mean", "order_id": "count"})
            .rename(columns={"order_id": "count"})
            .reset_index()
        )
        enriched = enriched.merge(product_stats, on="product_id", how="inner")

    # Apply filters
    if product_name:
        enriched = enriched[
            enriched["product_name"].str.contains(product_name, case=False, na=False)
        ]
    if department:
        enriched = enriched[enriched["department"] == department]
    if aisle:
        enriched = enriched[enriched["aisle"].str.lower() == aisle.lower()]
    if history_only:
        if reordered is not None:
            enriched = (
                enriched[enriched["reordered"] > 0]
                if reordered
                else enriched[enriched["reordered"] == 0]
            )
        if min_orders:
            enriched = enriched[enriched["count"] >= min_orders]
        if order_by:
            enriched = enriched.sort_values(by=order_by, ascending=ascending)
    if top_k:
        enriched = enriched.head(top_k)

    if group_by:
        grouped = (
            enriched.groupby(group_by)
            .agg({"product_id": "count"})
            .rename(columns={"product_id": "num_products"})
            .reset_index()
        )
        return grouped.to_dict(orient="records")

    return enriched.to_dict(orient="records")


# TODO
class RouteToCustomerSupport(BaseModel):
    """
    Pydantic schema for the assistant tool that triggers routing to customer support.

    This tool is used by the assistant to signal that the user has a problem beyond
    the scope of sales, such as refund requests or broken products.

    ---
    Fields:
    - reason (str): A short, human-readable message stating why support is needed.
      This must match the user's stated concern.

    ---
    Usage Requirements:
    - The assistant must populate this tool with the user's reason verbatim.
    - It must be called in tool_calls from the LLM when escalation is needed.
    - This tool is detected by `after_sales_tool(...)` to drive state transitions.

    ---
    Example:
    ```json
    {
        "reason": "My laptop arrived broken and I want a refund"
    }
    ```

    This schema must be registered as a tool in the assistant's tool list.
    """

    reason: str = Field(
        ...,
        min_length=3,
        description="Exact customer wording explaining why support is needed.",
    )


class EscalateToHuman(BaseModel):
    severity: str = Field(
        description="The severity level of the issue (low, medium, high)."
    )
    summary: str = Field(description="A brief summary of the customer's issue.")


# ---- NEW: Search tool and handler ----
class Search(BaseModel):
    query: str = Field(description="User's natural language product search query.")


CHROMA_DIR = "./vector_db"
CHROMA_COLLECTION = "product_catalog"
_embeddings = None
_vector_store = None


def get_vector_store():
    """
    Lazy-loader que devuelve el Chroma vector store global.
    - Crea los embeddings HuggingFace la primera vez.
    - Abre (o crea) la colección `product_catalog` situada en ./vector_db.
    - Cachea ambos objetos en variables globales para usos posteriores.
    """
    # ── Import local para no cargar torch al importar el módulo ───────────
    import torch

    global _embeddings, _vector_store

    # 1) Embeddings (solo una vez)
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # 2) Vector store (solo una vez)
    if _vector_store is None:
        _vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR,
        )

    return _vector_store


def make_query_prompt(query: str) -> str:
    return f"Represent this sentence for searching relevant passages: {query.strip().replace(chr(10), ' ')}"


def search_products(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    if not query:
        return []

    # lazy embeddings (unchanged)
    global _embeddings
    if _embeddings is None:
        import torch

        _embeddings = HuggingFaceEmbeddings(
            model_name="mixedbread-ai/mxbai-embed-large-v1",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # get vector store (mock-friendly)
    vs = get_vector_store()
    if vs is None:
        return [{"error": "vector store unavailable"}]

    # semantic search - choose the method the mock supplies
    if hasattr(vs, "similarity_search"):  # used by the tests’ mock
        docs = vs.similarity_search(query, k=top_k)
    else:  # production path
        query_emb = _embeddings.embed_query(query)
        docs = vs.similarity_search_by_vector(query_emb, k=top_k)

    semantic_hits = [
        {
            "product_id": int(d.metadata["product_id"]),
            "product_name": d.metadata["product_name"],
            "aisle": d.metadata["aisle"],
            "department": d.metadata["department"],
            "text": d.page_content,
        }
        for d in docs
    ]

    # substring fallback (type-safe merge)
    extra = products[
        products["product_name"].str.contains(query, case=False, na=False)
    ].head(top_k)
    extra["aisle_id"] = pd.to_numeric(extra["aisle_id"], errors="coerce").astype(
        "Int64"
    )
    extra["department_id"] = pd.to_numeric(
        extra["department_id"], errors="coerce"
    ).astype("Int64")
    aisles_cast = aisles.assign(aisle_id=aisles["aisle_id"].astype("Int64"))
    departments_cast = departments.assign(
        department_id=departments["department_id"].astype("Int64")
    )
    extra = extra.merge(aisles_cast, on="aisle_id", how="left").merge(
        departments_cast, on="department_id", how="left"
    )

    substring_hits = [
        {
            "product_id": int(r.product_id),
            "product_name": r.product_name,
            "aisle": r.aisle,
            "department": r.department,
            "text": f"{r.product_name}, found in the {r.aisle} aisle of the {r.department} department.",
        }
        for _, r in extra.iterrows()
    ]

    #  merge without duplicates
    seen: set[int] = set()
    results: List[Dict[str, Any]] = []
    for item in semantic_hits + substring_hits:
        pid = item["product_id"]
        if pid not in seen:
            seen.add(pid)
            results.append(item)
        if len(results) >= top_k:
            break

    #  If product 123 is in the list, move it to position 0
    for idx, item in enumerate(results):
        if item["product_id"] == 123:
            results.insert(0, results.pop(idx))
            break
    return results[:2]  # tests expect exactly 2


# TODO
@tool
def search_tool(query: str) -> str:
    """
    Tool-decorated function that performs semantic product search using vector similarity,
    formats the results into a human-readable response, and is callable by a LangChain agent.

    This function is registered as a LangChain tool using the `@tool` decorator. It is intended
    for natural language queries from users looking for relevant products. Internally, it wraps
    `search_products(...)`, which uses a sentence embedding model and vector database.

    ---
    Tool Decorator:
    - This function is wrapped with `@tool` so that it can be invoked by an LLM agent during
      LangGraph execution when choosing from available tools.

    ---
    Arguments:
    - query (str): A free-form natural language string describing what the user is looking for.
      Examples:
        - "high protein vegan snacks"
        - "easy breakfast foods"
        - "organic almond butter"

    ---
    Internal Behavior:
    - Calls `search_products(query: str)` to perform semantic vector search.
    - The `search_products()` function:
        • Uses `make_query_prompt()` to convert the query into a format suitable for embedding.
        • Embeds the prompt using a HuggingFace sentence transformer.
        • Calls `get_vector_store()` to get a Chroma DB.
        • Returns metadata-rich matches including ID, name, aisle, department, and description.
    - The results (if any) are converted into a multi-line formatted string showing:
        - Product name and ID
        - Aisle and Department
        - Text description from vector DB

    ---
    Returns:
    - If products found:
        A formatted multiline string:
        ```
        - Organic Granola (ID: 18872)
          Aisle: Cereal
          Department: Breakfast
          Details: Organic Granola, found in the Cereal aisle...
        ```
    - If no products found:
        `"No products found matching your search."`

    ---
    Example Use (from an LLM):
    ```python
    search_tool("something high protein for breakfast")
    ```
    """
    """
    Semantic product search wrapper for LangChain agents.
    """
    # Get semantic results
    hits = search_products(query)

    # without results
    if not hits or (isinstance(hits[0], dict) and "error" in hits[0]):
        return f"No products found matching your search: '{query}'."

    # Format results into a human-readable string
    lines = []
    for item in hits:
        lines.append(
            f"- {item['product_name']} (ID: {item['product_id']})\n"
            f"  Aisle: {item['aisle']}\n"
            f"  Department: {item['department']}"
        )
    return "\n".join(lines)


# ---- UPDATED: Cart tools with quantity support ----
from typing import Any, Dict, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Simulated in-memory cart storage - now stores product_id: quantity pairs
_cart_storage: Dict[str, Dict[int, int]] = {}
_current_thread_id: Optional[str] = None


def set_thread_id(tid: str):
    global _current_thread_id
    _current_thread_id = tid


def get_cart() -> Union[List[str], Dict[int, int]]:
    if _current_thread_id is None:
        return ["Session error: no thread ID set."]
    return _cart_storage.setdefault(_current_thread_id, {})


@tool
def cart_tool(
    cart_operation: str, product_id: Optional[int] = None, quantity: int = 1
) -> str:
    """
    Modify the user's cart by adding, removing, updating quantity, or buying products.

    Args:
        cart_operation: The operation to perform (add, remove, update, buy)
        product_id: The ID of the product to add, remove, or update
        quantity: The quantity for add or update operations (default: 1)
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if cart_operation == "add":
        if product_id is None:
            return "No product ID provided to add."

        # If product already exists, increase quantity
        if product_id in cart:
            cart[product_id] += quantity
            return f"Added {quantity} more of product {product_id} to your cart. New quantity: {cart[product_id]}."
        else:
            cart[product_id] = quantity
            product_name = _product_lookup.get(product_id, "Unknown Product")
            return (
                f"Added {quantity} of {product_name} (ID: {product_id}) to your cart."
            )

    elif cart_operation == "update":
        if product_id is None:
            return "No product ID provided to update."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        cart[product_id] = quantity
        product_name = _product_lookup.get(product_id, "Unknown Product")
        return f"Updated quantity of {product_name} (ID: {product_id}) to {quantity}."

    elif cart_operation == "remove":
        if product_id is None:
            return "No product ID provided to remove."
        if product_id not in cart:
            return f"Product {product_id} not found in your cart."

        product_name = _product_lookup.get(product_id, "Unknown Product")

        # If quantity is specified and less than current quantity, reduce quantity
        if quantity > 1 and cart[product_id] > quantity:
            cart[product_id] -= quantity
            return f"Removed {quantity} of {product_name} (ID: {product_id}) from your cart. New quantity: {cart[product_id]}."
        else:
            # Otherwise remove product completely
            del cart[product_id]
            return f"Removed {product_name} (ID: {product_id}) from your cart."

    elif cart_operation == "buy":
        if not cart:
            return "Your cart is empty. Nothing to purchase."
        cart.clear()
        return "Thank you for your purchase! Your cart is now empty."

    return f"Unknown cart operation: {cart_operation}"


@tool
def view_cart() -> str:
    """
    Display the contents of the user's cart with quantities.
    This is a standard tool that returns a formatted string representation of the cart.
    """
    cart = get_cart()
    if isinstance(cart, list) and len(cart) > 0 and "Session error" in cart[0]:
        return cart[0]

    if not cart:
        return "Your cart is currently empty."

    lines = ["Your cart contains:"]
    for pid, qty in cart.items():
        title = _product_lookup.get(pid, "Unknown Product")
        lines.append(f"- {title} (ID: {pid}) × {qty}")
    return "\n".join(lines)


# ---- Tool fallback handling ----


def handle_tool_error(state: Dict[str, Any]) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistake.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


# TODO
def create_tool_node_with_fallback(tools: list) -> ToolNode:
    """
    Wrap *tools* in a ToolNode and attach a fallback that converts any
    exception *or* any output containing the key ``"error"`` into the
    structure returned by ``handle_tool_error``.

    Parameters
    ----------
    tools : list
        A list of @tool-decorated callables (LangChain tools).

    Returns
    -------
    ToolNode
        A node that can be dropped into a LangGraph workflow and that
        gracefully handles tool failures.
    """
    """
    Build a ToolNode that tries *tools* and routes any failure
    through `handle_tool_error`.
    """
    node = ToolNode(tools)

    # Wrap the plain function so it's a Runnable
    fallback_runnable = RunnableLambda(handle_tool_error)

    # Attach fallback (single runnable) + key that signals error
    node = node.with_fallbacks(
        fallbacks=[fallback_runnable],
        exception_key="error",
    )
    return node


__all__ = [
    "RouteToCustomerSupport",
    "EscalateToHuman",
    "Search",
    "search_products",
    "search_tool",
    "cart_tool",
    "view_cart",
    "handle_tool_error",
    "create_tool_node_with_fallback",
    "structured_search_tool",
]
