from datetime import date

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Returns today's date in YYYY-MM-DD format.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web_stub",
            "description": "Search the web for a query. Returns a summary (demo stub — not a real search).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"],
            },
        },
    },
]

# OSS prompt-based tool descriptions (injected into system prompt)
OSS_TOOL_PROMPT = """You can optionally use these tools:
1. get_current_date() — only when the user asks about today's date or the current day.
2. search_web_stub(query) — only when the user explicitly asks to search the web.

Answer DIRECTLY for everything else — math, general knowledge, definitions,
explanations, and conversation. Do NOT use a tool for those.

ONLY if a tool is genuinely required, respond with a single line of JSON and nothing else:
{"tool": "get_current_date"}
or
{"tool": "search_web_stub", "query": "your search query"}

After a tool result is provided, reply in plain natural language (never JSON)."""


def get_current_date() -> str:
    return date.today().isoformat()


def search_web_stub(query: str) -> str:
    return (
        f"[STUB] Web search results for '{query}': "
        "This is a demo stub. In production, this would call a real search API "
        "such as Brave Search or SerpAPI and return live results."
    )


def dispatch_tool(tool_name: str, args: dict) -> str:
    if tool_name == "get_current_date":
        return get_current_date()
    if tool_name == "search_web_stub":
        return search_web_stub(args.get("query", ""))
    return f"Unknown tool: {tool_name}"
