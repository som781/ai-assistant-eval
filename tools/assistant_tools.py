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
OSS_TOOL_PROMPT = """TOOLS:
1. get_current_date() — returns today's date
2. search_web_stub(query: str) — returns a web search summary (demo only)

To use a tool, respond ONLY with JSON on a single line:
{"tool": "get_current_date"}
or
{"tool": "search_web_stub", "query": "your search query"}

After receiving a tool result, answer the user naturally."""


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
