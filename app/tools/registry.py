from app.tools.interfaces import python_sandbox, self_reflection, sql_lookup, web_search_stub

TOOL_REGISTRY = {
    "web_search": web_search_stub,
    "python_sandbox": python_sandbox,
    "sql_lookup": sql_lookup,
    "self_reflection": self_reflection,
}
