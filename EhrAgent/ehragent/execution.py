LOG_SEPARATOR = "\n----------------------------------------------------------\n"


def _flatten_logs(logs, question, answer=None):
    """Convert Autogen message dicts into a flat string list and joined log."""
    logs_string = [f"[question] {question}"]
    ordered_entries = []
    if isinstance(logs, dict):
        for agent in list(logs.keys()):
            for entry in logs[agent]:
                ordered_entries.append((entry, agent))
    else:
        ordered_entries = [(entry, None) for entry in logs]

    for entry, agent in ordered_entries:
        content = entry.get("content") if isinstance(entry, dict) else entry
        if content is None and isinstance(entry, dict):
            function_call = entry.get("function_call")
            if function_call:
                content = function_call.get("arguments")
            else:
                content = str(entry)
        if isinstance(content, dict) and "cell" in content:
            content = content["cell"]
        label = entry.get("role") if isinstance(entry, dict) else None
        label = label or (entry.get("name") if isinstance(entry, dict) else None) or agent or "message"
        logs_string.append(f"[{label}] {content}")
    return logs_string, LOG_SEPARATOR.join(logs_string)


def execute_with_memory(user_proxy, chatbot, question, answer, selected_memory, num_shots=None):
    # Execute task with given memory records and return logs
    memory_shots = num_shots if num_shots is not None else len(selected_memory)
    user_proxy.update_memory(memory_shots, selected_memory)

    user_proxy.initiate_chat(
        chatbot,
        message=question,
    )

    logs_source = getattr(user_proxy, "chat_messages", {}).get(chatbot, None)
    if logs_source is None:
        logs_source = user_proxy._oai_messages

    return _flatten_logs(logs_source, question, answer)
