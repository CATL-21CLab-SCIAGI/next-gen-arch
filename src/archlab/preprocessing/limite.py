"""Native Violetto rendering; tool trajectories are explicitly excluded."""


def exclusion_reason(row, tokenizer=None):
    if row.get("tools") or any(
        m.get("role") in ("tool", "function") or m.get("tool_calls") or m.get("function_call")
        for m in row["messages"]
    ):
        return "tools_unsupported_by_native_template"
    if tokenizer is not None:
        from jinja2.exceptions import TemplateError

        for message in row["messages"]:
            if message.get("role") == "system":
                try:
                    tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
                except TemplateError:
                    return "system_prompt_unsupported_by_native_template"
    return None


def render_row(tokenizer, row):
    if exclusion_reason(row, tokenizer):
        raise ValueError("Incompatible trajectory must be recorded in the exclusion manifest")
    # Parquet represents absent calls as []; the native template rejects even
    # empty non-null call fields. Remove only these empty optional fields.
    messages = [
        {k: v for k, v in m.items() if k not in ("tool_calls", "function_call")}
        for m in row["messages"]
    ]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected a completed assistant trajectory")
    kwargs = dict(add_generation_prompt=False)
    text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    # Violetto's conversation terminator differs from tokenizer.eos_token.
    if not text.endswith("<|im_end|>\n"):
        raise ValueError("Unexpected native conversation ending")
    return text, False, messages, kwargs
