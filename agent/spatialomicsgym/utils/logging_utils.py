from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.utils.interactive_env import is_interactive_env

_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}


def color_print(text, color="blue"):
    color_str = _TEXT_COLOR_MAPPING[color]
    print(f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m")


def pretty_print(message, printout=True):
    if isinstance(message, tuple):
        title = message
    elif isinstance(message.content, list):
        title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"

        for i in message.content:
            if not isinstance(i, dict):
                continue  # a bare-string content block (non-Anthropic providers) must not crash logging
            if i.get("type") == "text":
                title += f"\n{i.get('text', '')}\n"
            elif i.get("type") == "tool_use":
                title += f"\nTool: {i.get('name')}"
                title += f"\nInput: {i.get('input')}"
        if printout:
            print(f"{title}")
    else:
        title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"
        title += f"\n\n{message.content}"
        if printout:
            print(f"{title}")
    return title


class PromptLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        for message in messages[0]:
            color_print(message.pretty_repr(), color="green")


class NodeLogger(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):  # response of type LLMResult
        for generations in response.generations:  # response.generations of type List[List[Generations]] becuase "each input could have multiple candidate generations"
            for generation in generations:
                generated_text = generation.message.content
                # token_usage = generation.message.response_metadata["token_usage"]
                color_print(generated_text, color="yellow")

    def on_agent_action(self, action, **kwargs):
        color_print(action.log, color="pink")

    def on_agent_finish(self, finish, **kwargs):
        color_print(finish, color="red")

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name")
        color_print(f"Calling {tool_name} with inputs: {input_str}", color="pink")

    def on_tool_end(self, output, **kwargs):
        output = str(output)
        color_print(output, color="blue")
