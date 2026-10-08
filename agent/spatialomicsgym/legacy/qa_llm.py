from langchain_core.prompts import ChatPromptTemplate

from spatialomicsgym.llm import get_llm, message_to_text


class qa_llm:
    def __init__(self, path="./data", llm=None, lab_bench_reproduce=False):
        from spatialomicsgym.config import default_config

        self.path = path
        # Route through get_llm + config so model/source/timeout come from the live config, not a
        # hard-coded (now-deprecated) default id that would 404 on the eval path.
        self.llm = get_llm(llm, config=default_config)

        if lab_bench_reproduce:
            self.prompt_modifier = """
The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.

Think step by step. \n
            """
        else:
            self.prompt_modifier = ""
        self.log = []

    def configure(self):
        pass

    def go(self, input):
        self.log = []
        self.log.append(("user", input))
        message = self.llm.invoke(self.prompt_modifier + input)
        # Responses-API models return `content` as a list of blocks; the grader (and every caller)
        # expects the answer as text, so normalize before it leaves this function.
        answer = message_to_text(message)
        self.log.append(("assistant", answer))
        return [answer], answer

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        _r = checker_llm.invoke({"messages": [("user", str(self.log))]})
        result = _r.model_dump() if hasattr(_r, "model_dump") else _r.dict()
        return result
