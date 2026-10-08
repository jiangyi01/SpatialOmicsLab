import re

from spatialomicsgym.llm import get_llm, message_to_text


class base_agent:
    def __init__(self, llm=None, cheap_llm=None, tools=None, temperature=0.7):
        # No pinned id: the configured model (SOG_LLM). The pinned claude-3-haiku-20240307 is retired,
        # so every call with no model named 404'd (u16-llm-config-19).
        if llm is None:
            from spatialomicsgym.config import default_config

            llm = default_config.llm
        self.tools = tools
        self.llm = get_llm(llm, temperature)
        if cheap_llm is None:
            self.cheap_llm = llm
        else:
            self.cheap_llm = cheap_llm

    def configure(self):
        pass

    def go(self, input):
        pass


class FunctionGenerator(base_agent):
    """Agent that generates executable Python code scripts given a task description."""

    def __init__(self, llm=None, cheap_llm=None, temperature=0.7):
        """Initialize the PaperTaskExtractor agent.

        Args:
            llm (str): The LLM model to use
            cheap_llm (str, optional): A cheaper LLM for simpler tasks
        """
        # `temperature` MUST be keyword: base_agent.__init__ is (llm, cheap_llm, tools=None,
        # temperature=0.7), so passing it positionally lands it in the `tools` slot — self.tools
        # becomes a float and the LLM is silently built at the default 0.7, ignoring this argument.
        super().__init__(llm, cheap_llm, temperature=temperature)
        self.log = []
        self.configure()

    def configure(self):
        """Configure the agent with appropriate prompts."""
        # Prompt for Python code generation
        self.system_prompt = """You are a senior Python engineer. Generate robust, idiomatic Python code that solves the user's task. Requirements:
        1. Output ONLY Python code, ideally inside a single triple-backtick code block.
        2. Include minimal inline comments and a small docstring.
        3. Add a `main()` and an `if __name__ == '__main__':` guard when appropriate.
        4. Avoid external dependencies unless necessary; if used, show `pip` installs in comments.
        5. Do not include prose before or after the code.
        6. When applicable, prioritize the use of codes on public repositories, such as HuggingFace or Github

        Generate Python codes for the following task:
        {task}
"""

    def _generate_code(self, task_description: str) -> str:
        """Generate codes given a task description.
        Args:
            task_description (str): task descriptions (possibly generated from previous steps)

        Returns:
            str: generated code string

        """
        prompt = self.system_prompt.format(task=task_description)
        message = self.llm.invoke(prompt)
        # Responses-API models return a list of content blocks. `_extract_code_block` rejects a
        # non-str and returns "", so the caller silently wrote an EMPTY script to disk.
        return message_to_text(message)

    def _generate_script_filename(self, task_description: str, max_words: int = 6) -> str:
        """
        Generate a safe, meaningful Python script filename from a task description.
        Poised for update: may ask the agent to suggest meaningful names.

        Parameters:
        -----------
        task_description (str): task descriptions (possibly generated from previous steps)

        max_words : int
            Maximum number of words to include in the filename.

        Returns:
        --------
            str
            A lowercase, hyphen-free, safe filename ending in '.py'.
        """
        # Lowercase and remove non-alphanumeric (allow spaces for splitting)
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", task_description.lower())

        # Tokenize and select top words
        words = cleaned.split()
        selected_words = words[:max_words] if words else ["script"]

        # Join with underscores
        base_name = "_".join(selected_words)
        return f"{base_name}.py"

    def go(self, task_description: str):
        """Implement the inherited function to get the tasks done.

        Args:
            task_description (str): task descriptions (possibly generated from previous steps)

        Returns:
            tuple: (script_filename, results) where script_filename is a generated name for script file and results is the generated codes

        """
        self.log = []
        self.log.append(
            (
                "user",
                "Generate Python codes given a task description",
            )
        )

        script_filename = self._generate_script_filename(task_description)
        results = self._generate_code(task_description)
        return script_filename, self._extract_code_block(results)

    def _extract_code_block(self, s: str) -> str:
        """Extract the first fenced code block from ``s``.

        The generation prompt only *prefers* a triple-backtick fence, so a compliant
        model may return bare code. Two hazards the naive version had:

        * **No fence -> ``None``.** The sole caller writes the result straight to disk,
          so ``None`` raised ``TypeError`` and aborted the whole batch. We fall back to
          the stripped response instead, so the caller always gets writable code.
        * **Language tag leak.** The old ``(?:python)?`` only stripped the exact tag
          ``python``; a ```` ```py ```` / ```` ```python3 ```` fence left ``py`` /
          ``python3`` as a bogus first code line (``SyntaxError`` when run). We strip
          any short language token after the opening fence.
        """
        if not isinstance(s, str):
            return ""
        m = re.search(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?(.+?)```", s, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        return s.strip().strip("`").strip()
