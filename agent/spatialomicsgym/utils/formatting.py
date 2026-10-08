def _described(param):
    """The prose to show for one parameter, or the placeholder when the catalog gives none.

    ``param.get("description", "No description")``'s own fallback can never fire for an MCP parameter:
    ``add_mcp`` builds every entry with ``"description": param_spec.get("description", "")``
    (``agent/mcp_integration.py:632``), so the key is always present, holding an empty string when the
    config declares no prose. An outer always-insert fallback defeats an inner one, and 83 parameters --
    18 required and 65 optional -- reached the model as ``- output_dir (str): `` with nothing at all after
    the colon. Seventeen of the required ones are ``output_dir``, whose name carries most of its meaning;
    the other two are input paths, where a blank slot leaves the model guessing whether a Visium
    directory, a raw counts matrix or a processed object is wanted.

    A blank value tells the reader exactly as much as a missing key, so it is treated the same way. A
    description that exists is passed through byte for byte, whitespace and newlines included.
    """
    desc = param.get("description") or ""
    return desc if str(desc).strip() else "No description"


def textify_api_dict(api_dict):
    """Convert a nested API dictionary to a nicely formatted string."""
    lines = []
    for category, methods in api_dict.items():
        lines.append(f"Import file: {category}")
        lines.append("=" * (len("Import file: ") + len(category)))
        for method in methods:
            lines.append(f"Method: {method.get('name', 'N/A')}")
            lines.append(f"  Description: {method.get('description', 'No description provided.')}")

            # Process required parameters
            req_params = method.get("required_parameters", [])
            if req_params:
                lines.append("  Required Parameters:")
                for param in req_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = _described(param)
                    # A required parameter has no default -- that is what "required" means. Both halves of
                    # the catalog hand us one anyway: the builtin descriptions carry an explicit
                    # ``"default": None``, and ``add_mcp`` inserts the key as
                    # ``param_spec.get("default", None)``. Printing the clause unconditionally therefore told
                    # the model that 700 parameters it must supply default to None -- and nothing accepts None
                    # for them, since the MCP wrapper forwards kwargs verbatim and the server signature has no
                    # default. One description had to be written as "There is no default - pass the user's own
                    # file" to argue with this very line. A required parameter that declares a *real* value
                    # keeps it: whether it should be marked required at all is a config question, and the value
                    # is still information the model can act on.
                    param_default = param.get("default")
                    default_clause = "" if param_default is None else f" [Default: {param_default}]"
                    lines.append(f"    - {param_name} ({param_type}): {param_desc}{default_clause}")

            # Process optional parameters
            opt_params = method.get("optional_parameters", [])
            if opt_params:
                lines.append("  Optional Parameters:")
                for param in opt_params:
                    param_name = param.get("name", "N/A")
                    param_type = param.get("type", "N/A")
                    param_desc = _described(param)
                    param_default = param.get("default", "None")
                    lines.append(f"    - {param_name} ({param_type}): {param_desc} [Default: {param_default}]")

            lines.append("")  # Empty line between methods
        lines.append("")  # Extra empty line after each category

    return "\n".join(lines)


def clean_message_content(content: str) -> str:
    """Clean message content by removing ANSI escape codes.

    This function removes ANSI escape sequences (like color codes) from text content
    that might be present in terminal output or console messages. This ensures clean
    text for markdown generation and PDF conversion.

    Args:
        content: The raw message content that may contain ANSI escape codes

    Returns:
        Cleaned content with ANSI escape codes removed

    Example:
        >>> clean_message_content("Hello \x1b[31mworld\x1b[0m!")
        "Hello world!"
    """
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", content)


def should_skip_message(clean_output: str) -> bool:
    """Check if message should be skipped during markdown generation.

    This function determines whether a message should be excluded from the final
    markdown output. It skips empty or meaningless messages but preserves important
    error messages that should be displayed to users.

    Args:
        clean_output: The cleaned message content to evaluate

    Returns:
        True if the message should be skipped, False otherwise

    Note:
        Parsing error messages are intentionally not skipped as they provide
        important feedback to users about conversation flow issues.
    """
    return (
        clean_output.strip() in ["", "None", "null", "undefined"]
        # Don't skip parsing error messages - they should be displayed and increment step counter
        # or "There are no tags" in clean_output
        # or "Execution terminated due to repeated parsing errors" in clean_output
    )


def has_execution_results(clean_output: str, execution_results) -> bool:
    """Check if message contains code execution and has associated results.

    This function determines whether a message contains executable code and has
    corresponding execution results available for display in the markdown output.

    Args:
        clean_output: The cleaned message content to check for execute tags
        execution_results: List of execution results from the agent's execution history

    Returns:
        True if the message contains <execute> tags and has execution results available
    """
    return "<execute>" in clean_output and execution_results is not None and execution_results


def find_matching_execution(clean_output: str, execution_results) -> dict | None:
    """Find the execution result that matches the given message content.

    This function searches through the execution results to find the one that
    corresponds to the current message. It matches based on the triggering message
    content to associate execution results with their originating AI messages.

    Args:
        clean_output: The cleaned message content to match against
        execution_results: List of execution result dictionaries containing
                         triggering messages and execution data

    Returns:
        The matching execution result dictionary if found, None otherwise

    Note:
        The matching is bidirectional - it checks if either the triggering message
        is contained in the current output or vice versa to handle partial matches.

        Two rules beyond that, both because a bidirectional substring test over a list is a blunt
        instrument. An **exact** match beats a substring one, since any short block is contained
        in a longer block that quotes it. And within each pass the list is read **newest first**,
        so a block the model runs twice is rendered with the figures of the run it just did, not
        the first one. First-match-wins over a list that used to span every turn meant the first
        turn to run ``<execute>print(adata)</execute>`` owned that text for the life of the
        process; that list is now cleared per turn, and this makes the remaining within-turn
        repeat come out right too.
    """
    results = list(execution_results or [])
    for exec_result in reversed(results):
        if exec_result.get("triggering_message") == clean_output:
            return exec_result
    for exec_result in reversed(results):
        trigger = exec_result.get("triggering_message") or ""
        if trigger and (trigger in clean_output or clean_output in trigger):
            return exec_result
    return None


def create_parsing_error_html() -> str:
    """Create HTML markup for displaying parsing errors in markdown output.

    This function generates a styled HTML block that displays parsing errors
    when the agent's response doesn't contain the required tags. The HTML
    uses CSS classes for consistent styling in the final PDF output.

    Returns:
        HTML string containing a styled parsing error message box

    Note:
        The returned HTML uses CSS classes defined in get_pdf_css_content()
        for consistent styling across the document.
    """
    return """
<div class="parsing-error-box">
    <div class="parsing-error-header">Parsing Error</div>
    <div class="parsing-error-content">Each response must include thinking process followed by either execute or solution tag. But there are no tags in the current response.</div>
</div>
"""


def format_execute_tags_in_content(content: str, parse_tool_calls_with_modules_func) -> str:
    """Format execute tags in content by extracting code and creating highlighted tool call blocks.

    This function processes content that contains <execute>...</execute> tags and
    converts them into styled HTML blocks that display the code with syntax highlighting
    and information about which tools are being used.

    Args:
        content: The content string that may contain <execute> tags
        parse_tool_calls_with_modules_func: Function to parse tool calls with modules
                                          (typically parse_tool_calls_with_modules)

    Returns:
        Formatted content with execute tags converted to highlighted tool call blocks.
        Also processes <solution> tags in the same pass.

    Note:
        The function also calls format_solution_tags_in_content() to handle
        solution tags in the same processing pass.
    """
    import re

    # Pattern to match <execute>...</execute> blocks
    execute_pattern = r"<execute>(.*?)</execute>"

    def replace_execute_tag(match):
        code_content = match.group(1).strip()
        language, tool_name = detect_code_language_and_tool(code_content)
        code_content = clean_code_content(code_content, language)

        # Parse tools from the code content with module information
        detected_tool_modules = parse_tool_calls_with_modules_func(code_content)

        # Create the formatted block
        formatted_block = create_tool_call_block(code_content, language, tool_name, detected_tool_modules)
        return formatted_block

    # Replace all execute tags with formatted tool call blocks
    formatted_content = re.sub(execute_pattern, replace_execute_tag, content, flags=re.DOTALL)

    # Also format solution tags
    formatted_content = format_solution_tags_in_content(formatted_content)

    return formatted_content


# Kept in step with spatialomicsgym.action, which decides what actually executes. Imported rather
# than re-spelled so the two can never drift apart.
from spatialomicsgym.action import _SHEBANG_BASH, _SHEBANG_R


def detect_code_language_and_tool(code_content: str) -> tuple[str, str]:
    """Detect the programming language and tool name from code content.

    This function analyzes code content to determine the programming language
    and appropriate tool name based on language markers at the beginning of
    the code block.

    Args:
        code_content: The code content to analyze for language markers

    Returns:
        Tuple containing (language, tool_name) where:
        - language: The detected programming language ("python", "r", "bash")
        - tool_name: The human-readable tool name for display

    Example:
        >>> detect_code_language_and_tool("#!R\nlibrary(ggplot2)")
        ("r", "R REPL")
        >>> detect_code_language_and_tool("#!BASH\necho 'hello'")
        ("bash", "Bash Script")
    """
    if code_content.startswith("#!R") or code_content.startswith("# R code") or code_content.startswith("# R script"):
        return "r", "R REPL"
    elif code_content.startswith(_SHEBANG_R):
        return "r", "R REPL"
    elif code_content.startswith("#!BASH") or code_content.startswith("# Bash script"):
        return "bash", "Bash Script"
    elif code_content.startswith(_SHEBANG_BASH):
        # What ran and what the transcript shows have to be the same language.
        return "bash", "Bash Script"
    elif code_content.startswith("#!CLI"):
        return "bash", "CLI Command"
    else:
        return "python", "Python REPL"


def clean_code_content(code_content: str, language: str) -> str:
    """Clean code content by removing language markers.

    This function removes language-specific markers from the beginning of code
    content to prepare it for display in code blocks. The markers are used
    internally for language detection but should not appear in the final output.

    Args:
        code_content: The raw code content that may contain language markers
        language: The detected programming language ("python", "r", "bash")

    Returns:
        Cleaned code content with language markers removed

    Example:
        >>> clean_code_content("#!R\nlibrary(ggplot2)", "r")
        "library(ggplot2)"
        >>> clean_code_content("#!BASH\necho 'hello'", "bash")
        "echo 'hello'"
    """
    import re

    if language == "r":
        return re.sub(r"^#!R|^# R code|^# R script", "", code_content, count=1).strip()
    elif language == "bash":
        if code_content.startswith("#!BASH") or code_content.startswith("# Bash script"):
            return re.sub(r"^#!BASH|^# Bash script", "", code_content, count=1).strip()
        elif code_content.startswith("#!CLI"):
            return re.sub(r"^#!CLI", "", code_content, count=1).strip()
    return code_content


def create_tool_call_block(code_content: str, language: str, tool_name: str, detected_tool_modules: list) -> str:
    """Create the HTML block for tool call highlighting.

    This function generates a styled HTML block that displays code execution
    information including the code itself, syntax highlighting, and a list of
    tools that were used during execution.

    Args:
        code_content: The cleaned code content to display
        language: The programming language for syntax highlighting
        tool_name: The default tool name to display if no specific tools detected
        detected_tool_modules: List of (tool_name, module_name) tuples for tools used

    Returns:
        HTML string containing a styled tool call block with code and tool information

    Note:
        The HTML uses CSS classes defined in get_pdf_css_content() for styling.
        If no specific tools are detected, it falls back to a default tool name.
    """
    # Create the formatted block with code and tools used
    formatted_block = f"""<div class="tool-call-highlight">
<div class="tool-call-header">
<strong>Code Execution</strong>
</div>
<div class="tool-call-input">
```{language}
{code_content}
```
</div>"""

    # Add tools used section
    if detected_tool_modules:
        tools_list = format_detected_tools(detected_tool_modules)
        formatted_block += f"""
<div class="tools-used">
<strong>Tools Used:</strong> {tools_list}
</div>"""
    else:
        formatted_block += format_default_tool_name(language, tool_name)

    formatted_block += "</div>"
    return formatted_block


def format_detected_tools(detected_tool_modules: list) -> str:
    """Format detected tools with their modules for display.

    This function takes a list of (tool_name, module_name) tuples and formats
    them into a human-readable string for display in the tool call blocks.
    It handles special cases for common tools and formats module names appropriately.

    Args:
        detected_tool_modules: List of (tool_name, module_name) tuples

    Returns:
        Comma-separated string of formatted tool descriptions

    Example:
        >>> format_detected_tools([("analyze_data", "spatialomicsgym.tool"), ("pandas", "pandas")])
        "spatialomicsgym \u2192 analyze_data, pandas \u2192 pandas"
    """
    tool_descriptions = []
    for tool_name, module_name in detected_tool_modules:
        if tool_name == "python_repl":
            tool_descriptions.append("Python REPL")
        elif tool_name == "r_repl":
            tool_descriptions.append("R REPL")
        elif "bash" in tool_name.lower():
            tool_descriptions.append("Bash Script")
        else:
            # Extract the last part of the module name for display
            display_module = module_name.split(".")[-1] if "." in module_name else module_name
            tool_descriptions.append(f"{display_module} \u2192 {tool_name}")

    return ", ".join(sorted(tool_descriptions))


def format_default_tool_name(language: str, tool_name: str) -> str:
    """Format default tool name based on programming language.

    This function generates HTML for displaying the default tool name when
    no specific tools are detected in the code. It maps programming languages
    to their appropriate default tool names.

    Args:
        language: The programming language ("python", "r", "bash")
        tool_name: The detected tool name (used for bash CLI vs script distinction)

    Returns:
        HTML string containing a styled tools-used section

    Note:
        For bash, it distinguishes between CLI commands and bash scripts
        based on the tool_name parameter.
    """
    if language == "r":
        return """
<div class="tools-used">
<strong>Tools Used:</strong> R REPL
</div>"""
    elif language == "bash":
        if tool_name == "CLI Command":
            return """
<div class="tools-used">
<strong>Tools Used:</strong> CLI Command
</div>"""
        else:
            return """
<div class="tools-used">
<strong>Tools Used:</strong> Bash Script
</div>"""
    else:
        return """
<div class="tools-used">
<strong>Tools Used:</strong> Python REPL
</div>"""


def format_solution_tags_in_content(content: str) -> str:
    """Format solution tags in content by extracting text and formatting as solution blocks.

    This function processes content that contains <solution>...</solution> tags and
    converts them into styled HTML blocks that display solution content with appropriate
    formatting and CSS classes.

    Args:
        content: The content string that may contain <solution> tags

    Returns:
        Formatted content with solution tags converted to styled solution blocks

    Note:
        The solution blocks use the "title-text summary" CSS class for consistent
        styling with other content blocks in the markdown output.
    """
    from spatialomicsgym.answer import SOLUTION_TAG_RE

    def replace_solution_tag(match):
        solution_content = match.group(1).strip()
        # Format as regular text, not terminal
        return f"""<div class="title-text summary">
<div class="title-text-header">
<strong>Summary and Solution</strong>
</div>
<div class="title-text-content">
{solution_content}
</div>
</div>"""

    # Replace all solution tags with formatted solution blocks
    formatted_content = SOLUTION_TAG_RE.sub(replace_solution_tag, content)

    return formatted_content


def format_observation_as_terminal(content: str) -> str | None:
    """Format observation content with terminal-like styling.

    This function processes observation content from the agent's execution results
    and formats it as a styled terminal block. It handles both text and image content,
    with length limits to ensure the output fits within PDF page constraints.

    Args:
        content: The observation content string, potentially containing <observation> tags

    Returns:
        Formatted HTML content with terminal styling, or None if observation is
        empty, invalid, or contains only meaningless content

    Note:
        - Content is limited to 10,000 characters to fit within 2 A4 pages
        - Handles both text and base64-encoded images
        - Uses CSS classes for consistent styling with other content blocks
    """
    import re

    # Character limit for 2 A4 pages (approximately 10,000 characters)
    MAX_OBSERVATION_LENGTH = 10000

    # Remove the <observation> tags and extract the content
    observation_pattern = r"<observation>(.*?)</observation>"
    observation_match = re.search(observation_pattern, content, re.DOTALL)

    if observation_match:
        observation_content = observation_match.group(1).strip()
    else:
        # Fallback if no observation tags found - check if content is meaningful
        if not (content.strip() and content.strip() not in ["", "None", "null", "undefined"]):
            return None
        observation_content = content.strip()

    # Skip empty observations
    if not observation_content or observation_content in ["", "None", "null", "undefined"]:
        return None

    # Base64-embedded plots must never be truncated: slicing a data: URI mid-payload corrupts the
    # image and leaks the raw "[Output truncated ...]" notice into its src. Only cap plain text.
    has_image = "data:image/" in observation_content
    if not has_image and len(observation_content) > MAX_OBSERVATION_LENGTH:
        cropped_content = observation_content[:MAX_OBSERVATION_LENGTH]
        truncation_notice = f"\n\n[Output truncated - content was too long to display here ({len(observation_content)} characters total)]"
        observation_content = cropped_content + truncation_notice

    # Check if it contains plot data (base64 images)
    if has_image:
        content_html = process_observation_with_images(observation_content)
    else:
        # Regular text output - format as terminal output
        content_html = f"```terminal\n{observation_content}\n```"

    return f"""<div class="title-text observation">
<div class="title-text-header">
<strong>Observation</strong>
</div>
<div class="title-text-content">
{content_html}
</div>
</div>"""


def process_observation_with_images(observation_content: str) -> str:
    """Process observation content that contains both text and base64-encoded images.

    This function handles observation content that includes both text output and
    base64-encoded images (typically plots from data analysis). It separates the
    text and image content and formats them appropriately for markdown display.

    Args:
        observation_content: The observation content containing both text and images

    Returns:
        HTML string containing formatted text (as terminal blocks) and images
        (as markdown image tags)

    Note:
        The function uses "data:image/" as a delimiter to split content into
        text and image parts, then processes each part separately.
    """
    # Split content into text and image parts
    parts = observation_content.split("data:image/")
    text_parts = []
    image_parts = []

    for i, part in enumerate(parts):
        if i == 0:
            # First part is text only
            if part.strip():
                text_parts.append(part.strip())
        else:
            # Find the end of the base64 data
            end_markers = ["\n", "\r", " ", "\t", ">", "<", "]", ")", "}"]
            image_end = len(part)
            for marker in end_markers:
                marker_pos = part.find(marker)
                if marker_pos != -1 and marker_pos < image_end:
                    image_end = marker_pos

            # Extract image data
            image_data = "data:image/" + part[:image_end]
            image_parts.append(image_data)

            # Extract remaining text
            remaining_text = part[image_end:].strip()
            if remaining_text:
                text_parts.append(remaining_text)

    # Build the content
    content_html = ""
    if text_parts:
        # Add text content as terminal output
        text_content = "\n".join(text_parts)
        content_html += f"```terminal\n{text_content}\n```\n\n"

    if image_parts:
        # Add image content
        for image_data in image_parts:
            content_html += f"![Plot]({image_data})\n\n"

    return content_html


def remove_emojis_from_text(text: str) -> str:
    """Remove emojis from text for markdown/PDF output.

    This function removes common emojis used in the system prompt and configuration
    display from text content before it's converted to markdown or PDF. This ensures
    clean, professional output while preserving emojis in the console display.

    Args:
        text: The text content that may contain emojis

    Returns:
        Text content with emojis removed

    Note:
        The function targets specific emojis used in the SpatialOmicsLab system:
        - \U0001f527 for tools
        - \U0001f4ca for data
        - \u2699\ufe0f for software
        - \U0001f4cb for configuration
        - \U0001f916 for agent
    """
    import re

    # Remove common emojis used in the system prompt, this makes conversion simpler
    emoji_patterns = [
        r"\U0001f527\s*",  # Tool emoji
        r"\U0001f4ca\s*",  # Data emoji
        r"\u2699\ufe0f\s*",  # Software emoji
        r"\U0001f4cb\s*",  # Config emoji
        r"\U0001f916\s*",  # Agent emoji
    ]

    for pattern in emoji_patterns:
        text = re.sub(pattern, "", text)

    return text


def format_lists_in_text(text: str) -> str:
    """Format numbered lists and bullet points in text to proper markdown format.

    This function processes text content to identify and format various types of lists,
    including numbered lists with checkboxes, regular lists, and plan structures.
    It also handles preprocessing tasks like removing bold formatting from plan titles
    and removing emojis for clean PDF output.

    Args:
        text: The text content to process for list formatting

    Returns:
        Formatted text with properly structured lists and cleaned formatting

    Note:
        The function performs several preprocessing steps:
        - Removes bold formatting from plan titles
        - Removes emojis for PDF output
        - Identifies and formats checkbox lists
        - Processes regular text blocks
    """
    import re

    # Preprocess to remove bold formatting from plan titles
    # Remove **Plan:**, **Updated Plan:**, **Completed Plan:**, etc.
    text = re.sub(r"\*\*([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan):\*\*", r"\1:", text)
    # Also handle cases without colons
    text = re.sub(r"\*\*([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan)\*\*", r"\1", text)
    # Handle any other bold formatting patterns for plan titles
    text = re.sub(r"<strong>([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan):</strong>", r"\1:", text)
    text = re.sub(r"<strong>([Pp]lan|Updated [Pp]lan|Completed [Pp]lan|Final [Pp]lan)</strong>", r"\1", text)

    # Remove emojis from the text for markdown/PDF output
    text = remove_emojis_from_text(text)

    lines = text.split("\n")
    list_blocks = identify_list_blocks(lines)

    # Process each block
    result_blocks = []
    for block_text, is_checkbox_list in list_blocks:
        if is_checkbox_list:
            result_blocks.append(format_single_list(block_text))
        else:
            result_blocks.append(block_text)

    return "\n".join(result_blocks)


def identify_list_blocks(lines: list) -> list[tuple[str, bool]]:
    """Identify blocks of text that contain lists.

    This function analyzes a list of text lines to identify contiguous blocks
    that contain numbered lists with checkboxes. It groups lines into blocks
    and marks whether each block contains a checkbox list or regular text.

    Args:
        lines: List of text lines to analyze

    Returns:
        List of tuples containing (block_text, is_checkbox_list) where:
        - block_text: The text content of the block
        - is_checkbox_list: True if the block contains numbered items with checkboxes

    Note:
        The function looks for patterns like "1. [ ]", "2. [\u2713]", "3. [\u2717]" to
        identify checkbox sequences and groups them into separate blocks.
    """
    import re

    list_blocks = []
    current_block = []
    in_checkbox_sequence = False

    for line in lines:
        line_stripped = line.strip()

        # Check if this line starts a numbered item with checkbox
        if re.match(r"^\d+\.\s*\[[ \u2713\u2717]\]", line_stripped):
            if not in_checkbox_sequence:
                # Start of a new checkbox sequence
                if current_block:
                    list_blocks.append(("\n".join(current_block), False))
                current_block = [line]
                in_checkbox_sequence = True
            else:
                # Continue the sequence
                current_block.append(line)
        else:
            if in_checkbox_sequence:
                # End of checkbox sequence
                if current_block:
                    list_blocks.append(("\n".join(current_block), True))
                current_block = []
                in_checkbox_sequence = False
            current_block.append(line)

    # Handle the last block
    if current_block:
        if in_checkbox_sequence:
            list_blocks.append(("\n".join(current_block), True))
        else:
            list_blocks.append(("\n".join(current_block), False))

    return list_blocks


def format_single_list(text: str) -> str:
    """Format a single list block with checkboxes and plan titles.

    This function processes a text block that may contain numbered lists with
    checkboxes and plan titles. It converts checkbox symbols to HTML list items
    and wraps the content in a styled container with appropriate CSS classes.

    Args:
        text: The text block to format, potentially containing numbered lists

    Returns:
        HTML string containing either a formatted list with plan title or
        regular text if no list items are found

    Note:
        The function recognizes plan titles like "Plan", "Updated Plan", "Completed Plan"
        and converts checkbox symbols (\u2713, \u2717) to HTML format ([x], [ ]).
    """
    import re

    lines = text.split("\n")
    list_items = []
    has_list_items = False
    plan_title = "Plan"  # Default title

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for plan title patterns
        if re.match(r"^(Plan|Updated Plan|Completed Plan)$", line, re.IGNORECASE):
            plan_title = line
            continue

        # Check for numbered lists with checkboxes (1. [ ] or 1. [\u2713] or 1. [\u2717])
        if re.match(r"^\d+\.\s*\[[ \u2713\u2717]\]", line):
            has_list_items = True
            # Extract the content after the checkbox
            content = re.sub(r"^\d+\.\s*\[[ \u2713\u2717]\]\s*", "", line)

            # Replace checkbox symbols with text format
            if "[\u2713]" in line:
                list_items.append(f"<li><strong>[x]</strong> {content}</li>")
            elif "[\u2717]" in line:
                list_items.append(f"<li><strong>[ ]</strong> {content}</li>")
            else:
                list_items.append(f"<li><strong>[ ]</strong> {content}</li>")
        else:
            # Regular text - add as is (don't convert to list items)
            list_items.append(line)

    if has_list_items and list_items:
        # This is a list - return with container div and styled title
        return f"""<div class="title-text plan">
<div class="title-text-header">
<span class="plan-title">{plan_title}</span>
</div>
<div class="title-text-content">
<ul>
{chr(10).join(list_items)}
</ul>
</div>
</div>"""
    else:
        # Regular text
        return "\n".join(list_items)


def convert_markdown_to_pdf(markdown_path: str, pdf_path: str) -> None:
    """Convert markdown file to PDF using weasyprint or fallback libraries.

    This function converts a markdown file to PDF format using multiple fallback
    strategies. It prioritizes weasyprint for better layout control, then falls back
    to markdown2pdf and finally pandoc if the preferred libraries are not available.

    Args:
        markdown_path: Path to the input markdown file
        pdf_path: Path where the output PDF file should be saved

    Raises:
        ImportError: If no PDF conversion library is available
        Exception: If PDF conversion fails for any other reason

    Note:
        The function uses minimal markdown extensions for better performance
        and applies custom CSS styling for consistent formatting.
    """
    try:
        # Try weasyprint first (better for complex layouts)
        from weasyprint import HTML
        from weasyprint.text.fonts import FontConfiguration

        # Read markdown content
        with open(markdown_path, encoding="utf-8") as f:
            markdown_content = f.read()

        # Convert markdown to HTML with minimal extensions for better performance
        import markdown

        # Use minimal extensions to improve performance
        html_content = markdown.markdown(
            markdown_content,
            extensions=["fenced_code"],  # Removed codehilite for better performance
        )

        # Add CSS styling
        css_content = get_pdf_css_content()

        # Create HTML document
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>SpatialOmicsLab Conversation History</title>
            <style>{css_content}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert to PDF with performance optimizations
        font_config = FontConfiguration()
        html_obj = HTML(string=html_doc)
        html_obj.write_pdf(pdf_path, font_config=font_config, optimize_images=True)

    except ImportError:
        # Fallback to markdown2pdf if weasyprint is not available
        try:
            from markdown2pdf import markdown2pdf

            markdown2pdf(markdown_path, pdf_path)
        except ImportError:
            # Final fallback - try using pandoc if available
            import subprocess

            try:
                subprocess.run(["pandoc", markdown_path, "-o", pdf_path], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise ImportError(
                    "No PDF conversion library available. Please install weasyprint, markdown2pdf, or pandoc."
                ) from e
    except Exception as e:
        raise Exception(f"PDF conversion failed: {e}") from e


def get_pdf_css_content() -> str:
    """Get the CSS content for PDF generation.

    This function returns a comprehensive CSS stylesheet designed specifically
    for PDF generation from markdown content. It includes styling for all
    HTML elements that may appear in the converted markdown, with optimized
    typography, spacing, and layout for print media.

    Returns:
        CSS string containing all styles needed for PDF generation

    Note:
        The CSS includes styles for:
        - Typography and font families
        - Headings and text formatting
        - Code blocks and syntax highlighting
        - Tables and lists
        - Custom classes for tool calls, observations, and plans
        - Print-optimized spacing and layout
    """
    return """
    body {
        /* Previously: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif, 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 'Twemoji', 'EmojiOne Color' */
        font-family: sans-serif;
        font-size: 9pt;
        line-height: 1.4;
        max-width: 800px;
        margin: 0 auto;
        padding: 15px;
        color: #333;
    }
    h1, h2, h3, h4, h5, h6 {
        /* Previously: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif, 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 'Twemoji', 'EmojiOne Color' */
        font-family: sans-serif;
        color: #2c3e50;
        margin-top: 1em;
        margin-bottom: 0.5em;
    }
    h1 {
        border-bottom: 2px solid #3498db;
        padding-bottom: 8px;
        font-size: 16pt;
    }
    h2 {
        border-bottom: 1px solid #bdc3c7;
        padding-bottom: 3px;
        font-size: 14pt;
    }
    h3 {
        font-size: 12pt;
    }
    h4 {
        font-size: 10pt;
        margin-top: 0.8em;
        margin-bottom: 0.3em;
    }
    h5, h6 {
        font-size: 9pt;
        margin-top: 0.6em;
        margin-bottom: 0.2em;
    }
    code {
        background-color: #f8f9fa;
        padding: 1px 3px;
        border-radius: 2px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 8pt;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    pre {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 3px;
        overflow-x: auto;
        border-left: 3px solid #3498db;
        white-space: pre-wrap;
        word-wrap: break-word;
        font-size: 8pt;
        margin: 0.5em 0;
    }
    pre code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
    }
    /* Code header styling */
    strong {
        font-size: 9pt;
        font-weight: normal;
        color: #6c757d;
        font-style: italic;
    }
    blockquote {
        border-left: 3px solid #bdc3c7;
        margin: 0.5em 0;
        padding-left: 15px;
        color: #7f8c8d;
        font-size: 8pt;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.5em 0;
        font-size: 8pt;
    }
    th, td {
        border: 1px solid #bdc3c7;
        padding: 4px 8px;
        text-align: left;
    }
    th {
        background-color: #ecf0f1;
        font-weight: bold;
    }
    img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 10px auto;
        border: 1px solid #ddd;
        border-radius: 3px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    p {
        /* Previously: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif, 'Noto Color Emoji', 'Apple Color Emoji', 'Segoe UI Emoji', 'Twemoji', 'EmojiOne Color' */
        font-family: sans-serif;
        margin: 0.3em 0;
    }
    /* Tool call highlighting - matching observation and code formatting */
    .tool-call-highlight {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 0;
        margin: 10px 0;
        overflow: hidden;
    }
    .tool-call-header {
        background-color: #e9ecef;
        color: #495057;
        padding: 8px 12px;
        margin: 0;
        font-weight: normal;
        font-size: 9pt;
        font-style: italic;
        border-bottom: 1px solid #dee2e6;
    }
    .tool-call-input {
        background-color: #f8f9fa;
        border: none;
        border-radius: 0;
        padding: 10px 12px;
        margin: 0;
        color: #333;
        font-size: 8pt;
        line-height: 1.4;
    }
    .tool-call-input strong {
        color: #495057;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
    }
    .tool-call-input pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 10px;
        margin: 0;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .tool-call-input code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
        color: #2c3e50;
    }
    .tools-used {
        background-color: #f8f9fa;
        border-top: 1px solid #dee2e6;
        padding: 8px 12px;
        margin: 0;
        font-size: 8pt;
        color: #6c757d;
    }
    .tools-used strong {
        color: #6c757d;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
    }
    /* Title-text styling - unified for observations, plans, and solutions */
    .title-text {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 0;
        margin: 10px 0;
        overflow: hidden;
    }
    .title-text-header {
        background-color: #e9ecef;
        color: #495057;
        padding: 8px 12px;
        margin: 0;
        font-weight: normal;
        font-size: 9pt;
        font-style: italic;
        border-bottom: 1px solid #dee2e6;
    }
    .title-text-header strong {
        color: #495057;
        font-weight: normal;
        font-size: 9pt;
        font-style: italic;
    }
    .title-text-content {
        background-color: #f8f9fa;
        border: none;
        border-radius: 0;
        padding: 10px 12px;
        margin: 0;
        color: #333;
        font-size: 8pt;
        line-height: 1.4;
    }
    /* Plan-specific styling - soft blue pastel */
    .title-text.plan {
        background-color: #e3f2fd;
        border-color: #bbdefb;
    }
    .title-text.plan .title-text-header {
        background-color: #bbdefb;
        color: #1976d2;
    }
    .title-text.plan .title-text-content {
        background-color: #e3f2fd;
    }
    .plan-title {
        font-style: italic;
        font-weight: normal;
        color: #1565c0;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .plan-title strong {
        font-weight: normal;
    }
    /* Code execution-specific styling - matching title-text styling */
    .tool-call-highlight {
        background-color: #f8f9fa;
        border-color: #e9ecef;
    }
    .tool-call-header {
        background-color: #e9ecef;
        color: #495057;
    }
    .tool-call-input {
        background-color: #f8f9fa;
        color: #333;
    }
    /* Observation-specific styling - soft purple pastel */
    .title-text.observation {
        background-color: #f3e5f5;
        border-color: #e1bee7;
    }
    .title-text.observation .title-text-header {
        background-color: #e1bee7;
        color: #7b1fa2;
    }
    .title-text.observation .title-text-content {
        background-color: #f3e5f5;
    }
    /* Summary and solution-specific styling - soft orange pastel, no overlay */
    .title-text.summary {
        background-color: #fff3e0;
        border-color: #ffcc02;
    }
    .title-text.summary .title-text-header {
        background-color: #ffcc02;
        color: #f57c00;
    }
    .title-text.summary .title-text-content {
        background-color: #fff3e0;
    }
    .title-text-content ul {
        background-color: transparent;
        border: none;
        border-radius: 0;
        padding: 0;
        margin: 0;
        color: #333;
        font-size: 8pt;
        line-height: 1.4;
    }
    .title-text-content li {
        margin: 3px 0;
        color: #333;
    }
    .title-text-content li strong {
        color: #495057;
        font-weight: normal;
        font-size: 8pt;
        font-style: italic;
    }
    .title-text-content li code {
        background-color: #e9ecef;
        color: #333;
        padding: 1px 3px;
        border-radius: 2px;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 7pt;
    }
    .title-text-content pre {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 3px;
        padding: 10px;
        margin: 0;
        font-size: 8pt;
        line-height: 1.4;
        overflow-x: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    .title-text-content code {
        background-color: transparent;
        padding: 0;
        border-radius: 0;
        font-size: 8pt;
        color: #2c3e50;
    }
    /* Parsing error display styling */
    .parsing-error-box {
        background-color: #ffebee;
        border: 1px solid #f44336;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 8px 0;
        font-size: 9pt;
        color: #c62828;
        box-shadow: 0 2px 4px rgba(244, 67, 54, 0.1);
    }
    .parsing-error-header {
        font-weight: bold;
        margin-bottom: 4px;
        color: #d32f2f;
    }
    .parsing-error-content {
        font-family: 'Courier New', monospace;
        background-color: #ffcdd2;
        padding: 4px 6px;
        border-radius: 2px;
        margin-top: 4px;
        font-size: 8pt;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    """
