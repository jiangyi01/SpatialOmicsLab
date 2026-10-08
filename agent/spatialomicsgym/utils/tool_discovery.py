import importlib
import logging
from pathlib import Path


def read_module2api():
    """Auto-discover tool description modules instead of hardcoding."""
    tool_desc_dir = Path(__file__).parent.parent / "tool" / "tool_description"
    fields = []
    for f in sorted(tool_desc_dir.glob("*.py")):
        if f.name.startswith("_"):
            continue
        fields.append(f.stem)

    module2api = {}
    for field in fields:
        module_name = f"spatialomicsgym.tool.tool_description.{field}"
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "description"):
                module2api[f"spatialomicsgym.tool.{field}"] = module.description
        except Exception as e:
            # A broken tool_description module silently drops its tools from the agent's advertised
            # toolset; surface it so the omission is diagnosable instead of invisible.
            logging.getLogger(__name__).warning("read_module2api: skipping tool_description '%s': %s", field, e)
    return module2api


def parse_tool_calls_from_code(code: str, module2api: dict, custom_functions: dict = None) -> list[str]:
    """Parse code to detect imported tools by analyzing import statements.

    This function analyzes Python code to identify which tools/functions are being
    imported and used. It extracts tool names from import statements and function
    calls, then returns a deduplicated list of detected tool names.

    Args:
        code: The Python code string to analyze for tool imports
        module2api: Dictionary mapping module names to their available API tools
        custom_functions: Optional dictionary of custom functions that have been
                         added to the agent

    Returns:
        Sorted list of unique tool names detected in the code

    Example:
        >>> code = "from spatialomicsgym.tool import analyze_data\nimport pandas as pd"
        >>> parse_tool_calls_from_code(code, module2api)
        ['analyze_data', 'pandas']
    """
    tool_module_pairs = parse_tool_calls_with_modules(code, module2api, custom_functions)
    return sorted({pair[0] for pair in tool_module_pairs})


def parse_tool_calls_with_modules(code: str, module2api: dict, custom_functions: dict = None) -> list[tuple[str, str]]:
    """Parse code to detect imported tools and their associated modules.

    This function performs detailed analysis of Python code to identify which
    tools/functions are being imported and which modules they belong to. It
    handles various import patterns including direct imports, from-imports,
    and module.function patterns.

    Args:
        code: The Python code string to analyze for tool imports
        module2api: Dictionary mapping module names to their available API tools
        custom_functions: Optional dictionary of custom functions that have been
                         added to the agent

    Returns:
        List of tuples containing (tool_name, module_name) pairs for each
        detected tool and its associated module

    Note:
        The function uses regex patterns to match various import statement
        formats and also detects direct function calls without explicit imports.
    """
    import re

    detected_tools = set()

    # Get all available tools from module2api
    all_tools = {}
    for module_name, module_tools in module2api.items():
        for tool in module_tools:
            if isinstance(tool, dict) and "name" in tool:
                tool_name = tool["name"]
                if tool_name not in all_tools:
                    all_tools[tool_name] = []
                all_tools[tool_name].append(module_name)

    # Add custom tools
    if custom_functions:
        for tool_name in custom_functions.keys():
            if tool_name not in all_tools:
                all_tools[tool_name] = []
            all_tools[tool_name].append("custom_tools")

    # Look for import statements in the code
    import_patterns = [
        # The import list is space/tab/word/comma only. `\s` would match a newline, so the greedy
        # group swallowed the following source lines and glued the real names into one non-match.
        r"from\s+([\w.]+)\s+import\s+([ \t\w,]+)",  # from module import tool1, tool2
        r"import\s+([\w.]+)",  # import module
    ]

    for pattern in import_patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            # re.findall yields 2-tuples for the two-group "from ... import ..." pattern and bare
            # strings for the single-group "import ..." pattern. The old len(match) test misread a
            # 2-char module name ("os") as a from-import and dropped every longer import-module name.
            if isinstance(match, tuple):  # from module import tools
                module_name, tools_str = match
                # Split tools by comma and clean up
                tools = [tool.strip() for tool in tools_str.split(",")]

                for tool in tools:
                    # Check if this tool exists in any module
                    if tool in all_tools:
                        # Find the best matching module
                        best_module = find_best_module_match(module_name, all_tools[tool])
                        detected_tools.add((tool, best_module))
                    # Also check if it's a module.function pattern
                    elif "." in tool:
                        parts = tool.split(".")
                        if len(parts) == 2:
                            module_part, func_part = parts
                            if func_part in all_tools:
                                best_module = find_best_module_match(module_part, all_tools[func_part])
                                detected_tools.add((func_part, best_module))

            else:  # import module (single-group pattern -> re.findall yields a bare string)
                module_name = match
                # Check if any tools from this module are used
                for tool_name, modules in all_tools.items():
                    if any(module_name in mod for mod in modules):
                        # Look for usage of this tool in the code
                        if re.search(rf"\b{tool_name}\s*\(", code):
                            best_module = find_best_module_match(module_name, modules)
                            detected_tools.add((tool_name, best_module))

    # Also look for direct function calls without imports
    function_call_pattern = r"(\w+)\s*\("
    function_calls = re.findall(function_call_pattern, code)

    for func_call in function_calls:
        if func_call in all_tools:
            # For direct calls, use the first available module
            best_module = all_tools[func_call][0]
            detected_tools.add((func_call, best_module))

    return sorted(detected_tools)


def find_best_module_match(target_module: str, available_modules: list[str]) -> str:
    """Find the best matching module from a list of available modules.

    This function attempts to match a target module name against a list of
    available modules using various matching strategies: exact match, partial
    substring matches, and fallback to the first available module.

    Args:
        target_module: The module name we're trying to match
        available_modules: List of available module names to search through

    Returns:
        The best matching module name from the available modules list.
        Returns "unknown" if no modules are available.

    Note:
        The matching strategy prioritizes exact matches, then partial matches
        (where either the target is contained in the module name or vice versa),
        and finally falls back to the first available module.
    """
    # First try exact match
    if target_module in available_modules:
        return target_module

    # Try partial matches
    for module in available_modules:
        if target_module in module or module in target_module:
            return module

    # Return the first available module as fallback
    return available_modules[0] if available_modules else "unknown"
