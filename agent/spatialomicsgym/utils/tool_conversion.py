import ast
import importlib
from typing import Any, ClassVar

import pandas as pd
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, ValidationError


class api_schema(BaseModel):
    """api schema specification."""

    api_schema: str | None = Field(description="The api schema as a dictionary")


def function_to_api_schema(function_string, llm):
    prompt = """
    Based on a code snippet and help me write an API docstring in the format like this:

    {{'name': 'get_gene_set_enrichment',
    'description': 'Given a list of genes, identify a pathway that is enriched for this gene set. Return a list of pathway name, p-value, z-scores.',
    'required_parameters': [{{'name': 'genes',
    'type': 'List[str]',
    'description': 'List of g`ene symbols to analyze',
    'default': None}}],
    'optional_parameters': [{{'name': 'top_k',
    'type': 'int',
    'description': 'Top K pathways to return',
    'default': 10}},  {{'name': 'database',
    'type': 'str',
    'description': 'Name of the database to use for enrichment analysis',
    'default': "gene_ontology"}}]}}

    Strictly follow the input from the function - don't create fake optional parameters.
    For variable without default values, set them as None, not null.
    For variable with boolean values, use capitalized True or False, not true or false.
    Do not add any return type in the docstring.
    Be as clear and succint as possible for the descriptions. Please do not make it overly verbose.
    Here is the code snippet:
    {code}
    """
    llm = llm.with_structured_output(api_schema)

    api = None
    for _ in range(7):
        try:
            result = llm.invoke(prompt.format(code=function_string))
            api = (result.model_dump() if hasattr(result, "model_dump") else result.dict())["api_schema"]
            return ast.literal_eval(api)  # -> prefer "default": None
            # return json.loads(api) # -> prefer "default": null
        except Exception as e:
            print("API string:", api)
            print("Error parsing the API string:", e)
            continue

    return "Error: Could not parse the API schema"
    # return


def get_all_functions_from_file(file_path):
    with open(file_path, encoding="utf-8") as file:
        file_content = file.read()

    # Parse the file content into an AST (Abstract Syntax Tree)
    tree = ast.parse(file_content)

    # List to hold the top-level functions as strings
    functions = []

    # Walk through the AST nodes
    for node in tree.body:  # Only consider top-level nodes in the body
        if isinstance(node, ast.FunctionDef):  # Check if the node is a function definition
            # Skip if function name starts with underscore
            if node.name.startswith("_"):
                continue

            start_line = node.lineno - 1  # Get the starting line of the function
            end_line = node.end_lineno  # Get the ending line of the function (only available in Python 3.8+)
            func_code = file_content.splitlines()[start_line:end_line]
            functions.append("\n".join(func_code))  # Join lines of the function and add to the list

    return functions


def write_python_code(request: str):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from spatialomicsgym.config import default_config
    from spatialomicsgym.llm import get_llm

    # The configured model, through the factory every other call uses. The pinned
    # claude-3-5-sonnet-20240620 is retired, so this always 404'd (u16-llm-config-19).
    model = get_llm(config=default_config)
    template = """Write some python code to solve the user's problem.

    Return only python code in Markdown format, e.g.:

    ```python
    ....
    ```"""
    prompt = ChatPromptTemplate.from_messages([("system", template), ("human", "{input}")])

    def _sanitize_output(text: str):
        _, after = text.split("```python")
        return after.split("```")[0]

    chain = prompt | model | StrOutputParser() | _sanitize_output
    return chain.invoke({"input": "write a code that " + request})


def execute_graphql_query(
    query: str,
    variables: dict,
    api_address: str = "https://api.genetics.opentargets.org/graphql",
) -> dict:
    """Executes a GraphQL query with variables and returns the data as a dictionary."""
    import requests

    headers = {"Content-Type": "application/json"}
    response = requests.post(api_address, json={"query": query, "variables": variables}, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(response.text)
        response.raise_for_status()


def get_tool_decorated_functions(relative_path):
    import ast
    import importlib.util
    import os

    # Get the directory of the current file (__init__.py)
    # Use the spatialomicsgym package directory as the base
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Construct the absolute path from the relative path
    file_path = os.path.join(current_dir, relative_path)

    with open(file_path, encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)

    tool_function_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Name)
                    and decorator.id == "tool"
                    or (
                        isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "tool"
                    )
                ):
                    tool_function_names.append(node.name)

    # Calculate the module name from the relative path
    package_path = os.path.relpath(file_path, start=current_dir)
    module_name = package_path.replace(os.path.sep, ".").rsplit(".", 1)[0]

    # Import the module and get the function objects
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    tool_functions = [getattr(module, name) for name in tool_function_names]

    return tool_functions


class CustomBaseModel(BaseModel):
    api_schema: ClassVar[dict] = None  # Class variable to store api_schema

    # Add model_config with arbitrary_types_allowed=True
    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def set_api_schema(cls, schema: dict):
        cls.api_schema = schema

    @classmethod
    def model_validate(cls, obj):
        try:
            return super().model_validate(obj)
        except (ValidationError, AttributeError) as e:
            if not cls.api_schema:
                raise e  # If no api_schema is set, raise original error

            error_msg = "Required Parameters:\n"
            for param in cls.api_schema["required_parameters"]:
                error_msg += f"- {param['name']} ({param['type']}): {param['description']}\n"

            error_msg += "\nErrors:\n"
            for err in e.errors():
                field = err["loc"][0] if err["loc"] else "input"
                error_msg += f"- {field}: {err['msg']}\n"

            if not obj:
                error_msg += "\nNo input provided"
            else:
                error_msg += "\nProvided Input:\n"
                for key, value in obj.items():
                    error_msg += f"- {key}: {value}\n"

                missing_params = {param["name"] for param in cls.api_schema["required_parameters"]} - set(obj.keys())
                if missing_params:
                    error_msg += "\nMissing Parameters:\n"
                    for param in missing_params:
                        error_msg += f"- {param}\n"

            # # Create proper validation error structure
            raise ValidationError.from_exception_data(
                title="Validation Error",
                line_errors=[
                    {
                        "type": "value_error",
                        "loc": ("input",),
                        "input": obj,
                        "ctx": {
                            "error": error_msg,
                        },
                    }
                ],
            ) from None


def safe_execute_decorator(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return str(e)

    return wrapper


def api_schema_to_langchain_tool(api_schema, mode="generated_tool", module_name=None):
    if not isinstance(api_schema, dict):
        # function_to_api_schema returns the plain string "Error: Could not parse the API schema" when
        # all LLM retries fail; a caller (react.add_tool) passing that straight in would otherwise
        # crash with "string indices must be integers".
        raise ValueError(f"api_schema must be a dict, got {type(api_schema).__name__}: {str(api_schema)[:120]!r}")
    if mode == "generated_tool":
        module = importlib.import_module("spatialomicsgym.tool.generated_tool." + api_schema["tool_name"] + ".api")
    elif mode == "custom_tool":
        module = importlib.import_module(module_name)
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'generated_tool' or 'custom_tool')")

    api_function = getattr(module, api_schema["name"])
    api_function = safe_execute_decorator(api_function)

    # Define a mapping from string type names to actual Python type objects
    type_mapping = {
        "string": str,
        "integer": int,
        "boolean": bool,
        "pandas": pd.DataFrame,  # Use the imported pandas.DataFrame directly
        "str": str,
        "int": int,
        "bool": bool,
        "float": float,
        "number": float,
        "list": list,
        "List": list,
        "dict": dict,
        "List[str]": list[str],
        "List[int]": list[int],
        "Dict": dict,
        "Any": Any,
    }

    # Create the fields and annotations. NEVER eval() the type string: ``param["type"]`` comes from an
    # LLM-generated or file-provided schema, so eval() is an arbitrary-code-execution AND crash surface
    # (None -> TypeError, "1/0" -> ZeroDivisionError). An unknown type name falls back to permissive
    # ``Any`` -- the real type checking is done by the tool function itself.
    annotations = {}
    fields = {}
    for param in api_schema.get("required_parameters") or []:
        if not isinstance(param, dict) or "name" not in param:
            continue  # skip a malformed parameter entry rather than crash the whole tool
        pname = param["name"]
        ptype = param.get("type")
        annotations[pname] = type_mapping.get(ptype, Any) if isinstance(ptype, str) else Any
        fields[pname] = Field(description=param.get("description", ""))

    # Create the ApiInput class dynamically
    ApiInput = type("Input", (CustomBaseModel,), {"__annotations__": annotations, **fields})
    # Set the api_schema
    ApiInput.set_api_schema(api_schema)

    # Create the StructuredTool
    api_tool = StructuredTool.from_function(
        func=api_function,
        name=api_schema["name"],
        description=api_schema["description"],
        args_schema=ApiInput,
        return_direct=True,
    )

    return api_tool


def inject_custom_functions_to_repl(custom_functions: dict):
    """Inject custom functions into the Python REPL execution environment.

    This function makes custom tools available during code execution by injecting
    them into both the persistent execution namespace and the builtins module.
    This allows the agent to call custom functions that users have added via
    agent.add_tool() when executing Python code in <execute> blocks.

    Args:
        custom_functions: Dictionary mapping function names to their callable objects

    Note:
        The function modifies both the persistent namespace used by run_python_repl
        and the builtins module to ensure maximum compatibility and accessibility
        of custom functions during code execution.
    """
    if custom_functions:
        # Access the persistent namespace used by run_python_repl
        from spatialomicsgym.tool.support_tools import _default_repl

        # Inject all custom functions into the execution namespace
        for name, func in custom_functions.items():
            _default_repl._namespace[name] = func

        # Also make them available in builtins for broader access
        import builtins

        if not hasattr(builtins, "_spatialomicsgym_custom_functions"):
            builtins._spatialomicsgym_custom_functions = {}
        builtins._spatialomicsgym_custom_functions.update(custom_functions)
