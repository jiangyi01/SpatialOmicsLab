# Re-export all public names for backward compatibility.
# Existing code using `from spatialomicsgym.utils import X` will continue to work.

import importlib
from typing import Any

# --------------------------------------------------------------------------- #
# LAZY, deliberately. This package used to import its six submodules at package import, and two
# of them pull the heavy stack the moment they load: ``logging_utils`` (langchain_core) and
# ``tool_conversion`` (pandas). Any caller that wanted ONE light helper -- the web portal's
# ``/api/config`` asking ``tool_discovery.read_module2api`` how many tools ship -- paid for all
# of it, on a settings route the portal must answer with no key configured and no agent built
# (``test/test_webui_stays_out_of_the_heavy_stack.py``; found red in the 2026-09-20 gate).
#
# PEP 562: every public name is still importable from here (``from spatialomicsgym.utils import
# pretty_print``), and resolves to its submodule the first time it is asked for. ``__all__`` is
# unchanged. Nothing is imported until a name is used.
# --------------------------------------------------------------------------- #
_LAZY: dict[str, str] = {
    "CustomBaseModel": "tool_conversion",
    "ID": "data_processing",
    "NodeLogger": "logging_utils",
    "PromptLogger": "logging_utils",
    "_TEXT_COLOR_MAPPING": "logging_utils",
    "_get_gene_id_ensembl": "data_processing",
    "_get_gene_id_ensembl_with_version": "data_processing",
    "_get_gene_id_entrez": "data_processing",
    "api_schema": "tool_conversion",
    "api_schema_to_langchain_tool": "tool_conversion",
    "check_and_download_s3_files": "file_io",
    "check_or_create_path": "file_io",
    "clean_code_content": "formatting",
    "clean_message_content": "formatting",
    "color_print": "logging_utils",
    "configured_benchmark_mirror": "file_io",
    "convert_markdown_to_pdf": "formatting",
    "create_parsing_error_html": "formatting",
    "create_tool_call_block": "formatting",
    "detect_code_language_and_tool": "formatting",
    "download_and_unzip": "file_io",
    "execute_graphql_query": "tool_conversion",
    "find_best_module_match": "tool_discovery",
    "find_matching_execution": "formatting",
    "format_default_tool_name": "formatting",
    "format_detected_tools": "formatting",
    "format_execute_tags_in_content": "formatting",
    "format_lists_in_text": "formatting",
    "format_observation_as_terminal": "formatting",
    "format_single_list": "formatting",
    "format_solution_tags_in_content": "formatting",
    "function_to_api_schema": "tool_conversion",
    "get_all_functions_from_file": "tool_conversion",
    "get_gene_id": "data_processing",
    "get_pdf_css_content": "formatting",
    "get_tool_decorated_functions": "tool_conversion",
    "has_execution_results": "formatting",
    "identify_list_blocks": "formatting",
    "inject_custom_functions_to_repl": "tool_conversion",
    "load_pickle": "file_io",
    "load_pkl": "file_io",
    "parse_hpo_obo": "data_processing",
    "parse_tool_calls_from_code": "tool_discovery",
    "parse_tool_calls_with_modules": "tool_discovery",
    "pretty_print": "logging_utils",
    "process_bio_retrieval_ducoment": "data_processing",
    "process_observation_with_images": "formatting",
    "read_module2api": "tool_discovery",
    "remove_emojis_from_text": "formatting",
    "run_bash_script": "execution",
    "run_cli_command": "execution",
    "run_r_code": "execution",
    "run_with_timeout": "execution",
    "safe_execute_decorator": "tool_conversion",
    "save_pkl": "file_io",
    "should_skip_message": "formatting",
    "textify_api_dict": "formatting",
    "write_python_code": "tool_conversion",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"spatialomicsgym.utils.{submodule}")
    value = getattr(module, name)
    globals()[name] = value  # resolved once; the next access is a plain attribute
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    # execution
    "run_r_code",
    "run_bash_script",
    "run_cli_command",
    "run_with_timeout",
    # tool_conversion
    "api_schema",
    "function_to_api_schema",
    "get_all_functions_from_file",
    "write_python_code",
    "execute_graphql_query",
    "get_tool_decorated_functions",
    "CustomBaseModel",
    "safe_execute_decorator",
    "api_schema_to_langchain_tool",
    "inject_custom_functions_to_repl",
    # tool_discovery
    "read_module2api",
    "parse_tool_calls_from_code",
    "parse_tool_calls_with_modules",
    "find_best_module_match",
    # data_processing
    "process_bio_retrieval_ducoment",
    "parse_hpo_obo",
    "ID",
    "get_gene_id",
    "_get_gene_id_entrez",
    "_get_gene_id_ensembl",
    "_get_gene_id_ensembl_with_version",
    # file_io
    "load_pickle",
    "save_pkl",
    "load_pkl",
    "download_and_unzip",
    "check_and_download_s3_files",
    "configured_benchmark_mirror",
    "check_or_create_path",
    # logging
    "_TEXT_COLOR_MAPPING",
    "color_print",
    "PromptLogger",
    "NodeLogger",
    "pretty_print",
    # formatting
    "textify_api_dict",
    "clean_message_content",
    "should_skip_message",
    "has_execution_results",
    "find_matching_execution",
    "create_parsing_error_html",
    "format_execute_tags_in_content",
    "detect_code_language_and_tool",
    "clean_code_content",
    "create_tool_call_block",
    "format_detected_tools",
    "format_default_tool_name",
    "format_solution_tags_in_content",
    "format_observation_as_terminal",
    "process_observation_with_images",
    "remove_emojis_from_text",
    "format_lists_in_text",
    "identify_list_blocks",
    "format_single_list",
    "convert_markdown_to_pdf",
    "get_pdf_css_content",
]
