"""Tool, data, and software management for the STCoscientist agent."""

import inspect
import os

import pandas as pd

from spatialomicsgym.mcp_config_path import CANONICAL_CONFIG_DEFAULT
from spatialomicsgym.mcp_user_config import DEFAULT_USER_CONFIG
from spatialomicsgym.utils import function_to_api_schema


def _rebuild_registry_docs(tool_registry):
    """Build the ``[docid, tool]`` rows for a registry's ``document_df``.

    Iterate ``tool_registry.tools`` directly and use the row position as the docid, so
    every row's ``document_content`` is a real tool. The previous
    ``range(len(tools))`` + ``get_tool_by_id(i)`` assumed tool ids stay a contiguous
    ``0..N-1`` range — but ids are assigned monotonically and never renumbered, so after
    any ``remove_tool_by_name`` a positional index no longer matches a tool id:
    ``get_tool_by_id`` then returns ``None`` for the gaps (which crashes the retrieval
    corpus builder's ``doc.get(...)``) and silently drops the highest-id tools.
    """
    return [[int(i), tool] for i, tool in enumerate(tool_registry.tools)]


def add_tool(agent, api):
    """Add a new tool to the agent's tool registry and make it available for retrieval.

    Args:
        agent: The STCoscientist agent instance
        api: A callable function to be added as a tool

    """
    try:
        # Get function information
        function_code = inspect.getsource(api)
        module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
        function_name = api.__name__ if hasattr(api, "__name__") else str(api)

        # Generate API schema using the existing utility function
        schema = function_to_api_schema(function_code, agent.llm)

        # Ensure the schema has all required fields for the tool registry
        if not isinstance(schema, dict):
            raise ValueError("Generated schema is not a dictionary")

        # Validate and enhance the schema

        # Set default values if missing
        if "name" not in schema:
            schema["name"] = function_name
        if "description" not in schema:
            schema["description"] = f"Custom tool: {function_name}"
        if "required_parameters" not in schema:
            # Try to extract from parameters if available
            if "parameters" in schema and isinstance(schema["parameters"], dict):
                required_params = []
                params = schema["parameters"]
                if "properties" in params:
                    for param_name in params["properties"]:
                        if param_name in params.get("required", []):
                            required_params.append(param_name)
                schema["required_parameters"] = required_params
            else:
                schema["required_parameters"] = []

        # Add module information to the schema
        schema["module"] = module_name

        # Add the tool to the tool registry if it exists
        if hasattr(agent, "tool_registry") and agent.tool_registry is not None:
            try:
                agent.tool_registry.register_tool(schema)
                print(f"Successfully registered tool '{schema['name']}' in tool registry")
            except Exception as e:
                print(f"Warning: Failed to register tool in registry: {e}")
                # Continue with adding to module2api even if registry fails

        # Add the tool to module2api structure for system prompt generation
        if not hasattr(agent, "module2api") or agent.module2api is None:
            agent.module2api = {}

        if module_name not in agent.module2api:
            agent.module2api[module_name] = []

        # Check if tool already exists in module2api to avoid duplicates
        existing_tool = None
        for existing in agent.module2api[module_name]:
            if existing.get("name") == schema["name"]:
                existing_tool = existing
                break

        if existing_tool:
            # Update existing tool
            existing_tool.update(schema)
            print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
        else:
            # Add new tool
            agent.module2api[module_name].append(schema)
            print(f"Added new tool '{schema['name']}' to module '{module_name}'")

        # Update the tool registry's document dataframe if it exists
        if hasattr(agent, "tool_registry") and agent.tool_registry is not None:
            try:
                # Rebuild the document dataframe
                docs = _rebuild_registry_docs(agent.tool_registry)
                agent.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
            except Exception as e:
                print(f"Warning: Failed to update tool registry document dataframe: {e}")

        # Store the original function for potential future use
        if not hasattr(agent, "_custom_functions"):
            agent._custom_functions = {}
        agent._custom_functions[schema["name"]] = api

        # Also store in _custom_tools for highlighting
        if not hasattr(agent, "_custom_tools"):
            agent._custom_tools = {}
        agent._custom_tools[schema["name"]] = {
            "name": schema["name"],
            "description": schema["description"],
            "module": module_name,
        }

        # Make the function available in the global namespace for execution
        import builtins

        if not hasattr(builtins, "_spatialomicsgym_custom_functions"):
            builtins._spatialomicsgym_custom_functions = {}
        builtins._spatialomicsgym_custom_functions[schema["name"]] = api

        print(f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval")
        agent.configure()
        return schema

    except Exception as e:
        print(f"Error adding tool: {e}")
        import traceback

        traceback.print_exc()
        raise


def get_custom_tool(agent, name):
    """Get a custom tool by name.

    Args:
        agent: The STCoscientist agent instance
        name: The name of the custom tool

    Returns:
        The custom tool function if found, None otherwise

    """
    if hasattr(agent, "_custom_functions") and name in agent._custom_functions:
        return agent._custom_functions[name]
    return None


def list_custom_tools(agent):
    """List all custom tools that have been added.

    Args:
        agent: The STCoscientist agent instance

    Returns:
        A list of custom tool names

    """
    if hasattr(agent, "_custom_functions"):
        return list(agent._custom_functions.keys())
    return []


def remove_custom_tool(agent, name):
    """Remove a custom tool.

    Args:
        agent: The STCoscientist agent instance
        name: The name of the custom tool to remove

    Returns:
        True if the tool was removed, False if it wasn't found

    """
    removed = False
    wrapper = (getattr(agent, "_custom_functions", None) or {}).get(name)

    # Remove from custom functions
    if hasattr(agent, "_custom_functions") and name in agent._custom_functions:
        del agent._custom_functions[name]
        removed = True

    # Remove from custom tools (for highlighting)
    if hasattr(agent, "_custom_tools") and name in agent._custom_tools:
        del agent._custom_tools[name]
        removed = True

    # Remove from global namespace
    import builtins

    if hasattr(builtins, "_spatialomicsgym_custom_functions") and name in builtins._spatialomicsgym_custom_functions:
        del builtins._spatialomicsgym_custom_functions[name]

    # And from the REPL's own namespace, the fourth place the wrapper was written and, until the
    # ``mcp_servers`` module below was found, thought the only one this function missed. ``tool_conversion`` binds each custom tool there
    # too, so clearing the three catalogs above left the tool callable from ``<execute>`` for the
    # life of the process: the model could invoke a wrapper every catalog said did not exist,
    # while ``resync_user_tools``' docstring promised "a trashed tool stops being callable".
    try:
        from spatialomicsgym.tool.support_tools import forget_repl_name

        if forget_repl_name(name):
            removed = True
    except Exception:  # a REPL that cannot be reached must not fail the removal
        pass

    # And from the ``mcp_servers.<server>`` module ``add_mcp`` set it on. The in-process REPL (the CLI,
    # scored runs) shares this process's ``sys.modules``, so after the removals above
    # ``from mcp_servers.<server> import <tool>`` still handed back the removed wrapper -- and bound its
    # name into the cell again, undoing the REPL-namespace removal. Only the very wrapper being removed
    # is taken off, so a same-named tool of another server, or a user module's own alias, stays.
    if wrapper is not None:
        import sys

        for module_name, module in list(sys.modules.items()):
            if isinstance(module_name, str) and module_name.startswith("mcp_servers."):
                if getattr(module, name, None) is wrapper:
                    delattr(module, name)

    # Remove from tool registry
    if hasattr(agent, "tool_registry") and agent.tool_registry is not None:
        if agent.tool_registry.remove_tool_by_name(name):
            removed = True
            # Rebuild the document dataframe
            try:
                docs = _rebuild_registry_docs(agent.tool_registry)
                agent.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
            except Exception as e:
                print(f"Warning: Failed to update tool registry document dataframe: {e}")

    # Remove from module2api
    if hasattr(agent, "module2api"):
        for tools in agent.module2api.values():
            for i, tool in enumerate(tools):
                if tool.get("name") == name:
                    del tools[i]
                    removed = True
                    break

    if removed:
        print(f"Custom tool '{name}' has been removed")
        # Symmetry with add_tool (which calls agent.configure()): after dropping the tool from the
        # registries/module2api, refresh the agent config so the removed tool no longer appears in the
        # system prompt / retrieval corpus. Without this the tool lingers in the prompt until the next
        # unrelated reconfigure, so the model can still try to call a function that no longer exists.
        try:
            agent.configure()
        except Exception as e:  # never let a reconfigure failure mask a successful removal
            print(f"Warning: agent.configure() after removal failed: {e}")
    else:
        print(f"Custom tool '{name}' was not found")

    return removed


def _shipped_identity(config_path):
    """The shipped catalogue's server names and callable names, for the clash checks below.

    Empty on any failure, and deliberately so: an empty identity means "no name is taken", which
    makes the predicate skip FEWER user servers, which prunes FEWER live tools. Of the two ways to
    be wrong about a base config we could not read, leaving a tool registered is recoverable and
    unregistering a live one is what this whole function exists to stop.
    """
    from spatialomicsgym.agent.mcp_config_merger import shipped_identity

    try:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        servers = data.get("mcp_servers") if isinstance(data, dict) else None
        return shipped_identity(servers if isinstance(servers, dict) else {})
    except Exception:
        return (set(), set())


def _current_user_server_modules(user_path=DEFAULT_USER_CONFIG, config_path=CANONICAL_CONFIG_DEFAULT):
    """Module names (``mcp_servers.<name>``) of the user servers the merge would actually SERVE.

    "Would actually serve" and not "is enabled": the two are not the same question, and this
    function used to answer the second one. ``build_merged_mcp_config`` also drops a user server
    whose name collides with a shipped server, whose ``spatialomicsgym_name`` collides with a
    shipped callable, or whose meta is not a mapping -- so a tool the merge refuses read as live
    here, and the pruner left it registered in the agent's catalogs: a name the model can pick with
    nothing behind it. Both sides now go through ``user_server_skip_reason``, so there is one
    definition of live and they cannot drift again.

    Returns ``None`` when the file cannot be read *as a server mapping*, so the caller SKIPS pruning
    rather than dropping live tools on a transient read error. That covers both ways of failing to
    read it: YAML that will not parse, and YAML that parses into the wrong shape. An empty set is
    not the same answer -- it says "no user server is live", which is the instruction to unregister
    every user tool -- so a half-written or hand-corrupted file must not produce one.

    An absent or empty file does still yield an empty set. Absence is not ambiguity: it is the
    documented "nothing is registered" (all user tools are stale -> prunable).

    The path is resolved via ``resolve_user_config_path`` rather than trusted as CWD-relative: an
    absent file means "prune every user tool", so getting the location wrong would not merely fail
    to find the tools, it would actively unregister the live ones.
    """
    from spatialomicsgym.agent.mcp_config_merger import (
        _recover_top_level_servers,
        resolve_user_config_path,
        user_function_names,
        user_server_skip_reason,
    )

    user_path = resolve_user_config_path(user_path)
    if not os.path.exists(user_path):
        return set()
    try:
        import yaml

        with open(user_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return None
    if not isinstance(data, dict):
        print(f"Warning: {user_path} is a {type(data).__name__}, not a mapping - leaving user tools registered")
        return None
    servers = data.get("mcp_servers", {})
    if not isinstance(servers, dict):
        print(
            f"Warning: mcp_servers in {user_path} is a {type(servers).__name__}, not a mapping"
            " - leaving user tools registered"
        )
        return None
    # The merge's own net for a block the agent wrote beside `mcp_servers` rather than under it.
    # `resync_user_tools` repairs the file before calling here, so this only matters when the
    # repair could not be written -- and in that case the merge still wires the block, so it is
    # still live and must not be pruned.
    servers = {**_recover_top_level_servers(data, servers), **servers}
    original_names, original_function_names = _shipped_identity(config_path)
    out = set()
    # Accumulated in the same order as the merge's own loop. A user-vs-user name clash makes the
    # merge skip the second server, so the pruner has to reach the same verdict -- counting it
    # live here is exactly the desync this shared predicate was written to end.
    claimed_by_user: set[str] = set()
    for name, meta in servers.items():
        if user_server_skip_reason(name, meta, original_names, original_function_names, claimed_by_user) is None:
            out.add(f"mcp_servers.{name}")
            claimed_by_user |= user_function_names(meta)
    return out


def resync_user_tools(agent, config_path=CANONICAL_CONFIG_DEFAULT, user_path=DEFAULT_USER_CONFIG):
    """Re-sync the agent's live tool catalogs with the on-disk ``mcp_config_user.yaml``.

    User-tool create/delete/modify (trash_manager / knowledge_manager / the creation playbook) write
    files + config but NEVER touch in-memory agent state, so without this a trashed tool stays
    callable, a modified tool keeps running its OLD wrapper, and a just-created tool isn't callable
    until an agent restart. This:

      1. repairs ``mcp_config_user.yaml`` (``normalize_user_config``) BEFORE anything reads it;
      2. drops every currently-registered USER tool (module ``mcp_servers.user_*``) that the merge
         would no longer serve -- via ``remove_custom_tool`` so it clears module2api, tool_registry
         (+ document_df), ``_custom_functions`` and the REPL builtins;
      3. re-runs ``add_mcp`` so current user tools are (re)loaded and a modified tool's wrapper is
         replaced in place (register_tool / module2api dedup update-in-place by name).

    Step 1 used to be inside step 3: ``add_mcp`` is what invokes ``normalize_user_config``, so on
    the pass right after the agent wrote a server block at column 0 the pruner read "no user
    servers are live" and unregistered every one of them, and ``add_mcp`` then repaired the file
    and put them back. Transient rather than permanent, but during that window the tools are gone
    from the catalogs the prompt is built from -- and ``remove_custom_tool`` calls
    ``agent.configure()`` per tool, so the model really can be told its own tool no longer exists.
    Repairing first costs one no-op read on a correct file and removes the window entirely.

    Returns the list of pruned tool names. Never raises.
    """
    pruned = []
    try:
        from spatialomicsgym.agent.mcp_config_merger import normalize_user_config, resolve_user_config_path

        normalize_user_config(resolve_user_config_path(user_path))
    except Exception as e:
        # A repair that cannot run is not a reason to skip the resync: the recovery net inside
        # `_current_user_server_modules` and the merge still see the stray block.
        print(f"Warning: resync could not normalize {user_path}: {e}")
    current = _current_user_server_modules(user_path, config_path)
    if current is not None:
        module2api = getattr(agent, "module2api", {}) or {}
        for module_name, tools in list(module2api.items()):
            if not (isinstance(module_name, str) and module_name.startswith("mcp_servers.user_")):
                continue
            if module_name in current:
                continue  # still enabled -> add_mcp refreshes it in place (picks up a modification)
            for tool in list(tools):
                name = tool.get("name") if isinstance(tool, dict) else None
                if name:
                    try:
                        remove_custom_tool(agent, name)
                        pruned.append(name)
                    except Exception as e:
                        print(f"Warning: resync could not remove stale user tool '{name}': {e}")
    try:
        # With the merge_user the agent was wired with, when its caller chose one. Called bare, add_mcp
        # fell back to tool_creation_enabled -- on, whenever this resync runs -- and merged back every
        # user tool a merge_user=False caller had kept out (hunt 2026-09-30, uL6-parity-16).
        chosen = getattr(agent, "_mcp_merge_user", None)
        if chosen is None:
            agent.add_mcp(config_path)
        else:
            agent.add_mcp(config_path, merge_user=chosen)
    except Exception as e:
        print(f"Warning: resync_user_tools add_mcp({config_path}) failed: {e}")
    return pruned


def add_data(agent, data):
    """Add new data to the data lake.

    Args:
        agent: The STCoscientist agent instance
        data: Dictionary with file path as key and description as value
              e.g., {'my_dataset.csv': 'A dataset containing gene expression data'}
              or {'path/to/file.txt': 'Description of the file'}

    """
    try:
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary with file path as key and description as value")

        # Initialize custom data storage if it doesn't exist
        if not hasattr(agent, "_custom_data"):
            agent._custom_data = {}

        # Add each data item. The file stays where it is: the prompt lists it under CUSTOM DATA by
        # name AND path. Only the basename used to reach the prompt, under a data lake that does not
        # hold the file, while this said "added to the data lake" -- so the agent searched the disk
        # for a file the caller had already located (hunt 2026-09-30, uL4-honesty-4).
        added = 0
        for file_path, description in data.items():
            if not isinstance(file_path, str) or not isinstance(description, str):
                print("Warning: Skipping invalid data entry - file_path and description must be strings")
                continue

            # Extract filename from path for storage
            filename = os.path.basename(file_path) if "/" in file_path else file_path
            # Absolute when it exists, so a later chdir cannot move it; otherwise exactly as given.
            expanded = os.path.expanduser(file_path)
            location = os.path.abspath(expanded) if os.path.exists(expanded) else file_path
            held = agent._custom_data.get(filename)
            if held is not None and held.get("path") != location:
                # Two files with one basename silently replaced each other under the same key.
                print(
                    f"Warning: Skipping '{file_path}': a different file named '{filename}' was already "
                    f"added from '{held.get('path')}'. Remove it first with remove_custom_data('{filename}')."
                )
                continue

            # Store the data with both the full path and description
            agent._custom_data[filename] = {
                "path": location,
                "description": description,
            }

            # Also add to the data_lake_dict for consistency
            agent.data_lake_dict[filename] = description

            print(f"Added data item '{filename}' at {location}: {description}")
            added += 1
        agent.configure()
        print(f"Added {added} data item(s); each stays at its own path, listed in the prompt under CUSTOM DATA")
        return True

    except Exception as e:
        print(f"Error adding data: {e}")
        import traceback

        traceback.print_exc()
        return False


def get_custom_data(agent, name):
    """Get a custom data item by name.

    Args:
        agent: The STCoscientist agent instance
        name: The name of the custom data item

    Returns:
        The custom data item info if found, None otherwise

    """
    if hasattr(agent, "_custom_data") and name in agent._custom_data:
        return agent._custom_data[name]
    return None


def list_custom_data(agent):
    """List all custom data items that have been added.

    Args:
        agent: The STCoscientist agent instance

    Returns:
        A list of custom data item names and descriptions

    """
    if hasattr(agent, "_custom_data"):
        return [(name, info["description"]) for name, info in agent._custom_data.items()]
    return []


def remove_custom_data(agent, name):
    """Remove a custom data item.

    Args:
        agent: The STCoscientist agent instance
        name: The name of the custom data item to remove

    Returns:
        True if the data item was removed, False if it wasn't found

    """
    removed = False

    # Remove from custom data
    if hasattr(agent, "_custom_data") and name in agent._custom_data:
        del agent._custom_data[name]
        removed = True

    # Remove from data_lake_dict
    if hasattr(agent, "data_lake_dict") and name in agent.data_lake_dict:
        del agent.data_lake_dict[name]
        removed = True

    if removed:
        print(f"Custom data item '{name}' has been removed")
        # Symmetry with add_data (which reconfigures): refresh so the removed dataset drops out of the
        # data-lake description in the system prompt.
        try:
            agent.configure()
        except Exception as e:
            print(f"Warning: agent.configure() after removal failed: {e}")
    else:
        print(f"Custom data item '{name}' was not found")

    return removed


def add_software(agent, software):
    """Add new software to the software library.

    Args:
        agent: The STCoscientist agent instance
        software: Dictionary with software name as key and description as value
                 e.g., {'custom_tool': 'A custom analysis tool for processing data'}
                 or {'my_package': 'Description of the package functionality'}

    """
    try:
        if not isinstance(software, dict):
            raise ValueError("Software must be a dictionary with software name as key and description as value")

        # Initialize custom software storage if it doesn't exist
        if not hasattr(agent, "_custom_software"):
            agent._custom_software = {}

        # Add each software item
        for software_name, description in software.items():
            if not isinstance(software_name, str) or not isinstance(description, str):
                print("Warning: Skipping invalid software entry - software_name and description must be strings")
                continue

            # Store the software with description
            agent._custom_software[software_name] = {
                "name": software_name,
                "description": description,
            }

            # Also add to the library_content_dict for consistency
            agent.library_content_dict[software_name] = description

            print(f"Added software '{software_name}': {description}")

        print(f"Successfully added {len(software)} software item(s) to the library")
        agent.configure()
        return True

    except Exception as e:
        print(f"Error adding software: {e}")
        import traceback

        traceback.print_exc()
        return False


def get_custom_software(agent, name):
    """Get a custom software item by name.

    Args:
        agent: The STCoscientist agent instance
        name: The name of the custom software item

    Returns:
        The custom software item info if found, None otherwise

    """
    if hasattr(agent, "_custom_software") and name in agent._custom_software:
        return agent._custom_software[name]
    return None


def list_custom_software(agent):
    """List all custom software items that have been added.

    Args:
        agent: The STCoscientist agent instance

    Returns:
        A list of custom software item names and descriptions

    """
    if hasattr(agent, "_custom_software"):
        return [(name, info["description"]) for name, info in agent._custom_software.items()]
    return []


def remove_custom_software(agent, name):
    """Remove a custom software item.

    Args:
        agent: The STCoscientist agent instance
        name: The name of the custom software item to remove

    Returns:
        True if the software item was removed, False if it wasn't found

    """
    removed = False

    # Remove from custom software
    if hasattr(agent, "_custom_software") and name in agent._custom_software:
        del agent._custom_software[name]
        removed = True

    # Remove from library_content_dict
    if hasattr(agent, "library_content_dict") and name in agent.library_content_dict:
        del agent.library_content_dict[name]
        removed = True

    if removed:
        print(f"Custom software item '{name}' has been removed")
        # Symmetry with add_software (which reconfigures): refresh so the removed library drops out of
        # the software-library description in the system prompt.
        try:
            agent.configure()
        except Exception as e:
            print(f"Warning: agent.configure() after removal failed: {e}")
    else:
        print(f"Custom software item '{name}' was not found")

    return removed


def filter_know_how_for_commercial_mode(agent):
    """Filter out know-how documents that don't allow commercial use.

    This method removes documents from the know-how loader that have
    commercial use restrictions when the agent is in commercial mode.

    Args:
        agent: The STCoscientist agent instance
    """
    docs_to_remove = []

    for doc_id, doc in agent.know_how_loader.documents.items():
        if _forbids_commercial_use(doc):
            docs_to_remove.append(doc_id)

    # Remove documents that don't allow commercial use
    for doc_id in docs_to_remove:
        doc_name = agent.know_how_loader.documents[doc_id]["name"]
        agent.know_how_loader.remove_document(doc_id)
        print(f"  ⚠️  Excluded know-how '{doc_name}' (non-commercial license)")

    # Tier 2, the merged packs, when the loader holds any. getattr-guarded: the stub loaders in the
    # test suite and any loader built before tier 2 existed have no pack_documents, and that must
    # read as "no packs", not as an AttributeError at agent start-up.
    pack_documents = getattr(agent.know_how_loader, "pack_documents", None) or {}
    packs_to_remove = [doc_id for doc_id, doc in pack_documents.items() if _forbids_commercial_use(doc)]
    for doc_id in packs_to_remove:
        doc_name = pack_documents[doc_id]["name"]
        agent.know_how_loader.remove_pack_document(doc_id)
        print(f"  ⚠️  Excluded know-how pack '{doc_name}' (non-commercial license)")


def _forbids_commercial_use(doc: dict) -> bool:
    """The three strings a document's ``Commercial Use`` metadata line may carry to opt out."""
    commercial_use = (doc.get("metadata") or {}).get("commercial_use", "")
    return "❌" in commercial_use or "Not Allowed" in commercial_use or "Non-Commercial" in commercial_use
