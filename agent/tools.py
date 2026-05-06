"""Tool definitions and implementations for the coding agent.

Each tool has a JSON schema (for the Claude API) and a Python implementation.
All file operations are sandboxed to the repository root.
"""

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(os.environ.get("GITHUB_WORKSPACE", ".")).resolve()

MAX_FILE_SIZE = 100_000  # 100KB
MAX_SEARCH_RESULTS = 100
COMMAND_TIMEOUT = 30

EXCLUDED_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".next", "dist", "build"}

# --- Tool definitions (JSON schemas for Claude API) ---

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path relative to the repository root. "
            "Returns the file contents as a string. Returns an error if the file does not exist or is binary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root (e.g. 'server/main.py')",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files and directories in the given directory. "
            "Directories have a trailing '/'. Path is relative to the repo root. Use '.' for root."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path relative to repo root (e.g. '.', 'client/src')",
                }
            },
            "required": ["directory"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates the file and parent directories if needed. "
            "Overwrites existing content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the repo root",
                },
                "content": {
                    "type": "string",
                    "description": "The full content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Search for a pattern across repository files using grep. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex). E.g. 'def create_item', 'import.*fastapi'",
                },
                "file_pattern": {
                    "type": "string",
                    "description": "Optional glob to filter files (e.g. '*.py', '*.vue')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the repository root. "
            "Use for git history, linting, syntax checks, etc. Returns stdout and stderr."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run (e.g. 'git log --oneline -10')",
                }
            },
            "required": ["command"],
        },
    },
]


# --- Path safety ---

def _safe_path(relative: str) -> Path:
    """Resolve a relative path and ensure it stays within the repo root."""
    resolved = (REPO_ROOT / relative).resolve()
    if not str(resolved).startswith(str(REPO_ROOT)):
        raise ValueError(f"Path '{relative}' resolves outside the repository root.")
    return resolved


def _is_binary(path: Path) -> bool:
    """Heuristic: check first 1024 bytes for null bytes."""
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
        return b"\x00" in chunk
    except OSError:
        return False


# --- Tool implementations ---

def read_file(path: str) -> str:
    try:
        full = _safe_path(path)
        if not full.is_file():
            return f"Error: '{path}' is not a file or does not exist."
        if _is_binary(full):
            return f"Error: '{path}' appears to be a binary file."
        size = full.stat().st_size
        if size > MAX_FILE_SIZE:
            content = full.read_text(errors="replace")[:MAX_FILE_SIZE]
            return content + f"\n\n[Truncated — file is {size} bytes, showing first {MAX_FILE_SIZE}]"
        return full.read_text(errors="replace")
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error reading '{path}': {e}"


def list_files(directory: str) -> str:
    try:
        full = _safe_path(directory)
        if not full.is_dir():
            return f"Error: '{directory}' is not a directory."
        entries = sorted(full.iterdir())
        lines = []
        for entry in entries:
            name = entry.name
            if name in EXCLUDED_DIRS:
                continue
            lines.append(f"{name}/" if entry.is_dir() else name)
        return "\n".join(lines) if lines else "(empty directory)"
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error listing '{directory}': {e}"


def write_file(path: str, content: str) -> str:
    try:
        full = _safe_path(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        return f"Successfully wrote {len(content)} bytes to '{path}'."
    except ValueError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error writing '{path}': {e}"


def search_code(pattern: str, file_pattern: str | None = None) -> str:
    try:
        cmd = ["grep", "-rn", "--color=never"]
        for d in EXCLUDED_DIRS:
            cmd += [f"--exclude-dir={d}"]
        if file_pattern:
            cmd += [f"--include={file_pattern}"]
        cmd += [pattern, "."]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=COMMAND_TIMEOUT,
        )
        lines = result.stdout.strip().split("\n")
        if not result.stdout.strip():
            return "No matches found."
        if len(lines) > MAX_SEARCH_RESULTS:
            output = "\n".join(lines[:MAX_SEARCH_RESULTS])
            return output + f"\n\n[Showing first {MAX_SEARCH_RESULTS} of {len(lines)} matches]"
        return "\n".join(lines)
    except subprocess.TimeoutExpired:
        return "Error: search timed out."
    except Exception as e:
        return f"Error searching: {e}"


def run_command(command: str) -> str:
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=COMMAND_TIMEOUT,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        output = output.strip()
        if len(output) > 10_000:
            output = output[:10_000] + "\n\n[Truncated to 10000 characters]"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds."
    except Exception as e:
        return f"Error running command: {e}"


# --- Dispatcher ---

def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return the result as a string."""
    if tool_name == "read_file":
        return read_file(tool_input["path"])
    elif tool_name == "list_files":
        return list_files(tool_input["directory"])
    elif tool_name == "write_file":
        return write_file(tool_input["path"], tool_input["content"])
    elif tool_name == "search_code":
        return search_code(tool_input["pattern"], tool_input.get("file_pattern"))
    elif tool_name == "run_command":
        return run_command(tool_input["command"])
    else:
        return f"Error: unknown tool '{tool_name}'"
