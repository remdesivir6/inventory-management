"""GitHub Issue Coding Agent.

Reads a GitHub issue, uses Claude with tool-use to explore and modify the
codebase, then creates a branch and opens a pull request.

Expected environment variables (set by the GitHub Actions workflow):
  ANTHROPIC_API_KEY  — Anthropic API key
  GITHUB_TOKEN       — GitHub token with repo/PR/issue permissions
  ISSUE_NUMBER       — The issue number to work on
  REPO_FULL_NAME     — owner/repo (e.g. remdesivir6/inventory-management)
"""

import json
import os
import subprocess
import sys

import anthropic

from tools import TOOLS, execute_tool

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 8192
MAX_ITERATIONS = 50


def gh(*args: str) -> subprocess.CompletedProcess:
    """Run a `gh` CLI command with the GITHUB_TOKEN in the environment."""
    env = os.environ.copy()
    env["GH_TOKEN"] = os.environ["GITHUB_TOKEN"]
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        env=env,
    )


def git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command in the repo root."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
    )


def fetch_issue(repo: str, issue_number: str) -> dict:
    """Fetch issue title and body via `gh`."""
    result = gh("issue", "view", issue_number, "--repo", repo, "--json", "title,body,labels")
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch issue: {result.stderr}")
    return json.loads(result.stdout)


def post_comment(repo: str, issue_number: str, body: str) -> None:
    """Post a comment on the issue."""
    gh("issue", "comment", issue_number, "--repo", repo, "--body", body)


def build_system_prompt(repo: str, issue_number: str) -> str:
    return f"""\
You are a coding agent working on the repository {repo}.
You have been assigned GitHub issue #{issue_number}.

Your task: understand the issue, explore the codebase using the provided tools,
and make the necessary code changes to resolve it.

Rules:
- Start by listing the root directory to understand the project structure.
- Read relevant files before making changes.
- Follow existing code style and conventions.
- Make minimal, focused changes that directly address the issue.
- Do not modify files unrelated to the issue.
- Do not commit or push — that happens automatically after you finish.
- When you are confident all changes are complete, stop calling tools and
  write a brief summary of what you changed and why.

The repository uses Vue 3 (frontend) and Python FastAPI (backend)."""


def run_agent_loop(issue: dict, system_prompt: str) -> None:
    """Run the agentic tool-use loop with Claude."""
    client = anthropic.Anthropic()

    user_message = (
        f"Please resolve the following GitHub issue.\n\n"
        f"**Title:** {issue['title']}\n\n"
        f"**Description:**\n{issue.get('body') or '(no description provided)'}"
    )

    messages = [{"role": "user", "content": user_message}]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration}/{MAX_ITERATIONS} ---")

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        # Print any text blocks from the response
        for block in response.content:
            if hasattr(block, "text"):
                print(f"Claude: {block.text}")

        if response.stop_reason == "end_turn":
            print("\nAgent finished (end_turn).")
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  Tool call: {block.name}({json.dumps(block.input)[:200]})")
                    result = execute_tool(block.name, block.input)
                    print(f"  Result: {result[:200]}{'...' if len(result) > 200 else ''}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            print(f"\nUnexpected stop reason: {response.stop_reason}")
            break
    else:
        print(f"\nAgent hit iteration limit ({MAX_ITERATIONS}).")


def create_branch_and_pr(repo: str, issue_number: str, issue_title: str) -> None:
    """Create a branch, commit changes, push, and open a PR."""
    branch = f"agent/issue-{issue_number}"

    # Configure git identity for the commit
    git("config", "user.name", "coding-agent[bot]")
    git("config", "user.email", "coding-agent[bot]@users.noreply.github.com")

    git("checkout", "-b", branch)

    # Stage all changes (new, modified, deleted files)
    git("add", "-A")

    # Check if there are any changes to commit
    status = git("status", "--porcelain")
    if not status.stdout.strip():
        print("No changes were made by the agent.")
        post_comment(repo, issue_number, "The agent analyzed the issue but did not make any code changes.")
        return

    commit_msg = f"fix: resolve #{issue_number} — {issue_title}"
    git("commit", "-m", commit_msg)

    push = git("push", "origin", branch)
    if push.returncode != 0:
        raise RuntimeError(f"Failed to push branch: {push.stderr}")

    pr_body = (
        f"Resolves #{issue_number}\n\n"
        f"This PR was automatically generated by the coding agent.\n\n"
        f"---\n"
        f"*Please review the changes carefully before merging.*"
    )
    pr_result = gh(
        "pr", "create",
        "--repo", repo,
        "--base", "main",
        "--head", branch,
        "--title", f"fix: {issue_title}",
        "--body", pr_body,
    )
    if pr_result.returncode != 0:
        raise RuntimeError(f"Failed to create PR: {pr_result.stderr}")

    pr_url = pr_result.stdout.strip()
    print(f"\nPR created: {pr_url}")
    post_comment(repo, issue_number, f"The agent has opened a pull request: {pr_url}")


def main() -> None:
    issue_number = os.environ["ISSUE_NUMBER"]
    repo = os.environ["REPO_FULL_NAME"]

    print(f"Agent starting for issue #{issue_number} in {repo}")

    issue = fetch_issue(repo, issue_number)
    print(f"Issue: {issue['title']}")

    post_comment(
        repo,
        issue_number,
        "The coding agent has picked up this issue and is working on it. :robot:",
    )

    system_prompt = build_system_prompt(repo, issue_number)
    run_agent_loop(issue, system_prompt)
    create_branch_and_pr(repo, issue_number, issue["title"])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        # Try to post an error comment on the issue
        try:
            repo = os.environ.get("REPO_FULL_NAME", "")
            issue_number = os.environ.get("ISSUE_NUMBER", "")
            if repo and issue_number:
                post_comment(
                    repo,
                    issue_number,
                    f"The coding agent encountered an error:\n```\n{e}\n```",
                )
        except Exception:
            pass  # Don't mask the original error
        sys.exit(1)
