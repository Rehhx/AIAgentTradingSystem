"""
agents/_claude_sdk.py
---------------------
small sync wrapper around claude_agent_sdk's async query() function. the
research / autonomous / code agents use this so their orchestrator-facing
run(task) methods stay synchronous.

requires claude-agent-sdk (which in turn requires the Claude Code CLI to be
installed on the host machine — see https://docs.claude.com/claude-code).
"""

import asyncio
from typing import Optional


def ask_claude(prompt: str, system_prompt: str, allowed_tools: Optional[list] = None,
               model: Optional[str] = None, max_turns: int = 6) -> str:
    """
    fire a one-shot query to claude via the agent sdk. returns the
    concatenated text of all assistant messages. raises ClaudeSDKError on
    transport / cli failures so callers can decide whether to retry.
    """
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

    options = ClaudeAgentOptions(
        system_prompt = system_prompt,
        allowed_tools = allowed_tools or [],
        model         = model,
        max_turns     = max_turns,
        permission_mode = "bypassPermissions" if allowed_tools else "default",
    )

    async def _collect() -> str:
        parts = []
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
        return "".join(parts)

    return asyncio.run(_collect())
