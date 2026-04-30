#!/usr/bin/env python3
# Harness: compression -- clean memory for infinite sessions (OpenAI Version)
"""
s06_context_compact.py - Compact

Three-layer compression pipeline so the agent can work forever:

    Every turn:
    +------------------+
    | Tool call result |
    +------------------+
            |
            v
    [Layer 1: micro_compact]        (silent, every turn)
      Replace non-read_file tool_result content older than last 3
      with "[Previous: used {tool_name}]"
            |
            v
    [Check: tokens > 50000?]
       |               |
       no              yes
       |               |
       v               v
    continue    [Layer 2: auto_compact]
                  Save full transcript to .transcripts/
                  Ask LLM to summarize conversation.
                  Replace all messages with [summary].
                        |
                        v
                [Layer 3: compact tool]
                  Model calls compact -> immediate summarization.
                  Same as auto, triggered manually.

Key insight: "The agent can forget strategically and keep working forever."
"""

import json
import os
import subprocess
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

WORKDIR = Path.cwd()
client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)
MODEL = os.environ["LLM_MODEL_ID"]

SYSTEM = {"role": "system", "content": f"You are a coding agent at {WORKDIR}. Use tools to solve tasks."}

THRESHOLD = 50000
TRANSCRIPT_DIR = WORKDIR / ".transcripts"
KEEP_RECENT = 3
PRESERVE_RESULT_TOOLS = {"read_file"}


def estimate_tokens(messages: list) -> int:
    return len(str(messages)) // 4


# -- Layer 1: micro_compact --
def micro_compact(messages: list) -> list:
    # 找到所有的 tool 消息
    tool_results = [m for m in messages if m["role"] == "tool"]

    if len(tool_results) <= KEEP_RECENT:
        return messages

    # 获取工具名映射 (从 assistant 的 tool_calls 中提取)
    tool_name_map = {}
    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            for tc in msg["tool_calls"]:
                # 兼容对象或字典
                tc_id = tc.id if hasattr(tc, 'id') else tc.get('id')
                tc_name = tc.function.name if hasattr(tc, 'function') else tc.get('function', {}).get('name')
                tool_name_map[tc_id] = tc_name

    to_clear = tool_results[:-KEEP_RECENT]
    for msg in to_clear:
        # 如果已经压缩过或太短则跳过
        if "[Previous:" in msg["content"] or len(msg["content"]) <= 100:
            continue

        tool_name = tool_name_map.get(msg["tool_call_id"], "unknown")
        if tool_name in PRESERVE_RESULT_TOOLS:
            continue

        msg["content"] = f"[Previous: used {tool_name}]"
    return messages


# -- Layer 2: auto_compact --
def auto_compact(messages: list) -> list:
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    transcript_path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with open(transcript_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    print(f"[transcript saved: {transcript_path}]")

    conversation_text = json.dumps(messages, default=str)[-80000:]

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content":
            "Summarize this conversation for continuity. Include: "
            "1) What was accomplished, 2) Current state, 3) Key decisions made. "
            "Be concise but preserve critical details.\n\n" + conversation_text}],
        max_tokens=2000,
    )

    summary = response.choices[0].message.content or "No summary generated."

    # 返回新的消息列表，保留系统提示词
    return [
        SYSTEM,
        {"role": "user", "content": f"[Conversation compressed. Transcript: {transcript_path}]\n\n{summary}"},
    ]


# -- Tool implementations (保持逻辑不变) --
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except Exception as e:
        return f"Error: {e}"


# ... 其他工具函数 (run_read, run_write, run_edit) 逻辑相同 ...
def run_read(path: str, limit: int = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines): lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path);
        fp.parent.mkdir(parents=True, exist_ok=True);
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path);
        content = fp.read_text()
        if old_text not in content: return f"Error: Text not found"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "compact": lambda **kw: "Manual compression requested.",
}

# OpenAI 工具格式
TOOLS = [
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command.",
                                      "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                                                     "required": ["command"]}}},
    {"type": "function", "function": {"name": "read_file", "description": "Read file contents.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                                      "limit": {"type": "integer"}},
                                                     "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write content to file.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                                      "content": {"type": "string"}},
                                                     "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace text.",
                                      "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                                                                                      "old_text": {"type": "string"},
                                                                                      "new_text": {"type": "string"}},
                                                     "required": ["path", "old_text", "new_text"]}}},
    {"type": "function", "function": {"name": "compact", "description": "Trigger manual compression.",
                                      "parameters": {"type": "object", "properties": {"focus": {"type": "string"}},
                                                     "required": []}}},
]


def agent_loop(messages: list):
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, SYSTEM)

    while True:
        micro_compact(messages)
        if estimate_tokens(messages) > THRESHOLD:
            print("[auto_compact triggered]")
            messages[:] = auto_compact(messages)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )

        msg = response.choices[0].message
        # 将 assistant 消息存入历史（包含 tool_calls）
        # messages.append(msg)
        # 修复：将对象转换为字典再存入列表
        messages.append(msg.model_dump())

        if not msg.tool_calls:
            return

        manual_compact = False
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if name == "compact":
                manual_compact = True
                output = "Compressing..."
            else:
                handler = TOOL_HANDLERS.get(name)
                output = handler(**args) if handler else f"Unknown tool: {name}"

            print(f"> {name}: {str(output)[:100]}...")

            # OpenAI 的工具结果必须带上 tool_call_id，角色为 tool
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": str(output)
            })

        if manual_compact:
            print("[manual compact]")
            messages[:] = auto_compact(messages)
            return


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("\033[36ms06_openai >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if not query.strip() or query.lower() in ("q", "exit"): break

        history.append({"role": "user", "content": query})
        agent_loop(history)

        last_msg = history[-1]
        # 在你的代码第 246 行左右，为了兼容性，你将 OpenAI 的响应对象转换成了字典：messages.append(msg.model_dump())
        # 这意味着 history 列表中的最后一个元素现在是一个普通的 Python 字典。在 Python 中，访问字典的值应该使用 obj["key"] 语法，而不是 obj.key 属性语法。
        if last_msg["role"] == "assistant" and last_msg.get("content"):
            print(last_msg.get("content"))
