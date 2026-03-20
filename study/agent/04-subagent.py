#!/usr/bin/env python3
# Harness: the loop -- the model's first connection to the real world.
"""
s01_agent_loop.py - The Agent Loop

The entire secret of an AI coding agent in one pattern:

    while stop_reason == "tool_use":
        response = LLM(messages, tools)
        execute tools
        append results

    +----------+      +-------+      +---------+
    |   User   | ---> |  LLM  | ---> |  Tool   |
    |  prompt  |      |       |      | execute |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   tool_result |
                          +---------------+
                          (loop continues)

This is the core loop: feed tool results back to the model
until the model decides to stop. Production agents layer
policy, hooks, and lifecycle controls on top.
"""
import json
import os
import subprocess
from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
load_dotenv(override=True)

from openai import OpenAI
from pathlib import Path

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)
client = wrap_openai(client)
WORKDIR = Path.cwd()

SYSTEM = f"You are a coding agent at {WORKDIR}. Use the task tool to delegate exploration or subtasks."
SUBAGENT_SYSTEM = f"You are a coding subagent at {WORKDIR}. Complete the given task, then summarize your findings."


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@traceable(run_type="tool") # 标记为工具类型
def run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

@traceable(run_type="tool") # 标记为工具类型
def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"

@traceable(run_type="tool") # 标记为工具类型
def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

@traceable(run_type="tool") # 标记为工具类型
def run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_text not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- The dispatch map: {tool_name: handler} --
TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}

CHILD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact text in file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
]

import json


# -- Subagent: 使用 OpenAI 标准的循环逻辑 --
@traceable(name="Subagent Loop") # 可以在后台自定义显示名称
def run_subagent(prompt: str) -> str:
    # 1. 初始化独立上下文
    sub_messages = [
        {"role": "system", "content": SUBAGENT_SYSTEM},
        {"role": "user", "content": prompt}
    ]

    for _ in range(30):  # 安全限制，防止无限循环
        response = client.chat.completions.create(
            model="Qwen3-235B-A22B",
            messages=sub_messages,
            tools=CHILD_TOOLS,  # 传入子工具集
            max_tokens=8000,
        )

        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 2. 必须将模型的消息（包含 tool_calls）存入历史
        sub_messages.append(response_message)

        # 3. 检查是否需要调用工具
        if finish_reason != "tool_calls":
            break

        # 4. 处理并执行工具
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                # 执行对应的 Handler
                handler = TOOL_HANDLERS.get(function_name)
                if handler:
                    try:
                        output = handler(**args)
                    except Exception as e:
                        output = f"Error executing tool: {str(e)}"
                else:
                    output = f"Error: Tool {function_name} not found."

                # 5. 将结果以 role: tool 存入历史
                sub_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": str(output)
                })

    # 6. 只返回最后一条文本消息给父代理，丢弃子上下文
    final_content = response.choices[0].message.content
    return final_content if final_content else "(Subagent completed task without text summary)"


# -- Parent tools 结构定义 (OpenAI 格式) --
PARENT_TOOLS = CHILD_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Spawn a subagent with fresh context. It shares the filesystem but not conversation history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string"
                    },
                    "description": {
                        "type": "string",
                        "description": "Short description of the task"
                    }
                },
                "required": ["prompt"]
            }
        }
    }
]


# -- The core pattern: a while loop that calls tools until the model stops --
@traceable(name="Main Agent Loop")
def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model="Qwen3-235B-A22B",
            messages=messages,
            tools=PARENT_TOOLS,
            max_tokens=8000,
        )
        # Append assistant turn
        response_message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 必须把模型生成的这条消息（包含 tool_calls 或 content）存入历史
        messages.append(response_message)

        # 修正 3: 检查停止原因 (tool_calls)
        if finish_reason != "tool_calls":
            break

        # 修正 4: 处理工具调用
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                # 解析参数
                args = json.loads(tool_call.function.arguments)
                # 特殊逻辑：如果是 task 工具，则派生子代理
                if function_name == "task":
                    desc = args.get("description", "subtask")
                    prompt = args.get("prompt")
                    print(f"> task ({desc}): {prompt[:80]}...")

                    # 调用之前改好的 OpenAI 版 run_subagent
                    output = run_subagent(prompt)
                else:
                    # 普通工具逻辑
                    handler = TOOL_HANDLERS.get(function_name)
                    if handler:
                        output = handler(**args)
                    else:
                        output = f"Error: Tool {function_name} not found."

                print(f"  Result: {str(output)[:200]}")
                # 修正 5: 汇报结果，role 必须是 tool
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": output
                })


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)

        # 打印最后一条非工具调用的回复
        last_msg = history[-1]
        if hasattr(last_msg, 'content') and last_msg.content:
            print(f"\nAI: {last_msg.content}")
        elif isinstance(last_msg, dict) and last_msg.get("content"):
            print(f"\nAI: {last_msg['content']}")