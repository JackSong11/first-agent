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

load_dotenv(override=True)

from openai import OpenAI
from pathlib import Path

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)
WORKDIR = Path.cwd()

SYSTEM = f"""You are a coding agent at {WORKDIR}.
Use the todo tool to plan multi-step tasks. Mark in_progress before starting, completed when done.
Prefer tools over prose."""


# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):

            # 1. 提取并清洗数据（去空格、强制转字符串、设定默认值）
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))

            # 2. 字段校验：内容不能为空
            if not text:
                raise ValueError(f"Item {item_id}: text required")

            # 3. 枚举校验：状态必须在预定义范围内
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")

            # 4. 统计"进行中"的任务
            if status == "in_progress":
                in_progress_count += 1

            # 5. 添加到通过校验的列表
            validated.append({"id": item_id, "text": text, "status": status})

        # 严格限制：只能有一个任务处于"进行中"状态
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        # 只有全部通过校验，才更新内存中的数据
        self.items = validated
        # 更新成功后立即返回渲染后的字符串
        return self.render()

    def render(self) -> str:
        # 空列表处理
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            # 使用映射字典根据状态选择不同的前缀符号
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")

        # 计算进度：统计状态为 completed 的任务数量
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        # 以换行符连接所有行
        return "\n".join(lines)


TODO = TodoManager()


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


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


def run_read(path: str, limit: int = None) -> str:
    try:
        text = safe_path(path).read_text()
        lines = text.splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


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
    "todo": lambda **kw: TODO.update(kw["items"]),
}

TOOLS = [
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
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": "Update task list. Track progress on multi-step tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string"
                                },
                                "text": {
                                    "type": "string"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": [
                                        "pending",
                                        "in_progress",
                                        "completed"
                                    ]
                                }
                            },
                            "required": [
                                "id",
                                "text",
                                "status"
                            ]
                        }
                    }
                },
                "required": [
                    "items"
                ]
            }
        }
    },
]


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
    rounds_since_todo = 0

    while True:
        response = client.chat.completions.create(
            model="Qwen3-235B-A22B",
            messages=messages,
            tools=TOOLS,
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

        used_todo = False
        # 修正 4: 处理工具调用
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                # 解析参数
                args = json.loads(tool_call.function.arguments)
                # cmd = args.get("command")
                # 寻找对应的处理器并执行
                handler = TOOL_HANDLERS.get(function_name)
                try:
                    output = handler(**args) if handler else f"Error: Tool {function_name} not found."
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {function_name}: {str(output)[:200]}")

                if function_name == "todo":
                    used_todo = True

                # 修正 5: 汇报结果，role 必须是 tool
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": output
                })
        # 6. 维护任务提醒逻辑
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1

        if rounds_since_todo >= 3:
            # 在 OpenAI 模式下，提醒通常以 user 消息形式插入，引导模型下一步行动
            messages.append({
                "role": "user",
                "content": "Reminder: Please update your task list using the 'todo' tool if any progress was made."
            })
            # 重置计数器，避免每一轮都重复提醒
            rounds_since_todo = 0


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