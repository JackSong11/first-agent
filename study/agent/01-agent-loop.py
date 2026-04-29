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

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)

SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
# Anthropic 标准
# TOOLS = [{
#     "name": "bash",
#     "description": "Run a shell command.",
#     "input_schema": {
#         "type": "object",
#         "properties": {"command": {"type": "string"}},
#         "required": ["command"],
#     },
# }]
# openai 标准
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {  # OpenAI 使用 parameters 而不是 input_schema
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
}]


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


# -- The core pattern: a while loop that calls tools until the model stops --
def agent_loop(messages: list):
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

        # 修正 4: 处理工具调用
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                # 解析参数
                args = json.loads(tool_call.function.arguments)
                cmd = args.get("command")

                print(f"\033[33m$ {cmd}\033[0m")
                output = run_bash(cmd)
                print(output[:200])

                # 修正 5: 汇报结果，role 必须是 tool
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": "bash",
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
