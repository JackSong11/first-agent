#!/usr/bin/env python3
# Harness: protocols -- structured handshakes between models.
"""
s10_team_protocols.py - Team Protocols

Shutdown protocol and plan approval protocol, both using the same
request_id correlation pattern. Builds on s09's team messaging.

    Shutdown FSM: pending -> approved | rejected

    Lead                              Teammate
    +---------------------+          +---------------------+
    | shutdown_request     |          |                     |
    | {                    | -------> | receives request    |
    |   request_id: abc    |          | decides: approve?   |
    | }                    |          |                     |
    +---------------------+          +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | shutdown_response    | <------- | shutdown_response   |
    | {                    |          | {                   |
    |   request_id: abc    |          |   request_id: abc   |
    |   approve: true      |          |   approve: true     |
    | }                    |          | }                   |
    +---------------------+          +---------------------+
            |
            v
    status -> "shutdown", thread stops

    Plan approval FSM: pending -> approved | rejected

    Teammate                          Lead
    +---------------------+          +---------------------+
    | plan_approval        |          |                     |
    | submit: {plan:"..."}| -------> | reviews plan text   |
    +---------------------+          | approve/reject?     |
                                     +---------------------+
                                             |
    +---------------------+          +-------v-------------+
    | plan_approval_resp   | <------- | plan_approval       |
    | {approve: true}      |          | review: {req_id,    |
    +---------------------+          |   approve: true}     |
                                     +---------------------+

    Trackers: {request_id: {"target|from": name, "status": "pending|..."}}

Key insight: "Same request_id correlation pattern, two domains."
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

WORKDIR = Path.cwd()
# 初始化 OpenAI 客户端
# 注意：如果你使用第三方转发或本地模型，可以通过 base_url 和 api_key 配置
client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)
MODEL = os.environ["LLM_MODEL_ID"]
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"

SYSTEM = f"You are a team lead at {WORKDIR}. Manage teammates with shutdown and plan approval protocols."

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- Request trackers: correlate by request_id --
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()


# -- MessageBus: JSONL inbox per teammate --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)

    def send(self, sender: str, to: str, content: str,
             msg_type: str = "message", extra: dict = None) -> str:
        if msg_type not in VALID_MSG_TYPES:
            return f"Error: Invalid type '{msg_type}'. Valid: {VALID_MSG_TYPES}"
        msg = {
            "type": msg_type,
            "from": sender,
            "content": content,
            "timestamp": time.time(),
        }
        if extra:
            msg.update(extra)
        inbox_path = self.dir / f"{to}.jsonl"
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg) + "\n")
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            return []
        messages = []
        for line in inbox_path.read_text().strip().splitlines():
            if line:
                messages.append(json.loads(line))
        inbox_path.write_text("")
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager with shutdown + plan approval --
class TeammateManager:
    def __init__(self, team_dir: Path):
        self.dir = team_dir
        self.dir.mkdir(exist_ok=True)
        self.config_path = self.dir / "config.json"
        self.config = self._load_config()
        self.threads = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            return json.loads(self.config_path.read_text())
        return {"team_name": "default", "members": []}

    def _save_config(self):
        self.config_path.write_text(json.dumps(self.config, indent=2))

    def _find_member(self, name: str) -> dict:
        for m in self.config["members"]:
            if m["name"] == name:
                return m
        return None

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
            member["role"] = role
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(
            target=self._teammate_loop,
            args=(name, role, prompt),
            daemon=True,
        )
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    def _teammate_loop(self, name: str, role: str, prompt: str):
        sys_prompt = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Submit plans via plan_approval before major work. "
            f"Respond to shutdown_request with shutdown_response."
        )
        # OpenAI style messages
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ]
        tools = self._get_openai_tools(is_lead=False)
        should_exit = False
        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": json.dumps(msg)})
            if should_exit:
                break
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                )
                choice = response.choices[0]
                messages.append(choice.message)  # 必须添加完整消息对象

                if choice.finish_reason == "tool_calls":
                    for tool_call in choice.message.tool_calls:
                        args = json.loads(tool_call.function.arguments)
                        output = self._exec(name, tool_call.function.name, args)
                        print(f"  [{name}] {tool_call.function.name}: {str(output)[:80]}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(output)
                        })
                        if tool_call.function.name == "shutdown_response" and args.get("approve"):
                            should_exit = True
                else:
                    break
            except Exception as e:
                print(f"Error in {name} loop: {e}")
                break

        member = self._find_member(name)
        if member:
            member["status"] = "shutdown" if should_exit else "idle"
            self._save_config()

    def _get_openai_tools(self, is_lead=True):
        # OpenAI 定义工具的格式
        base_tools = [
            {"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
                "command": {"type": "string"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}}, "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "send_message", "parameters": {"type": "object", "properties": {
                "to": {"type": "string"}, "content": {"type": "string"}, "msg_type": {"type": "string"}},
                                                                                     "required": ["to", "content"]}}},
            {"type": "function",
             "function": {"name": "read_inbox", "parameters": {"type": "object", "properties": {}}}},
        ]
        if is_lead:
            lead_tools = [
                {"type": "function", "function": {"name": "spawn_teammate", "parameters": {"type": "object",
                                                                                           "properties": {"name": {
                                                                                               "type": "string"},
                                                                                               "role": {
                                                                                                   "type": "string"},
                                                                                               "prompt": {
                                                                                                   "type": "string"}},
                                                                                           "required": ["name", "role",
                                                                                                        "prompt"]}}},
                {"type": "function", "function": {"name": "shutdown_request", "parameters": {"type": "object",
                                                                                             "properties": {
                                                                                                 "teammate": {
                                                                                                     "type": "string"}},
                                                                                             "required": [
                                                                                                 "teammate"]}}},
                {"type": "function", "function": {"name": "plan_approval", "parameters": {"type": "object",
                                                                                          "properties": {"request_id": {
                                                                                              "type": "string"},
                                                                                              "approve": {
                                                                                                  "type": "boolean"}},
                                                                                          "required": ["request_id",
                                                                                                       "approve"]}}},
            ]
            return base_tools + lead_tools
        else:
            member_tools = [
                {"type": "function", "function": {"name": "shutdown_response", "parameters": {"type": "object",
                                                                                              "properties": {
                                                                                                  "request_id": {
                                                                                                      "type": "string"},
                                                                                                  "approve": {
                                                                                                      "type": "boolean"}},
                                                                                              "required": ["request_id",
                                                                                                           "approve"]}}},
                {"type": "function", "function": {"name": "plan_approval", "parameters": {"type": "object",
                                                                                          "properties": {"plan": {
                                                                                              "type": "string"}},
                                                                                          "required": ["plan"]}}},
            ]
            return base_tools + member_tools

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        # these base tools are unchanged from s02
        if tool_name == "bash":
            return _run_bash(args["command"])
        if tool_name == "read_file":
            return _run_read(args["path"])
        if tool_name == "write_file":
            return _run_write(args["path"], args["content"])
        if tool_name == "edit_file":
            return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message":
            return BUS.send(sender, args["to"], args["content"], args.get("msg_type", "message"))
        if tool_name == "read_inbox":
            return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "shutdown_response":
            req_id = args["request_id"]
            approve = args["approve"]
            with _tracker_lock:
                if req_id in shutdown_requests:
                    shutdown_requests[req_id]["status"] = "approved" if approve else "rejected"
            BUS.send(
                sender, "lead", args.get("reason", ""),
                "shutdown_response", {"request_id": req_id, "approve": approve},
            )
            return f"Shutdown {'approved' if approve else 'rejected'}"
        if tool_name == "plan_approval":
            plan_text = args.get("plan", "")
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock:
                plan_requests[req_id] = {"from": sender, "plan": plan_text, "status": "pending"}
            BUS.send(
                sender, "lead", plan_text, "plan_approval_response",
                {"request_id": req_id, "plan": plan_text},
            )
            return f"Plan submitted (request_id={req_id}). Waiting for lead approval."
        return f"Unknown tool: {tool_name}"

    def list_all(self) -> str:
        if not self.config["members"]:
            return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# -- Base tool implementations (these base tools are unchanged from s02) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _run_bash(command: str) -> str:
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(
            command, shell=True, cwd=WORKDIR,
            capture_output=True, text=True, timeout=120,
        )
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def _run_read(path: str, limit: int = None) -> str:
    try:
        lines = _safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more)"]
        return "\n".join(lines)[:50000]
    except Exception as e:
        return f"Error: {e}"


def _run_write(path: str, content: str) -> str:
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"Wrote {len(content)} bytes"
    except Exception as e:
        return f"Error: {e}"


def _run_edit(path: str, old_text: str, new_text: str) -> str:
    try:
        fp = _safe_path(path)
        c = fp.read_text()
        if old_text not in c:
            return f"Error: Text not found in {path}"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead-specific protocol handlers --
def handle_shutdown_request(teammate: str) -> str:
    req_id = str(uuid.uuid4())[:8]
    with _tracker_lock:
        shutdown_requests[req_id] = {"target": teammate, "status": "pending"}
    BUS.send(
        "lead", teammate, "Please shut down gracefully.",
        "shutdown_request", {"request_id": req_id},
    )
    return f"Shutdown request {req_id} sent to '{teammate}' (status: pending)"


def handle_plan_review(request_id: str, approve: bool, feedback: str = "") -> str:
    with _tracker_lock:
        req = plan_requests.get(request_id)
    if not req:
        return f"Error: Unknown plan request_id '{request_id}'"
    with _tracker_lock:
        req["status"] = "approved" if approve else "rejected"
    BUS.send(
        "lead", req["from"], feedback, "plan_approval_response",
        {"request_id": request_id, "approve": approve, "feedback": feedback},
    )
    return f"Plan {req['status']} for '{req['from']}'"


def _check_shutdown_status(request_id: str) -> str:
    with _tracker_lock:
        return json.dumps(shutdown_requests.get(request_id, {"error": "not found"}))


# -- Lead tool dispatch (12 tools) --
TOOL_HANDLERS = {
    "bash": lambda **kw: _run_bash(kw["command"]),
    "read_file": lambda **kw: _run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: _run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: _run_edit(kw["path"], kw["old_text"], kw["new_text"]),
    "spawn_teammate": lambda **kw: TEAM.spawn(kw["name"], kw["role"], kw["prompt"]),
    "list_teammates": lambda **kw: TEAM.list_all(),
    "send_message": lambda **kw: BUS.send("lead", kw["to"], kw["content"], kw.get("msg_type", "message")),
    "read_inbox": lambda **kw: json.dumps(BUS.read_inbox("lead"), indent=2),
    "broadcast": lambda **kw: BUS.broadcast("lead", kw["content"], TEAM.member_names()),
    "shutdown_request": lambda **kw: handle_shutdown_request(kw["teammate"]),
    "shutdown_response": lambda **kw: _check_shutdown_status(kw.get("request_id", "")),
    "plan_approval": lambda **kw: handle_plan_review(kw["request_id"], kw["approve"], kw.get("feedback", "")),
}


def agent_loop(messages: list):
    """
    OpenAI 版本的循环逻辑，现在使用与 Anthropic 版本相同的 Handler 模式
    """
    # 这里的 SYSTEM 应该作为列表的第一条消息或在 chat.completions 中处理
    # 为了保持上下文连续性，如果 messages 为空，先插入 system
    if not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM})

    while True:
        # 1. 检查收件箱
        inbox = BUS.read_inbox("lead")
        if inbox:
            messages.append({
                "role": "user",
                "content": f"<inbox>{json.dumps(inbox, indent=2)}</inbox>",
            })

        # 2. 调用模型
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TEAM._get_openai_tools(is_lead=True),  # 确保此方法返回完整的 12 个工具定义
        )

        choice = response.choices[0]
        msg = choice.message
        messages.append(msg)  # OpenAI 要求必须将包含 tool_calls 的 assistant 消息存入历史

        # 3. 如果模型不需要调用工具，退出循环
        if choice.finish_reason != "tool_calls":
            return

        # 4. 处理工具调用 (与 Anthropic 逻辑一致)
        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except Exception:
                args = {}

            # 使用统一的 TOOL_HANDLERS
            handler = TOOL_HANDLERS.get(tool_name)
            try:
                output = handler(**args) if handler else f"Unknown tool: {tool_name}"
            except Exception as e:
                output = f"Error: {e}"

            # 保持一致的打印格式
            print(f"> {tool_name}:")
            print(str(output)[:200])

            # 将结果添加回消息历史
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(output),
            })


if __name__ == "__main__":
    history = []
    print("OpenAI Team Control Shell (s10)")
    while True:
        try:
            query = input("\033[32ms10_openai >> \033[0m").strip()
            if query.lower() in ("q", "exit", ""):
                break
            if query == "/team":
                print(TEAM.list_all())
                continue
            if query == "/inbox":
                print(json.dumps(BUS.read_inbox("lead"), indent=2))
                continue

            history.append({"role": "user", "content": query})
            agent_loop(history)
        except KeyboardInterrupt:
            break
