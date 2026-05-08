#!/usr/bin/env python3
# Harness: team mailboxes -- multiple models, coordinated through files.
"""
s09_agent_teams.py - Agent Teams

Persistent named agents with file-based JSONL inboxes. Each teammate runs
its own agent loop in a separate thread. Communication via append-only inboxes.

    Subagent (s04):  spawn -> execute -> return summary -> destroyed
    Teammate (s09):  spawn -> work -> idle -> work -> ... -> shutdown

    .team/config.json                   .team/inbox/
    +----------------------------+      +------------------+
    | {"team_name": "default",   |      | alice.jsonl      |
    |  "members": [              |      | bob.jsonl        |
    |    {"name":"alice",        |      | lead.jsonl       |
    |     "role":"coder",        |      +------------------+
    |     "status":"idle"}       |
    |  ]}                        |      send_message("alice", "fix bug"):
    +----------------------------+        open("alice.jsonl", "a").write(msg)

                                        read_inbox("alice"):
    spawn_teammate("alice","coder",...)   msgs = [json.loads(l) for l in ...]
         |                                open("alice.jsonl", "w").close()
         v                                return msgs  # drain
    Thread: alice             Thread: bob
    +------------------+      +------------------+
    | agent_loop       |      | agent_loop       |
    | status: working  |      | status: idle     |
    | ... runs tools   |      | ... waits ...    |
    | status -> idle   |      |                  |
    +------------------+      +------------------+

    5 message types (all declared, not all handled here):
    +-------------------------+-----------------------------------+
    | message                 | Normal text message               |
    | broadcast               | Sent to all teammates             |
    | shutdown_request        | Request graceful shutdown (s10)   |
    | shutdown_response       | Approve/reject shutdown (s10)     |
    | plan_approval_response  | Approve/reject plan (s10)         |
    +-------------------------+-----------------------------------+

Key insight: "Teammates that can talk to each other."
"""

import json
import os
import subprocess
import threading
import time
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

SYSTEM = f"You are a team lead at {WORKDIR}. Spawn teammates and communicate via inboxes."

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}


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
        try:
            content = inbox_path.read_text().strip()
            if content:
                for line in content.splitlines():
                    messages.append(json.loads(line))
            inbox_path.write_text("")  # 读后即焚 (Drain)
        except Exception as e:
            return [f"Error reading inbox: {e}"]
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- TeammateManager: persistent named agents with config.json --
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
        sys_msg = (
            f"You are '{name}', role: {role}, at {WORKDIR}. "
            f"Use send_message to communicate. Complete your task."
        )
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": prompt}
        ]
        tools = self._get_openai_tools(is_lead=False)

        for _ in range(50):
            inbox = BUS.read_inbox(name)
            for msg in inbox:
                messages.append({"role": "user", "content": f"INBOX MESSAGE: {json.dumps(msg)}"})

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=tools,
                )
            except Exception as e:
                print(f"Error in {name} loop: {e}")
                break

            msg_obj = response.choices[0].message
            messages.append(msg_obj)

            if not msg_obj.tool_calls:
                break

            for tool_call in msg_obj.tool_calls:
                args = json.loads(tool_call.function.arguments)
                output = self._exec(name, tool_call.function.name, args)
                print(f"  [{name}] {tool_call.function.name}: {str(output)[:100]}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(output)
                })

        member = self._find_member(name)
        if member and member["status"] != "shutdown":
            member["status"] = "idle"
            self._save_config()

    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash": return _run_bash(args["command"])
        if tool_name == "read_file": return _run_read(args["path"])
        if tool_name == "write_file": return _run_write(args["path"], args["content"])
        if tool_name == "edit_file": return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message": return BUS.send(sender, args["to"], args["content"],
                                                        args.get("msg_type", "message"))
        if tool_name == "read_inbox": return json.dumps(BUS.read_inbox(sender), indent=2)
        return f"Unknown tool: {tool_name}"

    def _get_openai_tools(self, is_lead: bool = False) -> list:
        # 通用工具
        tools = [
            {"type": "function", "function": {"name": "bash", "description": "Run shell command",
                                              "parameters": {"type": "object",
                                                             "properties": {"command": {"type": "string"}},
                                                             "required": ["command"]}}},
            {"type": "function", "function": {"name": "read_file", "description": "Read file",
                                              "parameters": {"type": "object",
                                                             "properties": {"path": {"type": "string"}},
                                                             "required": ["path"]}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write file",
                                              "parameters": {"type": "object",
                                                             "properties": {"path": {"type": "string"},
                                                                            "content": {"type": "string"}},
                                                             "required": ["path", "content"]}}},
            {"type": "function", "function": {"name": "edit_file", "description": "Replace text",
                                              "parameters": {"type": "object",
                                                             "properties": {"path": {"type": "string"},
                                                                            "old_text": {"type": "string"},
                                                                            "new_text": {"type": "string"}},
                                                             "required": ["path", "old_text", "new_text"]}}},
            {"type": "function", "function": {"name": "send_message", "description": "Send msg to teammate",
                                              "parameters": {"type": "object", "properties": {"to": {"type": "string"},
                                                                                              "content": {
                                                                                                  "type": "string"},
                                                                                              "msg_type": {
                                                                                                  "type": "string",
                                                                                                  "enum": list(
                                                                                                      VALID_MSG_TYPES)}},
                                                             "required": ["to", "content"]}}},
            {"type": "function", "function": {"name": "read_inbox", "description": "Read own inbox",
                                              "parameters": {"type": "object", "properties": {}}}},
        ]
        if is_lead:
            tools.extend([
                {"type": "function", "function": {"name": "spawn_teammate", "description": "Spawn agent",
                                                  "parameters": {"type": "object",
                                                                 "properties": {"name": {"type": "string"},
                                                                                "role": {"type": "string"},
                                                                                "prompt": {"type": "string"}},
                                                                 "required": ["name", "role", "prompt"]}}},
                {"type": "function", "function": {"name": "list_teammates", "description": "List all team",
                                                  "parameters": {"type": "object", "properties": {}}}},
                {"type": "function", "function": {"name": "broadcast", "description": "Msg all",
                                                  "parameters": {"type": "object",
                                                                 "properties": {"content": {"type": "string"}},
                                                                 "required": ["content"]}}},
            ])
        return tools

    def list_all(self) -> str:
        if not self.config["members"]: return "No teammates."
        lines = [f"Team: {self.config['team_name']}"]
        for m in self.config["members"]:
            lines.append(f"  {m['name']} ({m['role']}): {m['status']}")
        return "\n".join(lines)

    def member_names(self) -> list:
        return [m["name"] for m in self.config["members"]]


TEAM = TeammateManager(TEAM_DIR)


# -- Filesystem Helpers (Unchanged Logic) --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError(f"Escape: {p}")
    return path


def _run_bash(command: str) -> str:
    if any(d in command for d in ["rm -rf /", "sudo", "shutdown"]): return "Blocked"
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        return (r.stdout + r.stderr).strip()[:5000]
    except Exception as e:
        return f"Error: {e}"


def _run_read(path: str) -> str:
    try:
        return _safe_path(path).read_text()[:5000]
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
        if old_text not in c: return "Text not found"
        fp.write_text(c.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


# -- Lead Agent Loop --
def lead_agent_loop(history: list):
    # 初始化 System Prompt 如果没有的话
    if not any(m.get("role") == "system" for m in history):
        history.insert(0, {"role": "system", "content": SYSTEM})

    while True:
        # Check Inbox
        inbox = BUS.read_inbox("lead")
        if inbox:
            history.append({"role": "user", "content": f"SYSTEM: New messages in lead inbox: {json.dumps(inbox)}"})

        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TEAM._get_openai_tools(is_lead=True),
        )

        msg_obj = response.choices[0].message
        history.append(msg_obj)

        if not msg_obj.tool_calls:
            if msg_obj.content:
                print(f"\n[Lead]: {msg_obj.content}")
            return

        for tool_call in msg_obj.tool_calls:
            t_name = tool_call.function.name
            t_args = json.loads(tool_call.function.arguments)

            # Dispatch
            if t_name == "spawn_teammate":
                out = TEAM.spawn(**t_args)
            elif t_name == "list_teammates":
                out = TEAM.list_all()
            elif t_name == "send_message":
                out = BUS.send("lead", t_args["to"], t_args["content"], t_args.get("msg_type", "message"))
            elif t_name == "read_inbox":
                out = json.dumps(BUS.read_inbox("lead"))
            elif t_name == "broadcast":
                out = TEAM.broadcast("lead", t_args["content"], TEAM.member_names())
            else:
                out = TEAM._exec("lead", t_name, t_args)

            print(f"> {t_name}: {str(out)[:100]}...")
            history.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(out)
            })


if __name__ == "__main__":
    chat_history = []
    print(f"--- Team Inbox System (OpenAI / {MODEL}) ---")
    while True:
        try:
            user_input = input("\033[36ms09 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("q", "exit"): break
        if user_input.strip() == "/team":
            print(TEAM.list_all())
            continue
        if user_input.strip() == "/inbox":
            print(json.dumps(BUS.read_inbox("lead"), indent=2));
            continue

        chat_history.append({"role": "user", "content": user_input})
        lead_agent_loop(chat_history)