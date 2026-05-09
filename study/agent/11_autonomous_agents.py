#!/usr/bin/env python3
# Harness: autonomy -- models that find work without being told.
"""
s11_autonomous_agents.py - Autonomous Agents

Idle cycle with task board polling, auto-claiming unclaimed tasks, and
identity re-injection after context compression. Builds on s10's protocols.

    Teammate lifecycle:
    +-------+
    | spawn |
    +---+---+
        |
        v
    +-------+  tool_use    +-------+
    | WORK  | <----------- |  LLM  |
    +---+---+              +-------+
        |
        | stop_reason != tool_use
        v
    +--------+
    | IDLE   | poll every 5s for up to 60s
    +---+----+
        |
        +---> check inbox -> message? -> resume WORK
        |
        +---> scan .tasks/ -> unclaimed? -> claim -> resume WORK
        |
        +---> timeout (60s) -> shutdown

    Identity re-injection after compression:
    messages = [identity_block, ...remaining...]
    "You are 'coder', role: backend, team: my-team"

Key insight: "The agent finds work itself."
"""

import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from langfuse.openai import OpenAI  # 注意这里：O 是大写的
from langfuse import observe
# from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# 初始化 OpenAI 客户端
# 注意：如果你使用第三方转发或本地模型，可以通过 base_url 和 api_key 配置
client = OpenAI(
    api_key=os.environ.get("LLM_API_KEY"),
    base_url=os.environ.get("LLM_BASE_URL"),
)
MODEL = os.environ["LLM_MODEL_ID"]

WORKDIR = Path.cwd()
TEAM_DIR = WORKDIR / ".team"
INBOX_DIR = TEAM_DIR / "inbox"
TASKS_DIR = WORKDIR / ".tasks"

POLL_INTERVAL = 5
IDLE_TIMEOUT = 60

SYSTEM = f"You are a team lead at {WORKDIR}. Teammates are autonomous -- they find work themselves."

VALID_MSG_TYPES = {
    "message",
    "broadcast",
    "shutdown_request",
    "shutdown_response",
    "plan_approval_response",
}

# -- Request trackers --
shutdown_requests = {}
plan_requests = {}
_tracker_lock = threading.Lock()
_claim_lock = threading.Lock()


# -- MessageBus: JSONL inbox per teammate with caching --
class MessageBus:
    def __init__(self, inbox_dir: Path):
        self.dir = inbox_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        # Cache for inbox data to reduce file I/O
        self._inbox_cache = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = 0.5  # Cache valid for 0.5 seconds

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
        
        # Invalidate cache for recipient
        with self._cache_lock:
            self._inbox_cache.pop(to, None)
        
        return f"Sent {msg_type} to {to}"

    def read_inbox(self, name: str) -> list:
        # Check cache first
        with self._cache_lock:
            cached = self._inbox_cache.get(name)
            if cached and (time.time() - cached['time']) < self._cache_ttl:
                return cached['messages']
        
        inbox_path = self.dir / f"{name}.jsonl"
        if not inbox_path.exists():
            with self._cache_lock:
                self._inbox_cache[name] = {'messages': [], 'time': time.time()}
            return []
        
        messages = []
        try:
            content = inbox_path.read_text().strip()
            if content:
                for line in content.splitlines():
                    try:
                        messages.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            # Clear inbox after reading
            inbox_path.write_text("")
            
            # Update cache
            with self._cache_lock:
                self._inbox_cache[name] = {'messages': messages, 'time': time.time()}
        except Exception:
            return []
        return messages

    def broadcast(self, sender: str, content: str, teammates: list) -> str:
        count = 0
        for name in teammates:
            if name != sender:
                self.send(sender, name, content, "broadcast")
                count += 1
        return f"Broadcast to {count} teammates"


BUS = MessageBus(INBOX_DIR)


# -- Task board scanning with caching --
_task_cache = {}
_task_cache_lock = threading.Lock()
_last_scan_time = {}


def scan_unclaimed_tasks() -> list:
    """Scan for unclaimed tasks with caching optimization."""
    TASKS_DIR.mkdir(exist_ok=True)
    unclaimed = []
    
    with _task_cache_lock:
        current_time = time.time()
        # Cache valid for 2 seconds to reduce redundant file I/O
        if _task_cache and (current_time - _last_scan_time.get('last', 0)) < 2:
            return [t for t in _task_cache.values() 
                    if t.get("status") == "pending" 
                    and not t.get("owner") 
                    and not t.get("blockedBy")]
        
        # Full scan - read all task files
        task_files = sorted(TASKS_DIR.glob("task_*.json"))
        _task_cache.clear()
        
        for f in task_files:
            try:
                task = json.loads(f.read_text())
                _task_cache[task.get("id")] = task
                if (task.get("status") == "pending"
                        and not task.get("owner")
                        and not task.get("blockedBy")):
                    unclaimed.append(task)
            except:
                continue
        
        _last_scan_time['last'] = current_time
    
    return unclaimed


def invalidate_task_cache():
    """Invalidate task cache after modifications."""
    with _task_cache_lock:
        _task_cache.clear()
        _last_scan_time.clear()


def claim_task(task_id: int, owner: str) -> str:
    with _claim_lock:
        path = TASKS_DIR / f"task_{task_id}.json"
        if not path.exists():
            return f"Error: Task {task_id} not found"
        
        # Read file once and parse
        task_content = path.read_text()
        task = json.loads(task_content)
        
        if task.get("owner"):
            return f"Error: Task {task_id} already claimed by {task.get('owner')}"
        
        task["owner"] = owner
        task["status"] = "in_progress"
        
        # Write updated task
        path.write_text(json.dumps(task, indent=2))
        
        # Invalidate cache to force re-scan
        invalidate_task_cache()
    
    return f"Claimed task #{task_id} for {owner}"


# -- Identity re-injection --
def make_identity_block(name: str, role: str, team_name: str) -> dict:
    return {
        "role": "user",
        "content": f"<identity>You are '{name}', role: {role}, team: {team_name}. Continue your work.</identity>",
    }


# -- Autonomous TeammateManager --
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

    def _set_status(self, name: str, status: str):
        member = self._find_member(name)
        if member:
            member["status"] = status
            self._save_config()

    def spawn(self, name: str, role: str, prompt: str) -> str:
        member = self._find_member(name)
        if member:
            if member["status"] not in ("idle", "shutdown"):
                return f"Error: '{name}' is currently {member['status']}"
            member["status"] = "working"
        else:
            member = {"name": name, "role": role, "status": "working"}
            self.config["members"].append(member)
        self._save_config()
        thread = threading.Thread(target=self._loop, args=(name, role, prompt), daemon=True)
        self.threads[name] = thread
        thread.start()
        return f"Spawned '{name}' (role: {role})"

    @observe(name="teammate_worker_loop")
    def _loop(self, name: str, role: str, prompt: str):
        team_name = self.config["team_name"]
        sys_msg = {"role": "system",
                   "content": f"You are '{name}', role: {role}, team: {team_name}. Use 'idle' tool when no work remains."}
        messages = [sys_msg, {"role": "user", "content": prompt}]
        tools = self._teammate_tools()

        while True:
            # -- WORK PHASE --
            for _ in range(50):
                inbox = BUS.read_inbox(name)
                for msg in inbox:
                    if msg.get("type") == "shutdown_request":
                        self._set_status(name, "shutdown")
                        return
                    messages.append({"role": "user", "content": json.dumps(msg)})

                try:
                    completion = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto"
                    )
                except Exception as e:
                    print(f"[{name}] API Error: {e}")
                    self._set_status(name, "idle")
                    return

                choice = completion.choices[0]
                messages.append(choice.message)  # 必须将完整 message 对象存入历史

                if choice.finish_reason != "tool_calls":
                    break

                idle_requested = False
                for tool_call in choice.message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args = json.loads(tool_call.function.arguments)

                    if fn_name == "idle":
                        idle_requested = True
                        output = "Entering idle phase."
                    else:
                        output = self._exec(name, fn_name, fn_args)

                    print(f"  [{name}] {fn_name}: {str(output)[:100]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": str(output)
                    })

                if idle_requested: break

            # -- IDLE PHASE --
            self._set_status(name, "idle")
            resume = False
            for _ in range(IDLE_TIMEOUT // POLL_INTERVAL):
                time.sleep(POLL_INTERVAL)
                inbox = BUS.read_inbox(name)
                if inbox:
                    for msg in inbox:
                        if msg.get("type") == "shutdown_request":
                            self._set_status(name, "shutdown")
                            return
                        messages.append({"role": "user", "content": json.dumps(msg)})
                    resume = True
                    break

                unclaimed = scan_unclaimed_tasks()
                if unclaimed:
                    task = unclaimed[0]
                    if not claim_task(task["id"], name).startswith("Error:"):
                        messages.append(
                            {"role": "user", "content": f"Task #{task['id']} auto-claimed: {task['subject']}"})
                        resume = True
                        break

            if not resume:
                self._set_status(name, "shutdown")
                return
            self._set_status(name, "working")

    @observe(name="tool_execution")  # 增加此装饰器
    def _exec(self, sender: str, tool_name: str, args: dict) -> str:
        if tool_name == "bash": return _run_bash(args["command"])
        if tool_name == "read_file": return _run_read(args["path"])
        if tool_name == "write_file": return _run_write(args["path"], args["content"])
        if tool_name == "edit_file": return _run_edit(args["path"], args["old_text"], args["new_text"])
        if tool_name == "send_message": return BUS.send(sender, args["to"], args["content"],
                                                        args.get("msg_type", "message"))
        if tool_name == "read_inbox": return json.dumps(BUS.read_inbox(sender), indent=2)
        if tool_name == "claim_task": return claim_task(args["task_id"], sender)
        if tool_name == "plan_approval":
            req_id = str(uuid.uuid4())[:8]
            with _tracker_lock: plan_requests[req_id] = {"from": sender, "status": "pending"}
            BUS.send(sender, "lead", args["plan"], "plan_approval_response", {"request_id": req_id})
            return f"Plan submitted. ID: {req_id}"
        return f"Unknown tool: {tool_name}"

    def _teammate_tools(self) -> list:
        # OpenAI 格式的工具定义
        tool_defs = [
            ("bash", "Run shell command", {"command": "string"}),
            ("read_file", "Read file", {"path": "string"}),
            ("write_file", "Write file", {"path": "string", "content": "string"}),
            ("edit_file", "Replace text", {"path": "string", "old_text": "string", "new_text": "string"}),
            ("send_message", "Message teammate", {"to": "string", "content": "string"}),
            ("read_inbox", "Check messages", {}),
            ("claim_task", "Claim task by ID", {"task_id": "integer"}),
            ("plan_approval", "Submit plan for approval", {"plan": "string"}),
            ("idle", "No more work", {}),
        ]
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": {k: {"type": v} for k, v in params.items()},
                        "required": list(params.keys())
                    }
                }
            } for name, desc, params in tool_defs
        ]

    def list_all(self):
        return "\n".join(
            [f"{m['name']} ({m['role']}): {m['status']}" for m in self.config["members"]]) or "No teammates."


TEAM = TeammateManager(TEAM_DIR)


# -- Utility Functions (Bash/File Ops) with optimizations --
def _safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR): raise ValueError("Path escape!")
    return path


def _run_bash(cmd):
    if any(x in cmd for x in ["rm -rf", "sudo"]): return "Error: Blocked"
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr).strip() or "(no output)"
    except Exception as e:
        return str(e)


# File content cache for read operations
_file_cache = {}
_file_cache_lock = threading.Lock()
_FILE_CACHE_TTL = 1.0  # 1 second cache


def _run_read(p):
    try:
        path = _safe_path(p)
        
        # Check cache first
        with _file_cache_lock:
            cached = _file_cache.get(str(path))
            if cached and (time.time() - cached['time']) < _FILE_CACHE_TTL:
                return cached['content'][:10000]
        
        # Read from disk
        content = path.read_text()
        
        # Update cache
        with _file_cache_lock:
            _file_cache[str(path)] = {'content': content, 'time': time.time()}
        
        return content[:10000]
    except Exception as e:
        return str(e)


def _run_write(p, c):
    try:
        fp = _safe_path(p)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(c)
        
        # Invalidate cache for this file
        with _file_cache_lock:
            _file_cache.pop(str(fp), None)
        
        return "Success"
    except Exception as e:
        return str(e)


def _run_edit(p, o, n):
    try:
        fp = _safe_path(p)
        
        # Read file once
        txt = fp.read_text()
        if o not in txt: 
            return "Text not found"
        
        # Perform replacement
        new_txt = txt.replace(o, n, 1)
        fp.write_text(new_txt)
        
        # Invalidate cache for this file
        with _file_cache_lock:
            _file_cache.pop(str(fp), None)
        
        return "Edited"
    except Exception as e:
        return str(e)


# -- Lead Logic --
LEAD_TOOLS = TEAM._teammate_tools() + [
    {
        "type": "function",
        "function": {
            "name": "spawn_teammate",
            "description": "Spawn an autonomous agent",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "role": {"type": "string"}, "prompt": {"type": "string"}},
                "required": ["name", "role", "prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_teammates",
            "description": "List team status",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

@observe(name="lead_process") # 增加此装饰器
def lead_loop(messages):
    while True:
        inbox = BUS.read_inbox("lead")
        if inbox: messages.append({"role": "user", "content": f"Inbox: {json.dumps(inbox)}"})

        comp = client.chat.completions.create(model=MODEL, messages=messages, tools=LEAD_TOOLS)
        choice = comp.choices[0]
        messages.append(choice.message)

        if choice.finish_reason != "tool_calls":
            if choice.message.content: print(f"\nLead: {choice.message.content}")
            break

        for tool_call in choice.message.tool_calls:
            fn = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if fn == "spawn_teammate":
                res = TEAM.spawn(args["name"], args["role"], args["prompt"])
            elif fn == "list_teammates":
                res = TEAM.list_all()
            else:
                res = TEAM._exec("lead", fn, args)

            print(f"> {fn}: {str(res)[:100]}")
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "name": fn, "content": str(res)})


if __name__ == "__main__":
    history = [{"role": "system", "content": SYSTEM}]
    print("S11 Autonomous Agents (OpenAI) - Type '/tasks', '/team', or a prompt.")
    while True:
        try:
            query = input("\033[36mlead >> \033[0m").strip()
            if not query: continue
            if query.lower() in ("exit", "q"): break

            if query == "/team":
                print(TEAM.list_all())
            elif query == "/tasks":
                for f in sorted(TASKS_DIR.glob("task_*.json")):
                    t = json.loads(f.read_text())
                    print(f"  [{t['status']}] #{t['id']}: {t['subject']} ({t.get('owner', 'unclaimed')})")
            else:
                history.append({"role": "user", "content": query})
                lead_loop(history)
        except KeyboardInterrupt:
            break