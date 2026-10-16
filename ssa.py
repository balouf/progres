"""Small and Stupid Agent (SSA): a LLM, one tool, a loop.

Works with any OpenAI-compatible endpoint: Ollama (local), Gemini, Mistral, OpenRouter...
Only URL, KEY and MODEL change.
"""
import json
import subprocess
import sys
import urllib.request

URL = "http://localhost:11434/v1/chat/completions"
KEY = "none"  # Ollama ignores it
MODEL = "gemma4:e4b"

TOOLS = [{"type": "function", "function": {
    "name": "run_python",
    "description": "Execute a Python script and return what it prints (stdout and stderr).",
    "parameters": {"type": "object",
                   "properties": {"code": {"type": "string", "description": "Python code"}},
                   "required": ["code"]}}}]


def run_python(code, cwd="."):
    r = subprocess.run([sys.executable, "-c", code], cwd=cwd,
                       capture_output=True, text=True, timeout=60)
    return (r.stdout + r.stderr)[-3000:] or "(no output)"


def llm(messages):
    data = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS,
                       "max_tokens": 3000}).encode()  # budget: a reasoning model can loop forever
    req = urllib.request.Request(URL, data, {"Content-Type": "application/json",
                                             "Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req))["choices"][0]["message"]


def agent(task, cwd=".", ask=False, steps=10, system=None):
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": task})
    for _ in range(steps):
        msg = llm(messages)
        messages.append(msg)
        if not msg.get("tool_calls"):         # no tool call: the agent has finished
            return msg["content"]
        for call in msg["tool_calls"]:
            args = call["function"]["arguments"]
            code = json.loads(args)["code"] if isinstance(args, str) else args["code"]
            print(f"--- run_python:\n{code}")
            if ask and input("Run it? [y/N] ") != "y":
                out = "The user refused to run this code."
            else:
                out = run_python(code, cwd)
            print(f"--- output:\n{out}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": out})
    return "(too many steps)"
