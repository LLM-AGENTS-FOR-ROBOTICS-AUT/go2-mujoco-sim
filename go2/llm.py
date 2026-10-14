"""App-owned Ollama process and chat client with native tool calling.

The console starts a private `ollama serve` on a random loopback port whose model
store only links to the checksum-pinned weights from models/catalog.json, so
the user's own Ollama daemon and its sessions are never touched.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

from . import procs

ROOT = Path(__file__).resolve().parents[1]
CATALOG = json.loads((ROOT / "models/catalog.json").read_text())


GEMMA_QUOTE = '<|"|>'


def _gemma_value(raw):
    raw = raw.strip()
    if raw.startswith(GEMMA_QUOTE) and raw.endswith(GEMMA_QUOTE) and len(raw) >= 2 * len(GEMMA_QUOTE):
        return raw[len(GEMMA_QUOTE):-len(GEMMA_QUOTE)]
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        return raw[1:-1]
    if raw in ("true", "false"):
        return raw == "true"
    try:
        number = float(raw)
        return int(number) if number.is_integer() and "." not in raw else number
    except ValueError:
        return raw


def recover_calls(content, names):
    """Gemma sometimes writes its native call syntax as plain text (name{key:value,...}) without the
    tool-call tokens, so Ollama does not parse it. Recover such calls for known tool names only."""
    import re
    pattern = re.compile(r"(?:<\|tool_call>)?\s*(?:call:)?\b(" + "|".join(map(re.escape, sorted(names))) +
                         r")\{([^{}]*)\}\s*(?:<tool_call\|>)?")
    calls, spans = [], []
    for match in pattern.finditer(content):
        body, args = match[2], {}
        parts, current, in_quote = [], "", False
        i = 0
        while i < len(body):
            if body.startswith(GEMMA_QUOTE, i):
                in_quote = not in_quote
                current += GEMMA_QUOTE
                i += len(GEMMA_QUOTE)
                continue
            ch = body[i]
            if ch == "," and not in_quote:
                parts.append(current)
                current = ""
            else:
                current += ch
            i += 1
        if current.strip():
            parts.append(current)
        ok = True
        for part in parts:
            if ":" not in part:
                ok = False
                break
            key, value = part.split(":", 1)
            args[key.strip().strip('"')] = _gemma_value(value)
        if ok:
            calls.append({"name": match[1], "arguments": args, "recovered": True})
            spans.append(match.span())
    for start, end in reversed(spans):
        content = content[:start] + content[end:]
    return calls, content.strip()


class ModelError(RuntimeError):
    pass


class Cancelled(ModelError):
    pass


def _api(base, path, value=None, timeout=3):
    request = urllib.request.Request(base + path, data=None if value is None else json.dumps(value).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


class OllamaModel:
    def __init__(self, choice=None, options=None, prompt_style=None):
        choice = choice or CATALOG["default"]
        if choice not in CATALOG["models"]:
            raise ModelError("Unknown model option.")
        if prompt_style not in (None, "full", "compact"):
            raise ModelError("Unknown prompt style.")
        self.choice = choice
        self.spec = CATALOG["models"][choice]
        self.model_id = self.spec["alias"]
        # "full": the long system prompt and tool descriptions; "compact": the short ones the fine-tunes learned.
        self.prompt_style = prompt_style or self.spec.get("prompt_style", "full")
        self.options = {"temperature": .4, "top_p": .95, "num_ctx": 8192, "num_predict": 320, **(options or {})}
        self.process = None
        self.base = None
        self.loaded = False
        self._verified = None
        self.directory = ROOT / "runtime" / ("model-" + uuid.uuid4().hex[:12])
        self.lock = threading.RLock()
        self.connection = None

    # ---- weights -------------------------------------------------------------
    def _weights(self):
        configured = os.environ.get(self.spec["weights_env"])
        if configured:
            path = Path(configured).expanduser()
            if not path.is_file():
                raise ModelError(f"{self.spec['weights_env']} points to a missing file.")
            return path.resolve()
        for path in (ROOT / "models" / self.spec["weights_filename"],
                     Path.home() / ".ollama/models/blobs" / ("sha256-" + self.spec["weights_sha256"])):
            if path.is_file():
                return path.resolve()
        raise ModelError(f"{self.spec['label']} weights are missing. Set {self.spec['weights_env']} to the GGUF.")

    def check(self):
        with self.lock:
            if os.environ.get("GO2_OLLAMA_URL"):     # a shared server owns the weights (see _start)
                return {"model": self.model_id, "label": self.spec["label"], "ready": True, "loaded": self.loaded}
            path = self._weights()
            stat = path.stat()
            signature = (str(path), stat.st_size, stat.st_mtime_ns)
            if signature != self._verified:
                if stat.st_size != self.spec["weights_size_bytes"]:
                    raise ModelError(f"{self.spec['label']} weights have the wrong size.")
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != self.spec["weights_sha256"]:
                        raise ModelError("Model weights checksum differs; refusing to load.")
                self._verified = signature
            if not shutil.which("ollama"):
                raise ModelError("Ollama is not installed.")
            return {"model": self.model_id, "label": self.spec["label"], "ready": True, "loaded": self.loaded}

    # ---- process -------------------------------------------------------------
    def _start(self, cancelled):
        with self.lock:
            shared = os.environ.get("GO2_OLLAMA_URL")
            if shared:
                # One server for several processes (parallel evaluations on one GPU); it is never stopped here.
                self.base = shared.rstrip("/")
                return self.base
            if self.process is not None and self.process.poll() is None:
                return self.base
            self.check()
            store = self.directory / "models"
            (store / "blobs").mkdir(parents=True, exist_ok=True)
            manifest = (ROOT / "models" / self.spec["manifest"]).read_bytes()
            if hashlib.sha256(manifest).hexdigest() != self.spec["manifest_sha256"]:
                raise ModelError("Pinned Ollama model manifest changed.")
            descriptor = json.loads(manifest)
            for item in [descriptor["config"]] + descriptor["layers"]:
                name = item["digest"].replace(":", "-")
                source = self._weights() if item["mediaType"] == "application/vnd.ollama.image.model" else ROOT / "models" / name
                target = store / "blobs" / name
                if not target.exists():
                    procs.link(target, source.resolve())
            target = store / self.spec["manifest_store_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(manifest)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            self.base = f"http://127.0.0.1:{port}"
            env = dict(os.environ, OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=str(store), OLLAMA_NO_CLOUD="1",
                       OLLAMA_KEEP_ALIVE="30m")
            log = open(self.directory / "ollama.log", "wb")
            self.process = subprocess.Popen([shutil.which("ollama"), "serve"], env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=log, **procs.detached())
            (self.directory / "server.pid").write_text(str(self.process.pid))
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if cancelled():
                    raise Cancelled("cancelled during model startup")
                if self.process.poll() is not None:
                    raise ModelError("The app-owned Ollama process exited; see runtime logs.")
                try:
                    models = _api(self.base, "/api/tags", timeout=.5)["models"]
                    if not any(m["name"] == self.model_id for m in models):
                        raise ModelError("The private model store does not list the selected model.")
                    return self.base
                except (OSError, TimeoutError, ValueError):
                    time.sleep(.1)
            raise ModelError("The app-owned Ollama did not become ready in 15 seconds.")

    def chat(self, messages, tools=None, cancelled=lambda: False, on_text=None, timeout=60):
        """One model turn. Returns {"content": str, "tool_calls": [{"name", "arguments"}], "stats": {...}}."""
        base = self._start(cancelled)
        port = int(base.rsplit(":", 1)[1])
        body = {"model": self.model_id, "messages": messages, "stream": True, "think": False,
                "keep_alive": "30m", "options": self.options}
        if tools:
            body["tools"] = tools
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        started = time.monotonic()
        content, calls, final = "", [], {}
        try:
            connection.connect()
            with self.lock:
                if cancelled():
                    raise Cancelled("cancelled before inference")
                self.connection = connection
            connection.request("POST", "/api/chat", body=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            if response.status != 200:
                raise ModelError(f"Model returned HTTP {response.status}: {response.read()[:300].decode(errors='replace')}")
            for line in response:
                if cancelled():
                    raise Cancelled("cancelled during inference")
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise ModelError(chunk["error"])
                message = chunk.get("message", {})
                piece = message.get("content", "")
                if piece:
                    content += piece
                    if on_text:
                        on_text(content)
                for call in message.get("tool_calls") or []:
                    fn = call.get("function", {})
                    arguments = fn.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except ValueError:
                            arguments = {"_unparsed": arguments}
                    calls.append({"name": fn.get("name"), "arguments": arguments})
                if chunk.get("done"):
                    final = chunk
                    break
        except (OSError, http.client.HTTPException) as error:
            if cancelled():
                raise Cancelled("cancelled") from None
            raise ModelError(f"Model connection failed: {error}") from None
        finally:
            with self.lock:
                if self.connection is connection:
                    self.connection = None
            connection.close()
        self.loaded = True
        return {"content": content.strip(), "tool_calls": calls, "stats": {
            "seconds": round(time.monotonic() - started, 2), "prompt_tokens": final.get("prompt_eval_count"),
            "output_tokens": final.get("eval_count"), "done_reason": final.get("done_reason")}}

    def cancel(self):
        with self.lock:
            connection, self.connection = self.connection, None
        if connection is not None:
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()

    def unload(self):
        self.cancel()
        with self.lock:
            process, self.process, self.base, self.loaded = self.process, None, None, False
        if process is not None:
            procs.stop_tree(process)
            # Only links to the weights and a log live here; the weights themselves are untouched.
            shutil.rmtree(self.directory, ignore_errors=True)
        return {"unloaded": True}


def model_for(spec, options=None):
    """'go2-ft-04' or, for prompt comparisons, 'base@compact' / 'go2-ft-04@full' (a model with the other prompt)."""
    choice, _, style = spec.partition("@")
    return OllamaModel(choice, options, style or None)
