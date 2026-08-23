#!/usr/bin/env python3
"""Local web GUI for launching cli.py's subprograms.

Introspects cli.py's own argparse parsers (see tunables.py -- every
reward/emulator/curriculum/model flag on those parsers already carries its
real default and help text) to build the form the browser renders, and
launches ``python cli.py <command> ...`` as a subprocess, streaming its
output back live. Nothing here hand-duplicates a flag name or default --
add a new argparse argument anywhere cli.py's parsers are built and it shows
up in the GUI automatically.

Usage:
  python gui_app.py [--port 8765] [--no-browser]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import Empty, Queue

sys.path.insert(0, "src")

import cli as _cli  # noqa: E402  (single source of truth for every flag)

_ROOT = Path(__file__).resolve().parent
_INDEX_HTML = _ROOT / "gui_static" / "index.html"


# ---------------------------------------------------------------------------
# argparse introspection -- one generic classifier reused for both the JSON
# schema the browser renders forms from, and the argv builder below, so the
# two can never drift apart.
# ---------------------------------------------------------------------------
def _long_flag(flags: list[str]) -> str:
    """Prefer a --long-form flag (readable in the command preview) over a
    -short one; option_strings order isn't guaranteed to put it first."""
    long_forms = [f for f in flags if f.startswith("--")]
    return long_forms[0] if long_forms else flags[0]


def _classify(action: argparse.Action) -> dict:
    dest = action.dest
    flags = list(action.option_strings)
    help_text = action.help or ""

    if isinstance(action, argparse.BooleanOptionalAction):
        no_flag = next((f for f in flags if f.startswith("--no-")), None)
        yes_flag = next((f for f in flags if not f.startswith("--no-")), flags[0])
        return {
            "dest": dest, "kind": "bool_optional", "flag": yes_flag,
            "no_flag": no_flag, "default": bool(action.default), "help": help_text,
        }
    if action.nargs == 0:  # store_true
        return {
            "dest": dest, "kind": "bool", "flag": _long_flag(flags),
            "default": bool(action.default), "help": help_text,
        }
    if action.nargs == "?":  # bare-flag-or-value (--resume, --migrate, --model)
        return {
            "dest": dest, "kind": "optional", "flag": _long_flag(flags),
            "default": action.default, "const": action.const, "help": help_text,
        }
    if action.nargs in ("+", "*") and action.choices:
        return {
            "dest": dest, "kind": "multi_choice", "flag": _long_flag(flags),
            "choices": list(action.choices), "default": list(action.default or []),
            "help": help_text,
        }
    if action.choices:
        return {
            "dest": dest, "kind": "choice", "flag": _long_flag(flags),
            "choices": list(action.choices), "default": action.default, "help": help_text,
        }
    py_type = action.type
    kind = "int" if py_type is int else "float" if py_type is float else "str"
    default = action.default
    if isinstance(default, (list, tuple)):
        # Fields typed with a custom comma-list parser (--net-arch-pi,
        # --screen-cnn-channels, ...) have a real Python list as their
        # argparse default (e.g. [256, 256]) but render as a plain text
        # input -- send the same comma string the type function itself
        # parses, not a JSON array (str(a_python_list) is "[256, 256]",
        # which the custom parser can't split on commas correctly).
        default = ",".join(str(x) for x in default)
    return {
        "dest": dest, "kind": kind, "flag": _long_flag(flags),
        "default": default, "help": help_text,
        "metavar": action.metavar,
    }


def commands_schema() -> dict:
    out = {}
    for name, sp in _cli.sub.choices.items():
        groups = []
        seen: set[str] = set()
        for group in sp._action_groups:
            args = []
            for action in group._group_actions:
                if not action.option_strings or action.dest == "help":
                    continue
                if action.dest in seen:
                    continue
                seen.add(action.dest)
                args.append(_classify(action))
            if args:
                groups.append({"title": group.title or "options", "args": args})
        out[name] = {"help": sp.description or "", "groups": groups}
    return out


def build_argv(command: str, values: dict) -> list[str]:
    """The single place a submitted form turns into a real command line."""
    if command not in _cli.sub.choices:
        raise ValueError(f"unknown command {command!r}")
    sp = _cli.sub.choices[command]
    # -u: unbuffered stdout -- without it, a piped (non-tty) child buffers
    # in ~4-8KB blocks, so the GUI's live console would sit empty until
    # that buffer fills or the process exits, defeating the point of
    # streaming.
    argv = [sys.executable, "-u", str(_ROOT / "cli.py"), command]
    seen: set[str] = set()
    for group in sp._action_groups:
        for action in group._group_actions:
            if not action.option_strings or action.dest == "help":
                continue
            dest = action.dest
            if dest in seen:
                continue
            seen.add(dest)
            info = _classify(action)
            kind = info["kind"]

            if kind == "bool_optional":
                if dest in values and values[dest] is not None:
                    argv.append(info["flag"] if values[dest] else info["no_flag"])
                continue
            if kind == "bool":
                if values.get(dest):
                    argv.append(info["flag"])
                continue
            if kind == "optional":
                if dest not in values or values[dest] is None:
                    continue
                v = values[dest]
                argv.append(info["flag"])
                if v != "":
                    argv.append(str(v))
                continue
            if kind == "multi_choice":
                vals = values.get(dest)
                if vals:
                    argv.append(info["flag"])
                    argv.extend(str(v) for v in vals)
                continue
            # int / float / str / choice: plain "--flag value"
            v = values.get(dest)
            if v is not None and v != "":
                argv.extend([info["flag"], str(v)])
    return argv


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------
class _Run:
    def __init__(self, argv: list[str]):
        self.id = uuid.uuid4().hex
        self.argv = argv
        self.queue: Queue[str | None] = Queue()
        popen_kwargs = dict(
            cwd=str(_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        self.proc = subprocess.Popen(argv, **popen_kwargs)
        self.done = False
        self.exit_code: int | None = None
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.queue.put(line.rstrip("\n"))
        self.exit_code = self.proc.wait()
        self.done = True
        self.queue.put(None)

    def stop(self) -> None:
        if self.proc.poll() is not None:
            return
        if sys.platform == "win32":
            # terminate()/CTRL_BREAK alone can leave SubprocVecEnv worker
            # processes (multiprocessing spawn, e.g. --workers > 1) running
            # -- taskkill /T walks the whole process tree.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                capture_output=True,
            )
        else:
            import os
            import signal

            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                self.proc.terminate()


_RUNS: dict[str, _Run] = {}
_RUNS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "PokemonRedAI-GUI/1.0"

    def log_message(self, fmt: str, *args) -> None:  # quieter console
        pass

    def _send_json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._serve_index()
        elif path == "/api/commands":
            self._send_json(commands_schema())
        elif path.startswith("/api/stream/"):
            self._stream(path.rsplit("/", 1)[-1])
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/api/run":
            self._start_run()
        elif path.startswith("/api/stop/"):
            self._stop_run(path.rsplit("/", 1)[-1])
        else:
            self.send_error(404)

    def _serve_index(self) -> None:
        try:
            body = _INDEX_HTML.read_bytes()
        except OSError:
            self.send_error(500, "gui_static/index.html missing")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _start_run(self) -> None:
        body = self._read_json()
        command = body.get("command")
        values = body.get("values") or {}
        try:
            argv = build_argv(command, values)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        run = _Run(argv)
        with _RUNS_LOCK:
            _RUNS[run.id] = run
        self._send_json({"run_id": run.id, "argv": argv[1:]})

    def _stop_run(self, run_id: str) -> None:
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if run is None:
            self._send_json({"error": "unknown run_id"}, status=404)
            return
        run.stop()
        self._send_json({"ok": True})

    def _stream(self, run_id: str) -> None:
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
        if run is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                try:
                    line = run.queue.get(timeout=1.0)
                except Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                if line is None:
                    payload = json.dumps({"exit_code": run.exit_code})
                    self.wfile.write(f"event: done\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    break
                payload = json.dumps({"line": line})
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Pokemon Red AI control panel: {url}")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        with _RUNS_LOCK:
            for run in _RUNS.values():
                run.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
