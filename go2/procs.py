"""Starting and stopping the helper processes (the private Ollama, the simulator) on macOS, Linux and Windows."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys

WINDOWS = sys.platform == "win32"


def detached():
    """Popen arguments that keep a helper out of the console's Ctrl+C and let it be stopped as a group."""
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if WINDOWS else {"start_new_session": True}


def stop_tree(process, timeout=3):
    """Stop a helper and everything it started (Ollama starts a model runner of its own)."""
    if process is None or process.poll() is not None:
        return
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def command_of(pid):
    """The command line of a running process, or "" if there's none with that id."""
    if WINDOWS:
        listed = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                                capture_output=True, text=True).stdout
        return listed if str(pid) in listed else ""
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def stop_pid(pid):
    """Stop a helper (and what it started) by process id: one left behind by an earlier run that was killed."""
    if WINDOWS:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def link(target, source):
    """Make `target` stand for `source` without copying when possible: a symlink, else (Windows without
    Developer Mode) a hard link, else a copy. Removing `target` never touches `source`."""
    try:
        target.symlink_to(source)
    except OSError:
        try:
            os.link(source, target)
        except OSError:
            shutil.copyfile(source, target)
