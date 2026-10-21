"""Upload flagged replies (feedback/<id>.json, feedback/README.md) to Cloud Firestore, so every flag reaches the
next training run even if nobody commits it.

The local file stays the source of truth and is still committed to git; this is an extra copy. Each flag becomes
one document, bad_replies/<id>, holding the file's exact text in `record` (so the training pipeline can write it
back as feedback/<id>.json unchanged) plus a few top-level fields to browse by in the Firebase console.

There is no secret in the repository. The console signs in to Firebase anonymously with the public web config
(firebase/web-config.json), and the security rules (firebase/firestore.rules) only let it create flags and
update the note on flags it created itself: it can never read or delete anything. Only the project's owners can
read the flags, in the Firebase console.

Uploads run on a background thread and never hold up the console. When the network is down they are retried,
and the next console start sends anything still missing (including flags pulled from git that nobody has
uploaded yet). The anonymous sign-in and the list of uploaded flags are kept outside the repository, in
~/.go2-console/ (or GO2_STATE_DIR). Turn uploads off with --no-upload or GO2_FEEDBACK_UPLOAD=0.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_CONFIG = ROOT / "firebase" / "web-config.json"
FLAG_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
MAX_RECORD = 900_000          # characters; a Firestore document is at most 1 MiB
RETRY_S = (15, 60, 300, 900)  # waits between attempts while the network is down
TIMEOUT_S = 15


def enabled_by_env():
    return os.environ.get("GO2_FEEDBACK_UPLOAD", "1").strip().lower() not in ("0", "false", "no", "off")


def state_dir():
    return Path(os.environ.get("GO2_STATE_DIR") or Path.home() / ".go2-console")


class UploadError(RuntimeError):
    def __init__(self, message, status=None, reason=None):
        super().__init__(message)
        self.status, self.reason = status, reason      # HTTP status and Google's error status, e.g. ALREADY_EXISTS

    @property
    def offline(self):
        return self.status is None or self.status >= 500 or self.status == 429


def http(method, url, body=None, headers=None, form=False):
    """One HTTP request; returns the JSON answer or raises UploadError."""
    data = None
    headers = dict(headers or {})
    if body is not None:
        if form:
            data = urllib.parse.urlencode(body).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            detail = json.loads(error.read() or b"{}").get("error", {})
        except ValueError:
            detail = {}
        reason = detail.get("status") or detail.get("message")
        raise UploadError(f"{error.code} {reason or error.reason}", error.code, reason) from None
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise UploadError(f"network: {error}") from None


def _string(value):
    return {"stringValue": "" if value is None else str(value)}


def document_fields(record, text, uid):
    """The Firestore document for one flag: the file's exact text plus fields to browse by."""
    return {
        "id": _string(record["id"]),
        "created": _string(record.get("created")),
        "model": _string(record.get("model")),
        "prompt_style": _string(record.get("prompt_style")),
        "kind": _string(record.get("kind")),
        "note": _string(record.get("note")),
        "record": _string(text),
        "record_sha256": _string(hashlib.sha256(text.encode()).hexdigest()),
        "uid": _string(uid),
        "updated": {"timestampValue": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")},
    }


UPDATABLE = ("note", "record", "record_sha256", "updated")   # what the rules let an update change


class FeedbackUploader:
    def __init__(self, folder, config=None, state=None, transport=http, log=print, retry_s=RETRY_S):
        self.folder = Path(folder)
        if config is None:
            config = json.loads(WEB_CONFIG.read_text())
        self.api_key, self.project = config["apiKey"], config["projectId"]
        self.collection = config.get("collection", "bad_replies")
        self.state = Path(state) if state else state_dir()
        self.transport, self.log, self.retry_s = transport, log, retry_s
        self.auth_file = self.state / "firebase-auth.json"
        self.ledger_file = self.state / f"uploaded-{self.project}.json"
        self.ledger = self._read(self.ledger_file)        # flag id -> sha256 of the text last sent (or settled)
        self.auth = self._read(self.auth_file)
        self.jobs = queue.Queue()
        self.failures = 0
        self.closed = threading.Event()
        self.thread = None

    # ---- the public side: called by the console -----------------------------------------------------------
    def start(self, sweep=True):
        """Start the background thread; with sweep, also send every flag in the folder not uploaded yet."""
        self.thread = threading.Thread(target=self._worker, daemon=True, name="feedback-upload")
        self.thread.start()
        if sweep:
            self.jobs.put("*")
        return self

    def submit(self, flag_id):
        """Upload (or update) one flag soon; returns at once."""
        self.jobs.put(flag_id)

    def close(self):
        self.closed.set()
        self.jobs.put(None)

    def wait_idle(self, timeout=10.):
        """For tests and scripts: wait until everything queued has been tried."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.jobs.unfinished_tasks == 0:
                return True
            time.sleep(.02)
        return False

    # ---- the background thread ----------------------------------------------------------------------------
    def _worker(self):
        while True:
            job = self.jobs.get()
            try:
                if job is None:
                    return
                pending = sorted(p.stem for p in self.folder.glob("*.json")) if job == "*" else [job]
                waiting = [flag for flag in pending if self._sync(flag) is False]
                if waiting and not self.closed.is_set():
                    self._retry_later(waiting)
            finally:
                self.jobs.task_done()

    def _retry_later(self, flags):
        delay = self.retry_s[min(self.failures, len(self.retry_s) - 1)]
        self.failures += 1
        self.log(f"Firebase upload: offline, will retry {len(flags)} flagged repl{'y' if len(flags) == 1 else 'ies'}"
                 f" in {delay:g} s.")

        def again():
            if not self.closed.wait(delay):
                for flag in flags:
                    self.jobs.put(flag)
        threading.Thread(target=again, daemon=True, name="feedback-upload-retry").start()

    def _sync(self, flag_id):
        """Bring one flag's document up to date. True: settled (sent, or never sendable). False: offline, retry
        soon. None: refused for now (rules not deployed, sign-in turned off): left for the next console start."""
        if not FLAG_ID.fullmatch(flag_id):
            return True
        path = self.folder / f"{flag_id}.json"
        try:
            text = path.read_text(encoding="utf-8")
            record = json.loads(text)
        except (OSError, ValueError):
            return True                                    # gone or half-written; a later save submits it again
        digest = hashlib.sha256(text.encode()).hexdigest()
        if self.ledger.get(flag_id) == digest:
            return True
        if not isinstance(record, dict) or record.get("id") != flag_id or len(text) > MAX_RECORD:
            self.log(f"Firebase upload: skipped {flag_id} (not a flag file, or too large).")
            return self._settle(flag_id, digest)
        try:
            uid = self._uid()
        except UploadError as error:
            if error.offline:
                return False
            self.log(f"Firebase upload: can't sign in ({error}); is Anonymous sign-in enabled? Trying again at the "
                     "next start.")
            return None
        fields = document_fields(record, text, uid)
        try:
            if flag_id in self.ledger:
                self._patch(flag_id, fields)
            else:
                try:
                    self._create(flag_id, fields)
                except UploadError as error:
                    if error.reason != "ALREADY_EXISTS":
                        raise
                    try:
                        self._patch(flag_id, fields)       # ours from a run whose ledger wasn't saved...
                    except UploadError as refused:
                        if refused.status != 403:
                            raise
                        return self._settle(flag_id, digest)   # ...or another person's: already in the database
        except UploadError as error:
            if error.offline:
                return False
            if error.status in (401, 403):
                self.log(f"Firebase upload: {flag_id} refused ({error}); are the rules deployed? Trying again at "
                         "the next start.")
                return None
            self.log(f"Firebase upload: {flag_id} failed ({error}); it stays in git only.")
            return self._settle(flag_id, digest)
        self.failures = 0
        self.log(f"Firebase upload: sent flagged reply {flag_id}.")
        return self._settle(flag_id, digest)

    def _settle(self, flag_id, digest):
        self.ledger[flag_id] = digest
        self._write(self.ledger_file, self.ledger)
        return True

    # ---- Firestore and Firebase Auth REST calls --------------------------------------------------------------
    def _documents(self):
        return f"https://firestore.googleapis.com/v1/projects/{self.project}/databases/(default)/documents"

    def _headers(self):
        return {"Authorization": f"Bearer {self.auth['id_token']}"}

    def _create(self, flag_id, fields):
        url = f"{self._documents()}/{self.collection}?documentId={flag_id}"
        self._authorised("POST", url, {"fields": fields})

    def _patch(self, flag_id, fields):
        mask = "&".join(f"updateMask.fieldPaths={name}" for name in UPDATABLE)
        url = f"{self._documents()}/{self.collection}/{flag_id}?{mask}&currentDocument.exists=true"
        self._authorised("PATCH", url, {"fields": {name: fields[name] for name in UPDATABLE}})

    def _authorised(self, method, url, body):
        try:
            return self.transport(method, url, body, self._headers())
        except UploadError as error:
            if error.status != 401:
                raise
        self._refresh(force=True)                          # the token expired early: once more with a new one
        return self.transport(method, url, body, self._headers())

    def _uid(self):
        self._refresh()
        return self.auth["uid"]

    def _refresh(self, force=False):
        """A valid ID token for this computer's anonymous Firebase user (signing up the first time)."""
        auth = self.auth
        if not force and auth.get("id_token") and auth.get("expires", 0) - time.time() > 120:
            return
        if auth.get("refresh_token"):
            try:
                answer = self.transport("POST", f"https://securetoken.googleapis.com/v1/token?key={self.api_key}",
                                        {"grant_type": "refresh_token", "refresh_token": auth["refresh_token"]},
                                        None, True)
                self._keep(answer["id_token"], answer["refresh_token"], answer["user_id"], answer["expires_in"])
                return
            except UploadError as error:
                if error.offline:
                    raise                                  # keep the user; sign up again only if it was refused
        answer = self.transport("POST", f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={self.api_key}",
                                {"returnSecureToken": True}, None)
        self._keep(answer["idToken"], answer["refreshToken"], answer["localId"], answer["expiresIn"])

    def _keep(self, id_token, refresh_token, uid, expires_in):
        if self.auth.get("uid") and self.auth["uid"] != uid:
            self.ledger.clear()                            # a new user can't update the old one's flags
            self._write(self.ledger_file, self.ledger)
        self.auth = {"id_token": id_token, "refresh_token": refresh_token, "uid": uid,
                     "expires": time.time() + float(expires_in)}
        self._write(self.auth_file, self.auth)

    # ---- small files outside the repository ---------------------------------------------------------------
    @staticmethod
    def _read(path):
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, path, value):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, indent=1))
            try:
                os.chmod(temporary, 0o600)                 # the sign-in is only this person's business
            except OSError:
                pass
            temporary.replace(path)
        except OSError as error:
            self.log(f"Firebase upload: could not save {path.name} ({error}).")


# ---- command line: check the setup, or send the folder now ---------------------------------------------------
def check(log=print, transport=http, config=None):
    """Prove the deployed rules work: sign in, create and update a test flag, and fail to read it back.
    Leaves one document with kind "check" (id 00000000-000000-...) that an owner may delete."""
    import secrets
    import tempfile
    folder = Path(tempfile.mkdtemp())
    flag_id = f"00000000-000000-{secrets.token_hex(3)}"
    record = {"id": flag_id, "created": datetime.now(timezone.utc).isoformat(), "model": "setup-check",
              "prompt_style": "", "note": "", "kind": "check"}
    uploader = FeedbackUploader(folder, config, state=folder, transport=transport, log=lambda *_: None)
    results = []

    def step(name, ok, detail=""):
        results.append(ok)
        log(f"  {'ok  ' if ok else 'FAIL'} {name}{f' ({detail})' if detail else ''}")

    log(f"Checking Firebase project {uploader.project}:")
    try:
        uid = uploader._uid()
        step("anonymous sign-in", True)
    except UploadError as error:
        step("anonymous sign-in", False, f"{error}; enable Anonymous in Authentication > Sign-in method")
        return 1
    text = json.dumps(record, indent=1)
    try:
        uploader._create(flag_id, document_fields(record, text, uid))
        step("create a flag", True)
    except UploadError as error:
        step("create a flag", False, f"{error}; deploy the rules: cd firebase, then firebase deploy --only firestore")
        return 1
    record["note"] = "setup check"
    try:
        uploader._patch(flag_id, document_fields(record, json.dumps(record, indent=1), uid))
        step("update its note", True)
    except UploadError as error:
        step("update its note", False, str(error))
    try:
        uploader._authorised("GET", f"{uploader._documents()}/{uploader.collection}/{flag_id}", None)
        step("reading is blocked", False, "anyone could read the flags: deploy firebase/firestore.rules")
    except UploadError as error:
        step("reading is blocked", error.status == 403, str(error))
    try:
        bad = document_fields(record, text, "someone-else")
        uploader._create(f"00000000-000000-{secrets.token_hex(3)}", bad)
        step("writing as someone else is blocked", False, "the rules accepted it")
    except UploadError as error:
        step("writing as someone else is blocked", error.status == 403, str(error))
    log("All good: flagged replies will be uploaded." if all(results) else "Something is wrong; see above.")
    return 0 if all(results) else 1


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Copy flagged replies (feedback/) to Firebase.")
    parser.add_argument("--check", action="store_true", help="check sign-in and the deployed security rules")
    parser.add_argument("--folder", default=str(ROOT / "feedback"), help="the flagged-reply folder to send now")
    args = parser.parse_args(argv)
    if args.check:
        return check()
    uploader = FeedbackUploader(args.folder).start()
    uploader.close()                                   # after the sweep: ends once everything has been tried
    uploader.thread.join(120)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
