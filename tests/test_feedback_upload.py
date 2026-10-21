"""Flagged replies are copied to Firebase (go2/feedback_upload.py): against a fake Firestore that applies the same
rules as firebase/firestore.rules, so nothing here touches the network."""
import json
import re
import sys
import tempfile
import time
import unittest
import unittest.mock
from argparse import Namespace
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "command_ui")]
from go2 import feedback_upload as F  # noqa: E402

CONFIG = {"apiKey": "key", "projectId": "proj", "collection": "bad_replies"}


class FakeFirebase:
    """Firebase Auth and Firestore REST, with the security rules of firebase/firestore.rules."""

    def __init__(self):
        self.docs, self.calls = {}, []
        self.users = 0
        self.online, self.rules_deployed, self.sign_in_enabled = True, True, True
        self.expire_next = False
        self.tokens = {}                                   # id token -> uid

    def _token(self, uid):
        token = f"t{len(self.tokens)}"
        self.tokens[token] = uid
        return token

    def __call__(self, method, url, body=None, headers=None, form=False):
        self.calls.append((method, url))
        if not self.online:
            raise F.UploadError("network: down")
        parsed = urlparse(url)
        if parsed.netloc == "identitytoolkit.googleapis.com":
            if not self.sign_in_enabled:
                raise F.UploadError("400 ADMIN_ONLY_OPERATION", 400, "ADMIN_ONLY_OPERATION")
            self.users += 1
            uid = f"user{self.users}"
            return {"idToken": self._token(uid), "refreshToken": f"r-{uid}", "localId": uid, "expiresIn": "3600"}
        if parsed.netloc == "securetoken.googleapis.com":
            uid = body["refresh_token"][2:]
            return {"id_token": self._token(uid), "refresh_token": body["refresh_token"], "user_id": uid,
                    "expires_in": "3600"}
        token = (headers or {}).get("Authorization", "")[7:]
        if self.expire_next:
            self.expire_next = False
            self.tokens.pop(token, None)
        if token not in self.tokens:
            raise F.UploadError("401 UNAUTHENTICATED", 401, "UNAUTHENTICATED")
        uid = self.tokens[token]
        path = parsed.path.split("/documents/")[1]
        query = parse_qs(parsed.query)
        denied = F.UploadError("403 PERMISSION_DENIED", 403, "PERMISSION_DENIED")
        if method == "GET":
            raise denied                                   # allow read: if false
        if method == "POST":
            flag_id = query["documentId"][0]
            fields = body["fields"]
            if not self.rules_deployed or not self._valid(flag_id, fields, uid):
                raise denied
            if flag_id in self.docs:
                raise F.UploadError("409 ALREADY_EXISTS", 409, "ALREADY_EXISTS")
            self.docs[flag_id] = fields
            return {}
        if method == "PATCH":
            flag_id = path.split("/")[1]
            mask = query["updateMask.fieldPaths"]
            current = self.docs.get(flag_id)
            if current is None:
                raise F.UploadError("404 NOT_FOUND", 404, "NOT_FOUND")
            merged = {**current, **{k: body["fields"][k] for k in mask}}
            changed = {k for k in merged if merged[k] != current.get(k)}
            if (not self.rules_deployed or current["uid"]["stringValue"] != uid or not self._valid(flag_id, merged, uid)
                    or not changed <= {"note", "record", "record_sha256", "updated"}):
                raise denied
            self.docs[flag_id] = merged
            return {}
        raise AssertionError(method)

    @staticmethod
    def _valid(flag_id, fields, uid):
        names = {"id", "created", "model", "prompt_style", "kind", "note", "record", "record_sha256", "uid", "updated"}
        return (set(fields) == names and re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{6}", flag_id)
                and fields["id"]["stringValue"] == flag_id and fields["uid"]["stringValue"] == uid
                and len(fields["note"]["stringValue"]) <= 2000 and "timestampValue" in fields["updated"])


def flag_file(folder, flag_id="20261021-213000-abc123", note=""):
    record = {"id": flag_id, "created": "2026-10-21T21:30:00+1300", "model": "go2-ft-04", "prompt_style": "compact",
              "note": note, "kind": "user", "robot": {"target": "sim", "x": 1.0}, "history": [],
              "input": {"role": "user", "content": "turn left"}, "bad_turn": [{"role": "assistant", "content": "Hi"}]}
    text = json.dumps(record, indent=1)
    (folder / f"{flag_id}.json").write_text(text)
    return flag_id, text


class UploaderTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp())
        self.state = Path(tempfile.mkdtemp())
        self.cloud = FakeFirebase()
        self.logs = []
        self.uploaders = []

    def tearDown(self):
        for uploader in self.uploaders:
            uploader.close()

    def uploader(self, sweep=True, retry_s=(.05,)):
        uploader = F.FeedbackUploader(self.folder, CONFIG, self.state, self.cloud, self.logs.append, retry_s)
        self.uploaders.append(uploader)
        return uploader.start(sweep)

    def settle(self, uploader, condition=lambda: True, timeout=5.):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if uploader.wait_idle(timeout) and condition():
                return True
            time.sleep(.02)
        return False

    def test_a_flag_is_uploaded_with_its_exact_file_text(self):
        uploader = self.uploader(sweep=False)
        flag_id, text = flag_file(self.folder)
        uploader.submit(flag_id)
        self.assertTrue(self.settle(uploader, lambda: flag_id in self.cloud.docs))
        doc = self.cloud.docs[flag_id]
        self.assertEqual(doc["record"]["stringValue"], text, "the training run can write the file back unchanged")
        self.assertEqual(json.loads(doc["record"]["stringValue"]), json.loads((self.folder / f"{flag_id}.json").read_text()))
        self.assertEqual({k: doc[k]["stringValue"] for k in ("id", "model", "prompt_style", "kind", "note", "uid")},
                         {"id": flag_id, "model": "go2-ft-04", "prompt_style": "compact", "kind": "user", "note": "",
                          "uid": "user1"})

    def test_a_later_note_updates_the_same_document(self):
        uploader = self.uploader(sweep=False)
        flag_id, _ = flag_file(self.folder)
        uploader.submit(flag_id)
        self.assertTrue(self.settle(uploader, lambda: flag_id in self.cloud.docs))
        _, text = flag_file(self.folder, note="It should have turned left.")
        uploader.submit(flag_id)
        self.assertTrue(self.settle(uploader, lambda: self.cloud.docs[flag_id]["note"]["stringValue"] != ""))
        self.assertEqual(self.cloud.docs[flag_id]["note"]["stringValue"], "It should have turned left.")
        self.assertEqual(self.cloud.docs[flag_id]["record"]["stringValue"], text)
        self.assertEqual(len(self.cloud.docs), 1)

    def test_nothing_is_sent_twice_across_restarts_and_the_sign_in_is_kept(self):
        flag_id, _ = flag_file(self.folder)
        self.assertTrue(self.settle(self.uploader(), lambda: flag_id in self.cloud.docs))
        before = len(self.cloud.calls)
        self.assertTrue(self.settle(self.uploader()))
        self.assertEqual(len(self.cloud.calls), before, "an unchanged flag is not sent again")
        flag_file(self.folder, "20261021-213500-def456")
        uploader = self.uploader()
        self.assertTrue(self.settle(uploader, lambda: len(self.cloud.docs) == 2))
        self.assertEqual(self.cloud.users, 1, "the same anonymous user, not a new one each start")
        self.assertTrue((self.state / "firebase-auth.json").is_file())
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()),
                         ["20261021-213000-abc123.json", "20261021-213500-def456.json"], "nothing written to feedback/")

    def test_flags_pulled_from_git_that_someone_else_uploaded_are_left_alone(self):
        flag_id, text = flag_file(self.folder)
        self.cloud.docs[flag_id] = F.document_fields(json.loads(text), text, "someone-else")
        uploader = self.uploader()
        self.assertTrue(self.settle(uploader, lambda: flag_id in uploader.ledger))
        self.assertEqual(self.cloud.docs[flag_id]["uid"]["stringValue"], "someone-else")
        before = len(self.cloud.calls)
        self.assertTrue(self.settle(self.uploader()))
        self.assertEqual(len(self.cloud.calls), before)

    def test_offline_flags_are_retried_until_they_go_through(self):
        self.cloud.online = False
        uploader = self.uploader(sweep=False)
        flag_id, _ = flag_file(self.folder)
        uploader.submit(flag_id)
        time.sleep(.2)
        self.assertNotIn(flag_id, self.cloud.docs)
        self.assertTrue(any("offline" in line for line in self.logs))
        self.cloud.online = True
        self.assertTrue(self.settle(uploader, lambda: flag_id in self.cloud.docs))

    def test_refused_uploads_wait_for_the_next_start_instead_of_being_dropped(self):
        self.cloud.rules_deployed = False                   # the database is still in locked production mode
        flag_id, _ = flag_file(self.folder)
        uploader = self.uploader()
        self.assertTrue(self.settle(uploader))
        self.assertNotIn(flag_id, uploader.ledger)
        self.assertTrue(any("rules deployed" in line for line in self.logs))
        self.cloud.rules_deployed = True
        self.assertTrue(self.settle(self.uploader(), lambda: flag_id in self.cloud.docs))

    def test_sign_in_turned_off_is_reported_and_retried_later(self):
        self.cloud.sign_in_enabled = False
        flag_id, _ = flag_file(self.folder)
        self.assertTrue(self.settle(self.uploader()))
        self.assertTrue(any("Anonymous sign-in" in line for line in self.logs))
        self.cloud.sign_in_enabled = True
        self.assertTrue(self.settle(self.uploader(), lambda: flag_id in self.cloud.docs))

    def test_an_expired_token_is_refreshed(self):
        uploader = self.uploader(sweep=False)
        flag_id, _ = flag_file(self.folder)
        self.cloud.expire_next = True
        uploader.submit(flag_id)
        self.assertTrue(self.settle(uploader, lambda: flag_id in self.cloud.docs))
        self.assertTrue(any("securetoken" in url for _, url in self.cloud.calls))

    def test_files_that_are_not_flags_are_skipped(self):
        (self.folder / "notes.json").write_text("{}")
        (self.folder / "20261021-213000-abc123.json").write_text("{not json")
        self.assertTrue(self.settle(self.uploader()))
        self.assertEqual(self.cloud.docs, {})

    def test_the_setup_check_passes_only_with_the_rules_deployed(self):
        lines = []
        self.assertEqual(F.check(lines.append, self.cloud, CONFIG), 0, lines)
        self.assertTrue(lines[-1].startswith("All good"))
        self.assertEqual([d["kind"]["stringValue"] for d in self.cloud.docs.values()], ["check"])
        locked = FakeFirebase()
        locked.rules_deployed = False
        self.assertEqual(F.check(lines.append, locked, CONFIG), 1)
        self.assertIn("firebase deploy", lines[-1])

    def test_the_rules_and_the_uploader_agree_on_the_fields(self):
        rules = (ROOT / "firebase" / "firestore.rules").read_text()
        listed = set(re.findall(r"'(\w+)'", re.search(r"return \[(.*?)\]", rules).group(1)))
        record = json.loads(flag_file(self.folder)[1])
        self.assertEqual(listed, set(F.document_fields(record, "{}", "u")))
        updatable = set(re.findall(r"'(\w+)'", re.search(r"affectedKeys\(\)\.hasOnly\(\[(.*?)\]", rules).group(1)))
        self.assertEqual(updatable, set(F.UPDATABLE))
        config = json.loads((ROOT / "firebase" / "web-config.json").read_text())
        self.assertEqual(json.loads((ROOT / "firebase" / ".firebaserc").read_text())["projects"]["default"],
                         config["projectId"])


try:
    import server
except ImportError:
    server = None


@unittest.skipIf(server is None, "command console requirements not installed")
class ConsoleTests(unittest.TestCase):
    def test_flagging_and_noting_a_reply_submits_it_for_upload(self):
        from tests.fakes import FakeRobot, ScriptedModel, wait_for
        page = "http://127.0.0.1:5050"
        sent = []
        uploader = Namespace(submit=sent.append, close=lambda: None)
        robot = FakeRobot()
        robot.connect()
        console = server.Console(ScriptedModel([("", [("turn", {"direction": "right"})]), "Turning right!"]), robot,
                                 {"scene": "room"}, feedback=Path(tempfile.mkdtemp()), uploader=uploader)
        client = server.make_app(console, 5050).test_client()
        try:
            def post(path, value):
                return client.post(path, json=value, base_url=page, headers={"Origin": page}).get_json()

            def replies():
                entries = client.get("/api/state", base_url=page).get_json()["conversation"]["entries"]
                return [e for e in entries if e["kind"] == "reply"]
            post("/api/messages", {"text": "turn left"})
            self.assertTrue(wait_for(lambda: replies(), 10))
            saved = post("/api/feedback", {"entry": replies()[0]["id"]})
            post("/api/feedback", {"id": saved["id"], "note": "left, not right"})
            self.assertEqual(sent, [saved["id"], saved["id"]])
        finally:
            console.close()

    def test_uploads_are_on_by_default_but_not_for_scripted_replies_or_when_turned_off(self):
        args = Namespace(mock=False, no_upload=False)
        made = []
        original = server.feedback_upload.FeedbackUploader
        server.feedback_upload.FeedbackUploader = lambda folder: Namespace(start=lambda: made.append(folder) or "up")
        try:
            self.assertEqual(server.start_uploader(args), "up")
            self.assertIsNone(server.start_uploader(Namespace(mock=True, no_upload=False)))
            self.assertIsNone(server.start_uploader(Namespace(mock=False, no_upload=True)))
            with unittest.mock.patch.dict("os.environ", {"GO2_FEEDBACK_UPLOAD": "0"}):
                self.assertIsNone(server.start_uploader(args))
        finally:
            server.feedback_upload.FeedbackUploader = original
        self.assertEqual(made, [server.FEEDBACK])


if __name__ == "__main__":
    unittest.main()
