"""Model download (go2/llm.py download_weights): pinned URL, resume after a dropped connection, checksum."""
import hashlib
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from go2.llm import ModelError, download_weights  # noqa: E402

BLOB = bytes(range(256)) * 4000            # ~1 MB stand-in for the GGUF


class Host(BaseHTTPRequestHandler):
    """Serves BLOB at /model.gguf with Range support (as the Hugging Face download network does)."""
    served = []

    def log_message(self, *_):
        pass

    def do_GET(self):
        start = 0
        if self.headers.get("Range"):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
        body = self.server.blob[start:]
        Host.served.append(start)
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Host)
        self.server.blob = BLOB
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.folder = Path(tempfile.mkdtemp())
        self.spec = {"weights_filename": "model.gguf", "weights_size_bytes": len(BLOB),
                     "weights_sha256": hashlib.sha256(BLOB).hexdigest(),
                     "download_url": f"http://127.0.0.1:{self.server.server_port}/model.gguf"}
        Host.served = []

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_downloads_verifies_and_reports_progress(self):
        seen = []
        path = download_weights(self.spec, self.folder, progress=lambda done, total: seen.append((done, total)),
                                block=100_000)
        self.assertEqual(path.read_bytes(), BLOB)
        self.assertEqual(seen[-1], (len(BLOB), len(BLOB)))
        self.assertGreater(len(seen), 5)
        self.assertFalse((self.folder / "model.gguf.part").exists())

    def test_a_partial_download_is_resumed_not_restarted(self):
        (self.folder / "model.gguf.part").write_bytes(BLOB[:400_000])
        path = download_weights(self.spec, self.folder)
        self.assertEqual(Host.served, [400_000], "asked only for the missing bytes")
        self.assertEqual(path.read_bytes(), BLOB)

    def test_a_corrupted_download_is_deleted_and_never_kept(self):
        self.server.blob = BLOB[:-1] + b"x"
        with self.assertRaises(ModelError):
            download_weights(self.spec, self.folder)
        self.assertFalse((self.folder / "model.gguf").exists())
        self.assertFalse((self.folder / "model.gguf.part").exists())

    def test_a_dropped_connection_keeps_what_arrived(self):
        self.server.blob = BLOB[:300_000]          # the host stops early
        with self.assertRaises(ModelError):
            download_weights(self.spec, self.folder)
        self.assertEqual((self.folder / "model.gguf.part").stat().st_size, 300_000)
        self.server.blob = BLOB
        download_weights(self.spec, self.folder)
        self.assertEqual((self.folder / "model.gguf").read_bytes(), BLOB)


if __name__ == "__main__":
    unittest.main()
