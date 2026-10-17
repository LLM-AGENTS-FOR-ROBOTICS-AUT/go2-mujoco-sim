"""Speech recognition for hands-free voice: Whisper large-v3-turbo (MIT), on whatever this computer runs best.

On Apple Silicon it runs on the Mac's GPU through MLX; elsewhere (Windows, Intel Macs) through faster-whisper,
on an NVIDIA GPU when CUDA is available and on the CPU otherwise. The page records what the user says (cut into
sentences where they pause) and posts 16 kHz mono samples; nothing is stored. The model comes from Hugging Face
at a pinned revision (about 1.6 GB, downloaded once and cached) and loads in the background at start-up.
"""
from __future__ import annotations

import importlib.util
import platform
import re
import threading

import numpy as np
from huggingface_hub import snapshot_download

MLX = ("mlx-community/whisper-large-v3-turbo", "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb")
CTRANSLATE = ("dropbox-dash/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf")
SAMPLE_RATE = 16000
MAX_SECONDS = 30
MAX_BYTES = MAX_SECONDS * SAMPLE_RATE * 4          # float32 samples
# What Whisper tends to "hear" in silence or noise (it was trained on captioned video).
HALLUCINATIONS = {"thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching",
                  "you", "bye", "okay", "so", "subtitles by the amara.org community"}
BRACKETED = re.compile(r"\[[^\]]*\]|\([^)]*\)|♪+")


def use_mlx():
    return (platform.system() == "Darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx_whisper") is not None)


class Listener:
    def __init__(self):
        self.lock = threading.Lock()
        self.backend, self.model, self.path = None, None, None
        self.error = None
        self.ready = threading.Event()

    def warm_up(self):
        """Load the model in the background so the first sentence isn't delayed."""
        threading.Thread(target=self._load, daemon=True, name="stt-load").start()

    def _load(self):
        try:
            with self.lock:
                if use_mlx():
                    import mlx.core as mx
                    # MLX keeps freed GPU buffers for reuse; 256 MB of them is as fast as keeping them all
                    # (about 0.9 GB more after one sentence) on an M1 Pro.
                    mx.set_cache_limit(256 << 20)
                    self.backend, self.path = "mlx", snapshot_download(MLX[0], revision=MLX[1])
                else:
                    self._load_ctranslate()
                self._run(np.zeros(SAMPLE_RATE // 2, np.float32))
            self.ready.set()
        except Exception as error:   # missing package or offline on first use: the page says so
            self.error = f"{type(error).__name__}: {error}"

    def _load_ctranslate(self):
        """faster-whisper on an NVIDIA GPU when there is one and its CUDA libraries load, else on the CPU."""
        import ctranslate2
        from faster_whisper import WhisperModel
        path = snapshot_download(CTRANSLATE[0], revision=CTRANSLATE[1])
        if ctranslate2.get_cuda_device_count() > 0:
            try:
                self.model = WhisperModel(path, device="cuda", compute_type="float16")
                self.backend = "faster-whisper (GPU)"
                self._run(np.zeros(SAMPLE_RATE // 2, np.float32))
                return
            except Exception:              # no cuBLAS/cuDNN installed: the CPU still works
                self.model = None
        self.model = WhisperModel(path, device="cpu", compute_type="int8")
        self.backend = "faster-whisper (CPU)"

    def _run(self, audio):
        """A Whisper result as {"segments": [{"text", "no_speech_prob", "avg_logprob"}]}, whichever backend."""
        if self.backend == "mlx":
            import mlx_whisper
            return mlx_whisper.transcribe(audio, path_or_hf_repo=self.path, language="en", temperature=0.,
                                          condition_on_previous_text=False, verbose=None)
        segments, _ = self.model.transcribe(audio, language="en", beam_size=1, temperature=0.,
                                            condition_on_previous_text=False)
        return {"segments": [{"text": s.text, "no_speech_prob": s.no_speech_prob, "avg_logprob": s.avg_logprob}
                             for s in segments]}

    def transcribe(self, audio, wanted=lambda: True):
        """Text for 16 kHz mono float samples ('' for silence). `wanted` is checked once it is this request's
        turn: the page may have moved on meanwhile (a newer partial, or the user cut in)."""
        if not self.ready.is_set():
            raise RuntimeError(self.error or "speech recognition is still loading")
        audio = np.nan_to_num(np.asarray(audio, np.float32))[:MAX_SECONDS * SAMPLE_RATE]
        if len(audio) < SAMPLE_RATE // 5:
            return ""
        with self.lock:
            if not wanted():
                raise LookupError("no longer wanted")
            result = self._run(np.clip(audio, -1, 1))
        return clean(result, float(np.sqrt(np.mean(audio ** 2))))

    def status(self):
        return {"ready": self.ready.is_set(), "error": self.error, "model": "Whisper large-v3-turbo",
                "runs_on": self.backend}


def clean(result, rms):
    """The user's words from a Whisper result, without what it invents for silence and noise."""
    kept = [s for s in result.get("segments", [])
            if not (s.get("no_speech_prob", 0) > .6 and s.get("avg_logprob", 0) < -.5)]
    text = " ".join(BRACKETED.sub(" ", s.get("text", "")) for s in kept)
    text = " ".join(text.split())
    if not re.search(r"[A-Za-z0-9]", text):
        return ""
    quiet = rms < .01 or any(s.get("no_speech_prob", 0) > .2 for s in kept)
    if quiet and text.lower().strip(" .!?,") in HALLUCINATIONS:
        return ""
    return text
