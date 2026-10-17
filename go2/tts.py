"""Natural speech for Go2's replies with Kokoro-82M (Apache-2.0), generated on this computer.

The model and voices come from Hugging Face at a pinned revision (~330 MB, downloaded once on first use
and cached). Synthesis runs on the CPU, one reply at a time; the web page plays the WAV it returns
and falls back to the browser's own voice if this is unavailable (or still loading).
"""
from __future__ import annotations

import io
import os
import threading
import wave

REPO = "hexgrad/Kokoro-82M"
REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
SAMPLE_RATE = 24000
# The best-rated voices (Kokoro's VOICES.md grades), friendliest first.
VOICES = {
    "af_heart": "Heart · warm (US)", "af_bella": "Bella · bright (US)", "af_nicole": "Nicole · soft (US)",
    "am_puck": "Puck · playful (US)", "am_michael": "Michael · calm (US)", "am_fenrir": "Fenrir · deep (US)",
    "bf_emma": "Emma (UK)", "bm_george": "George (UK)",
}
DEFAULT_VOICE = "af_heart"


class Speaker:
    def __init__(self):
        self.lock = threading.Lock()
        self.model = None
        self.pipelines = {}
        self.error = None
        self.ready = threading.Event()

    def warm_up(self):
        """Load the model in the background so the first reply isn't delayed."""
        threading.Thread(target=self._load, daemon=True, name="tts-load").start()

    def _load(self):
        try:
            with self.lock:
                self._model()
                self._synthesize("Hi.", DEFAULT_VOICE)
            self.ready.set()
        except Exception as error:   # missing package or offline on first use: the browser voice is used instead
            self.error = f"{type(error).__name__}: {error}"

    def _model(self):
        if self.model is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from huggingface_hub import hf_hub_download
            from kokoro import KModel
            config = hf_hub_download(REPO, "config.json", revision=REVISION)
            weights = hf_hub_download(REPO, "kokoro-v1_0.pth", revision=REVISION)
            # CPU on purpose: on an M1 Pro it makes speech ~2.6x faster than real time for new sentences, while
            # the Apple GPU recompiles for every new sentence length and only manages about real time.
            self.model = KModel(repo_id=REPO, config=config, model=weights).eval()
        return self.model

    def _pipeline(self, voice):
        lang = voice[0]                          # 'a' American English, 'b' British English
        if lang not in self.pipelines:
            from kokoro import KPipeline
            self.pipelines[lang] = KPipeline(lang_code=lang, repo_id=REPO, model=self._model())
        return self.pipelines[lang]

    def _synthesize(self, text, voice):
        import numpy as np
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(REPO, f"voices/{voice}.pt", revision=REVISION)
        chunks = [result.audio.detach().cpu().numpy() for result in self._pipeline(voice)(text, voice=path)
                  if result.audio is not None]
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(pcm)
        return buffer.getvalue()

    def wav(self, text, voice=DEFAULT_VOICE, wanted=lambda: True):
        """WAV bytes for `text`. `wanted` is checked once it is this request's turn: the page may have dropped
        the phrase meanwhile, and discarded audio shouldn't delay the next line."""
        if not self.ready.is_set():
            raise RuntimeError(self.error or "the voice is still loading")
        if voice not in VOICES:
            raise ValueError("unknown voice")
        text = " ".join(str(text).split())[:600]
        if not text:
            raise ValueError("nothing to say")
        with self.lock:
            if not wanted():
                raise LookupError("no longer wanted")
            return self._synthesize(text, voice)

    def status(self):
        return {"ready": self.ready.is_set(), "error": self.error, "voices": VOICES, "default": DEFAULT_VOICE}
