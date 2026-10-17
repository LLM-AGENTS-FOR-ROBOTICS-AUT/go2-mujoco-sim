'use strict';
// Voice for the command page: Go2's spoken replies (Go2Speaker), the microphone cut into sentences where you
// pause (Go2Utterances), and spoken commands, one at a time or as a hands-free conversation (Go2LiveVoice).
// Everything runs on this computer: the console transcribes with Whisper (/api/stt) and speaks with Kokoro
// (/api/tts). No audio is stored. See command_ui/API.md, "Voice".
const STOP_WORDS = /^\s*(stop|halt|freeze|whoa)\b/i;

// Go2's voice. Its words arrive from the console phrase by phrase while the model is still writing (each turn
// ends with an end marker; see /api/speech). Each phrase's audio comes from the console, which starts making it
// as soon as the phrase is written, and phrases play in order while the next one loads. The browser's own voice
// is the fallback. A turn the user started replaces whatever is playing; a robot update (a proactive turn, such
// as "finished the walk") waits until everything before it has been said.
class Go2Speaker {
  constructor({voice = () => null, onidle = () => {}, onstart = () => {}} = {}) {
    this.voice = voice; this.onidle = onidle; this.onstart = onstart;
    this.context = null; this.gain = null; this.source = null; this.generation = 0; this.ducked = false;
    this.turns = [];            // in order: {epoch, proactive, phrases: [{text, audio, controller}], ended}
    this.playing = false; this.talking = false; this.lastEpoch = -1; this.minEpoch = 0;
  }
  // Browsers only allow sound after a click or key press; the page calls this on every one.
  unlock() {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!this.context && Context) {
      this.context = new Context();
      this.gain = this.context.createGain(); this.gain.connect(this.context.destination);
    }
    this.context?.resume().catch(() => {});
  }
  // Quieter while the user may be cutting in, back to full volume if it was only a noise.
  duck(on) {
    if (on === this.ducked) return;
    this.ducked = on;
    if (this.gain) this.gain.gain.setTargetAtTime(on ? Go2Speaker.DUCKED : 1, this.context.currentTime, .02);
    if (window.speechSynthesis?.speaking) { if (on) window.speechSynthesis.pause(); else window.speechSynthesis.resume(); }
  }
  // Phrases of a turn that was cut off (Stop, an interruption, a new message) can still be in flight: ignore them.
  accepts(epoch) { return typeof epoch !== 'number' || epoch >= this.minEpoch; }
  say(epoch, text, proactive = false) {
    if (!this.accepts(epoch)) return;
    if (typeof epoch === 'number') this.lastEpoch = Math.max(this.lastEpoch, epoch);
    let turn = this.turns.find(t => t.epoch === epoch);
    if (!turn) {
      if (!proactive) this.reset();
      turn = {epoch, proactive, phrases: [], ended: false};
      this.turns.push(turn);
    }
    turn.phrases.push({text});
    if (this.playing) this.prefetch(); else this.pump();
  }
  end(epoch) {
    const turn = this.turns.find(t => t.epoch === epoch);
    if (turn) { turn.ended = true; this.pump(); }
  }
  // Everything the user starts (a message, Stop, an interruption, Clear) silences Go2, queued updates included.
  cancel() { this.minEpoch = this.lastEpoch + 1; this.reset(); }
  reset() {
    this.generation++; this.playing = false;
    for (const turn of this.turns) for (const phrase of turn.phrases) phrase.controller?.abort();
    this.turns = [];
    const source = this.source; this.source = null;
    try { source?.stop(); } catch (_) { /* Already stopped. */ }
    window.speechSynthesis?.cancel();
    this.duck(false);
    if (this.talking) { this.talking = false; this.onidle(); }
  }
  pump() {
    if (this.playing) return;
    while (this.turns.length && !this.turns[0].phrases.length && this.turns[0].ended) this.turns.shift();
    const turn = this.turns[0];
    if (!turn) {
      if (this.talking) { this.talking = false; this.onidle(); }   // everything queued has been said
      return;
    }
    if (!turn.phrases.length) return;   // more words of this turn are still being written
    const phrase = turn.phrases.shift(), generation = this.generation;
    this.playing = true; this.talking = true;
    const done = () => { if (generation === this.generation) { this.playing = false; this.pump(); } };
    this.play(phrase, generation).then(done, done);
    this.prefetch();
  }
  // Ask for the audio of the next phrase in line (this turn's, or a queued update's) while one plays.
  prefetch() {
    const next = this.turns.flatMap(turn => turn.phrases)[0];
    if (next && !next.audio) this.load(next);
  }
  load(phrase) {
    phrase.controller = new AbortController();
    phrase.audio = this.fetchAudio(phrase.text, phrase.controller.signal);
    phrase.audio.catch(() => {});
    return phrase.audio;
  }
  async fetchAudio(text, signal) {
    if (!this.context) throw new Error('sound not unlocked yet');
    const response = await fetch('/api/tts', {method: 'POST', signal, headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, voice: this.voice()})});
    if (!response.ok || response.status === 204) throw new Error((await response.json().catch(() => ({}))).error || response.statusText);
    return this.context.decodeAudioData(await response.arrayBuffer());
  }
  async play(phrase, generation) {
    let buffer = null;
    try { buffer = await (phrase.audio || this.load(phrase)); } catch (_) { buffer = null; }
    if (generation !== this.generation) return;
    this.onstart(phrase.text);
    if (!buffer) return this.browserSay(phrase.text, generation);
    await new Promise(resolve => {
      const source = this.context.createBufferSource();
      source.buffer = buffer; source.connect(this.gain); this.source = source;
      source.onended = resolve; source.start();
    });
  }
  browserSay(text, generation) {
    return new Promise(resolve => {
      if (!window.speechSynthesis || generation !== this.generation) return resolve();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = navigator.language || 'en-US';
      const voices = window.speechSynthesis.getVoices().filter(v => v.lang?.startsWith('en'));
      utterance.voice = voices.find(v => /premium|enhanced|natural|neural/i.test(v.name)) || voices.find(v => /google/i.test(v.name)) || null;
      utterance.onend = resolve; utterance.onerror = resolve;
      window.speechSynthesis.speak(utterance);
    });
  }
}
Go2Speaker.DUCKED = .12;
window.Go2Speaker = Go2Speaker;

// Cuts the microphone stream (16 kHz, 20 ms frames) into sentences: one starts when the level rises well above
// the room's noise and ends after a pause. Pure logic (no microphone), so it can be tested on its own.
// Modes per frame: 'listen'; 'barge' while Go2 is talking (louder, so its own voice leaking past echo cancellation
// doesn't count, and counted over a short window because spoken words have gaps); 'mute' just after Go2 stops
// (its echo tail).
class Go2Utterances {
  constructor({onStart = () => {}, onPartial = () => {}, onEnd = () => {}} = {}) {
    Object.assign(this, {onStart, onPartial, onEnd});
    this.noise = .005; this.ring = []; this.recent = []; this.frames = null; this.loud = 0;
  }
  get inSpeech() { return this.frames !== null; }
  reset() { this.frames = null; this.loud = 0; this.ring = []; this.recent = []; }
  feed(frame, mode = 'listen') {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    if (!this.inSpeech) {
      this.ring.push(frame);
      if (this.ring.length > Go2Utterances.PRE_ROLL) this.ring.shift();
      const barge = mode === 'barge';
      const loud = mode !== 'mute' && rms > (barge ? Math.max(.022, this.noise * 4) : Math.max(.012, this.noise * 3.5));
      this.recent.push(loud);
      if (this.recent.length > Go2Utterances.BARGE_WINDOW) this.recent.shift();
      if (loud) this.loud++;
      else {
        this.loud = 0;
        if (rms < this.noise * 3) this.noise = Math.max(.002, .95 * this.noise + .05 * rms);
      }
      const count = barge ? this.recent.filter(Boolean).length : this.loud;
      if (count >= (barge ? Go2Utterances.BARGE_FRAMES : Go2Utterances.START_FRAMES)) {
        this.frames = this.ring; this.ring = []; this.recent = [];
        this.voiced = count; this.quiet = 0; this.sincePartial = 0; this.partials = 0;
        this.onStart(mode);
      }
      return;
    }
    this.frames.push(frame); this.sincePartial++;
    if (rms > Math.max(.008, this.noise * 2.2)) { this.voiced++; this.quiet = 0; } else this.quiet++;
    if (this.sincePartial >= (this.partials ? Go2Utterances.PARTIAL_EVERY : Go2Utterances.FIRST_PARTIAL)) {
      this.sincePartial = 0; this.partials++; this.onPartial(this.audio(0));
    }
    if (this.quiet >= Go2Utterances.END_FRAMES || this.frames.length >= Go2Utterances.MAX_FRAMES) {
      const audio = this.audio(Math.max(0, this.quiet - 10)), voiced = this.voiced;   // keep 200 ms of the pause
      this.frames = null; this.loud = 0;
      this.onEnd(voiced >= Go2Utterances.MIN_VOICED ? audio : null);
    }
  }
  audio(dropLast) {
    const frames = this.frames.slice(0, this.frames.length - dropLast);
    const out = new Float32Array(frames.reduce((n, f) => n + f.length, 0));
    let at = 0;
    for (const f of frames) { out.set(f, at); at += f.length; }
    return out;
  }
}
// In 20 ms frames: 300 ms kept from before a sentence starts, 60 ms of voice to start one (while Go2 talks, 120 ms
// of voice within 200 ms), a 600 ms pause ends it, a partial transcript 400 ms in (so a short "stop" is caught
// while it is said) and then about once a second, 20 s at most, and 160 ms of voice to count.
Object.assign(Go2Utterances, {PRE_ROLL: 15, START_FRAMES: 3, BARGE_FRAMES: 6, BARGE_WINDOW: 10, END_FRAMES: 30,
                              FIRST_PARTIAL: 20, PARTIAL_EVERY: 50, MAX_FRAMES: 1000, MIN_VOICED: 8});
window.Go2Utterances = Go2Utterances;

// Spoken commands. The microphone is cut into sentences in the page and each one is transcribed by Whisper on
// this computer. While the user is still talking, the sentence so far is transcribed about once a second, for
// the caption and so "stop" acts without waiting for the end.
// start({single: true}) takes one command and turns the microphone off again (the mic button); start() keeps
// listening for a hands-free conversation, in which talking over Go2, or while it thinks, cuts it off: Go2's
// voice drops at once, and the reply is dropped once it is clear the user is speaking (CONFIRM_FRAMES of voice,
// or a whole sentence). A cough or a bump only dips Go2's voice for a moment.
class Go2LiveVoice {
  constructor(callbacks) {
    this.callbacks = callbacks;
    this.active = false; this.single = false; this.session = 0; this.turn = 0; this.utterance = 0;
    this.stream = null; this.audio = null; this.node = null; this.cutter = null;
    this.speaking = false; this.waiting = false; this.phase = 'off';
    this.caption = ''; this.historyCursor = 0; this.interrupting = null;
    this.timers = new Set(); this.held = []; this.lastHeard = 0; this.guardUntil = 0;
    this.partial = null; this.finalPending = false; this.tentative = false;
  }
  get supported() {
    return Boolean(navigator.mediaDevices?.getUserMedia && (window.AudioContext || window.webkitAudioContext) &&
      window.AudioWorkletNode && window.fetch);
  }
  get talking() { return this.speaking || this.callbacks.speaker.talking; }
  report(phase, detail) {
    this.phase = phase;
    this.callbacks.onState({active: this.active, single: this.single, phase, detail, caption: this.caption});
  }
  later(callback, delay) {
    const session = this.session;
    const timer = setTimeout(() => { this.timers.delete(timer); if (this.active && session === this.session) callback(); }, delay);
    this.timers.add(timer); return timer;
  }
  clearTimer(timer) { clearTimeout(timer); this.timers.delete(timer); }
  cancelSpeech() {
    this.speaking = false; this.held = []; this.tentative = false;
    this.clearTimer(this.speechTimer); this.clearTimer(this.holdTimer); this.holdTimer = null;
    this.callbacks.speaker.cancel();
  }
  // keepSpeech: the microphone goes off but Go2 still answers aloud (after a single command).
  stop(detail = '', keepSpeech = false) {
    this.active = false; this.session++; this.turn++; this.utterance++;
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
    this.partial?.abort(); this.partial = null; this.finalPending = false;
    if (keepSpeech) { this.speaking = false; this.held = []; this.tentative = false; this.callbacks.speaker.duck(false); }
    else this.cancelSpeech();
    try { this.node?.disconnect(); } catch (_) { /* Already disconnected. */ }
    this.node = null; this.cutter = null;
    this.stream?.getTracks().forEach(track => track.stop()); this.stream = null;
    if (this.audio) this.audio.close().catch(() => {});
    this.audio = null; this.interrupting = null;
    this.waiting = false; this.caption = '';
    this.callbacks.onLevel(0); this.report('off', detail);
  }
  fail(message) { this.stop(message); this.callbacks.onError(message); }
  async start({single = false} = {}) {
    if (this.active) return;
    if (!this.supported) return this.fail('Voice needs a browser with microphone and Web Audio support (Chrome, Edge, Safari or Firefox).');
    this.active = true; this.single = single; const session = ++this.session;
    this.turn++; this.historyCursor = this.callbacks.historyLength();
    this.caption = ''; this.held = [];
    this.report('starting', 'Starting the microphone…');
    let audio, stream;
    try {
      for (let waited = 0; !this.callbacks.sttReady(); waited += 500) {
        if (this.callbacks.sttError()) throw new Error(`speech recognition is unavailable (${this.callbacks.sttError()})`);
        if (waited > 180000) throw new Error('speech recognition did not finish loading');
        this.report('starting', 'Loading speech recognition (Whisper, first start only takes a while)…');
        await new Promise(resolve => setTimeout(resolve, 500));
        if (!this.active || this.session !== session) return;
      }
      audio = new (window.AudioContext || window.webkitAudioContext)();
      this.audio = audio;
      await audio.resume();
      stream = await navigator.mediaDevices.getUserMedia({audio: {
        echoCancellation: true, noiseSuppression: true, autoGainControl: true}, video: false});
      if (!this.active || this.session !== session) {
        stream.getTracks().forEach(track => track.stop()); await audio.close(); return;
      }
      this.stream = stream;
      const track = stream.getAudioTracks()[0];
      if (!track || track.readyState === 'ended') throw new Error('No live microphone audio track.');
      track.addEventListener('ended', () => { if (this.active && session === this.session) this.fail('Microphone disconnected.'); });
      await audio.audioWorklet.addModule('/static/mic-worklet.js');
      if (!this.active || this.session !== session) return;
      this.cutter = new Go2Utterances({onStart: mode => this.heard(mode), onPartial: samples => this.transcribePartial(samples),
                                       onEnd: samples => this.transcribeFinal(samples)});
      const node = new AudioWorkletNode(audio, 'go2-mic-tap'), mute = audio.createGain();
      mute.gain.value = 0;           // keeps the tap running; nothing reaches the speakers
      audio.createMediaStreamSource(stream).connect(node);
      node.connect(mute).connect(audio.destination);
      node.port.onmessage = event => this.frame(event.data, session);
      this.node = node;
      if (single) this.awaitSpeech();
      this.report('listening', single ? 'Listening… say your command' : 'Listening · speak naturally, pause when done');
    } catch (error) {
      stream?.getTracks().forEach(track => track.stop());
      if (!this.active || session !== this.session) { audio?.close().catch(() => {}); return; }
      const denied = error.name === 'NotAllowedError';
      this.fail(denied ? 'Microphone access was blocked: allow it for this page in the browser.' :
        `The microphone could not start: ${error.message || error.name}`);
    }
  }
  // A single command: give up when nothing is said for a while.
  awaitSpeech() {
    this.clearTimer(this.idleTimer);
    this.idleTimer = this.later(() => {
      if (!this.cutter?.inSpeech && !this.finalPending) this.stop("Didn't hear anything.");
    }, 8000);
  }
  frame(samples, session) {
    if (!this.active || session !== this.session || !this.cutter) return;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    this.callbacks.onLevel(Math.min(1, Math.sqrt(sum / samples.length) * 8));
    const mode = this.talking ? 'barge' : performance.now() < this.guardUntil ? 'mute' : 'listen';
    this.cutter.feed(samples, mode);
    if (this.cutter.inSpeech) {
      this.lastHeard = performance.now();
      if (this.tentative && this.cutter.voiced >= Go2LiveVoice.CONFIRM_FRAMES) this.cutIn();
    }
  }
  heard(mode) {
    this.utterance++; this.lastHeard = performance.now();
    this.partial?.abort(); this.partial = null;
    this.clearTimer(this.idleTimer);
    // Talking over Go2, or while it is still thinking, may be cutting in: quieten Go2 now, decide shortly.
    if (mode === 'barge' || this.talking || this.waiting || this.callbacks.isBusy()) {
      this.tentative = true; this.callbacks.speaker.duck(true);
    }
    this.report('listening', 'Hearing you…');
  }
  cutIn() {
    if (!this.tentative) return;
    this.tentative = false;
    this.interrupt();
  }
  async stt(samples, signal) {
    const response = await fetch('/api/stt', {method: 'POST', body: samples, signal,
      headers: {'Content-Type': 'application/octet-stream'}});
    if (response.status === 204) throw new Error('superseded');
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || `speech recognition failed (${response.status})`);
    return (result.text || '').trim();
  }
  async transcribePartial(samples) {
    if (this.partial || this.finalPending) return;     // one at a time; the final one matters more
    const controller = new AbortController(), utterance = this.utterance, session = this.session;
    this.partial = controller;
    try {
      const text = await this.stt(samples, controller.signal);
      if (!this.active || session !== this.session || utterance !== this.utterance) return;
      // Safety: act on "stop" as soon as it is heard, without waiting for the end of the sentence (or for the
      // whole sentence's transcript, if it has just ended).
      if (STOP_WORDS.test(text)) return this.stopWord(text);
      if (!this.cutter?.inSpeech) return;
      this.caption = text;
      this.report(this.interrupting ? 'interrupting' : 'listening', this.interrupting ? 'Cutting in…' : 'Hearing you…');
    } catch (_) { /* Superseded or failed: the final transcription still follows. */ }
    finally { if (this.partial === controller) this.partial = null; }
  }
  async transcribeFinal(samples) {
    // A partial still in flight is left to finish: the console is already working on it (the whole sentence
    // waits its turn anyway), and it may be a "stop".
    if (!samples) {                                   // a click or bump, not speech
      if (this.tentative) {
        this.tentative = false; this.callbacks.speaker.duck(false);
        this.report(this.speaking ? 'speaking' : this.waiting ? 'thinking' : 'listening',
          this.speaking ? 'Go2 is talking · speak to cut in' : this.waiting ? 'Thinking · speak to cut in' : 'Listening');
      }
      if (!this.finalPending) this.releaseHeld();
      if (this.single) this.awaitSpeech();
      return;
    }
    this.cutIn();                                     // a whole sentence: they meant to cut in
    const utterance = this.utterance, session = this.session;
    this.finalPending = true;
    this.report('listening', 'Understanding…');
    let text;
    try {
      text = await this.stt(samples);
    } catch (error) {
      if (this.active && session === this.session) {
        if (this.single) return this.fail(`Speech recognition: ${error.message}`);
        this.callbacks.onError(`Speech recognition: ${error.message}`); this.report('listening', 'Listening');
      }
      return;
    } finally { if (session === this.session) this.finalPending = false; }
    if (!this.active || session !== this.session || utterance !== this.utterance) return;
    if (!text) {
      this.caption = ''; this.releaseHeld();
      if (this.single) this.awaitSpeech();
      this.report('listening', this.single ? "Didn't catch that, say it again" : 'Listening');
      return;
    }
    if (STOP_WORDS.test(text)) return this.stopWord(text);
    this.flush(text);
  }
  stopWord(text) {
    // Safety: act on "stop" immediately, even mid-sentence and while Go2 is talking.
    this.utterance++; this.cutter?.reset();
    this.partial?.abort(); this.partial = null;
    this.caption = ''; this.cancelSpeech(); this.waiting = false;
    this.callbacks.onStopWord(text);
    if (this.single) return this.stop('', true);
    this.report('listening', 'Stopped · listening');
  }
  async flush(text) {
    if (!this.active) return;
    this.held = []; this.clearTimer(this.holdTimer);   // their sentence starts a new turn; a held update is stale
    this.caption = text;
    const session = this.session, turn = ++this.turn;
    try {
      if (this.interrupting) await this.interrupting;
      if (!this.active || session !== this.session || turn !== this.turn) return;
      if (this.single) {
        this.stop('', true);                          // the microphone goes off; Go2 still answers aloud
        await this.callbacks.onUtterance(text);
        return;
      }
      this.waiting = true; this.report('thinking', 'Thinking · speak to cut in');
      const accepted = await this.callbacks.onUtterance(text);
      if (accepted === false && this.active && session === this.session && turn === this.turn) {
        this.waiting = false; this.report('listening', 'Listening');
      }
    } catch (error) {
      if (this.active && session === this.session && turn === this.turn) this.fail(`Voice command failed: ${error.message}`);
    }
  }
  async interrupt() {
    if (!this.active || this.interrupting) return this.interrupting;
    this.turn++; this.cancelSpeech(); this.waiting = false; this.caption = '';
    const session = this.session;
    this.report('interrupting', 'Cutting in…');
    const pending = this.callbacks.onInterrupt();
    this.interrupting = pending;
    try {
      await pending;
      if (this.active && session === this.session) {
        this.historyCursor = this.callbacks.historyLength();
        this.report('listening', this.cutter?.inSpeech ? 'Hearing you…' : 'Listening');
      }
    } catch (error) {
      if (this.active && session === this.session) this.fail(`Interruption failed: ${error.message}`);
    } finally { if (this.interrupting === pending) this.interrupting = null; }
  }
  // The conversation's entries (from the state): a failed reply ends the wait for it.
  observe(entries) {
    if (!this.active) return;
    if (entries.length < this.historyCursor) this.historyCursor = 0;     // the conversation was cleared
    const fresh = entries.slice(this.historyCursor);
    this.historyCursor = entries.length;
    const failure = fresh.filter(entry => entry.kind === 'error').at(-1);
    if (failure && this.waiting) { this.waiting = false; this.callbacks.onError(failure.text); this.report('listening', 'Listening · the last reply failed'); }
  }
  // Go2's words arrive phrase by phrase while the model is still writing, so it starts talking right away.
  onSpeech(item) {
    if (!this.active || this.interrupting || !this.callbacks.speaker.accepts(item.epoch)) return;
    // A robot update is not spoken over the user's own sentence: it waits until they stop, and is dropped if
    // what they said becomes a new request.
    if (this.held.some(h => h.epoch === item.epoch) || (item.proactive && !item.end && this.userSpeaking())) {
      this.held.push(item);
      if (!this.holdTimer) this.holdTimer = this.later(() => this.checkHeld(), 300);
      return;
    }
    if (this.held.length && !this.userSpeaking()) this.releaseHeld();
    this.deliver(item);
  }
  userSpeaking() { return Boolean(this.cutter?.inSpeech) || this.finalPending || performance.now() - this.lastHeard < 1200; }
  checkHeld() {
    this.holdTimer = null;
    if (!this.held.length) return;
    if (this.userSpeaking()) this.holdTimer = this.later(() => this.checkHeld(), 300); else this.releaseHeld();
  }
  releaseHeld() {
    this.clearTimer(this.holdTimer); this.holdTimer = null;
    const held = this.held; this.held = [];
    for (const item of held) this.deliver(item);
  }
  deliver(item) {
    const speaker = this.callbacks.speaker;
    if (!speaker.accepts(item.epoch)) return;
    if (item.end) return speaker.end(item.epoch);
    if (!this.speaking) {
      this.waiting = false; this.speaking = true;
      this.report('speaking', 'Go2 is talking · speak to cut in');
    }
    this.clearTimer(this.speechTimer);
    this.speechTimer = this.later(() => this.speechDone(), 60000);    // never stay deaf if playback hangs
    speaker.say(item.epoch, item.text, item.proactive);
  }
  // Called by the speaker when everything queued (robot updates included) has been said.
  speechDone() {
    if (!this.active) return;
    this.clearTimer(this.speechTimer);
    this.guardUntil = performance.now() + 300;      // the tail of Go2's own voice is not the user
    if (!this.speaking) return;
    this.speaking = false; this.caption = '';
    this.report('listening', 'Listening · your turn');
  }
}
Go2LiveVoice.CONFIRM_FRAMES = 15;                   // 300 ms of voice
window.Go2LiveVoice = Go2LiveVoice;
