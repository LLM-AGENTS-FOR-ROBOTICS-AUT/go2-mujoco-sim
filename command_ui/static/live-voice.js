'use strict';
// Voice for the command page: Go2's spoken replies (Go2Speaker), the microphone cut into sentences where you
// pause (Go2Utterances), and spoken commands, one at a time or as a hands-free conversation (Go2LiveVoice).
// Everything runs on this computer: the console transcribes with Whisper (/api/stt) and speaks with Kokoro
// (/api/tts). No audio is stored. See command_ui/API.md, "Voice".
const STOP_WORDS = /^\s*(stop|halt|freeze|whoa)\b/i;
// Go2's voice. Its words arrive from the console phrase by phrase while the model is still writing (each turn
// ends with an end marker; see /api/speech). Each phrase's audio comes from the console, which starts making it
// as soon as the phrase is written, and phrases play in order while the next one loads. The browser's own voice
// is the fallback.
// Turns queue up: a turn the user started replaces whatever is playing, while a robot update (a proactive
// turn, e.g. "finished the walk") waits until everything before it has been said.
class Go2Speaker {
  constructor({voice = () => null, onidle = () => {}} = {}) {
    this.voice = voice; this.onidle = onidle;
    this.context = null; this.source = null; this.generation = 0;
    this.turns = [];            // FIFO of {epoch, proactive, phrases: [{text, audio, controller}], ended}
    this.playing = false; this.talking = false; this.lastEpoch = -1; this.minEpoch = 0;
  }
  // Browsers only allow sound after a click or key press; the page calls this on every one.
  unlock() {
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!this.context && Context) this.context = new Context();
    this.context?.resume().catch(() => {});
  }
  // Phrases of a turn that was cut off (Stop, Interrupt, a new message) can still be in flight: ignore them.
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
  // Everything the user starts (a typed message, Stop, Interrupt, Clear, a spoken request) silences Go2,
  // queued robot updates included.
  cancel() { this.minEpoch = this.lastEpoch + 1; this.reset(); }
  reset() {
    this.generation++; this.playing = false;
    for (const turn of this.turns) for (const phrase of turn.phrases) phrase.controller?.abort();
    this.turns = [];
    const source = this.source; this.source = null;
    try { source?.stop(); } catch (_) { /* Already stopped. */ }
    window.speechSynthesis?.cancel();
    this.talking = false;
  }
  pump() {
    if (this.playing) return;
    while (this.turns.length && !this.turns[0].phrases.length && this.turns[0].ended) this.turns.shift();
    const turn = this.turns[0];
    if (!turn) {
      if (this.talking) { this.talking = false; this.onidle?.(); }   // everything queued has been said
      return;
    }
    if (!turn.phrases.length) return;   // more words of this turn are still being written
    const phrase = turn.phrases.shift(), generation = this.generation;
    this.playing = true; this.talking = true;
    const done = () => { if (generation === this.generation) { this.playing = false; this.pump(); } };
    this.play(phrase, generation).then(done, done);
    this.prefetch();
  }
  // Make the audio of the next phrase in line (this turn's, or a queued update's) while one plays.
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
    if (!buffer) return this.browserSay(phrase.text, generation);
    await new Promise(resolve => {
      const source = this.context.createBufferSource();
      source.buffer = buffer; source.connect(this.context.destination); this.source = source;
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
window.Go2Speaker = Go2Speaker;

// Cuts the microphone stream (16 kHz, 20 ms frames) into sentences: one starts when the level rises well above
// the room's noise and ends after a pause. Pure logic (no microphone), so it can be tested on its own.
// Modes per frame: 'listen'; 'barge' while Go2 is talking (louder and longer, so its own voice leaking past echo
// cancellation doesn't count); 'mute' just after Go2 stops (its echo tail).
class Go2Utterances {
  constructor({onStart = () => {}, onPartial = () => {}, onEnd = () => {}} = {}) {
    Object.assign(this, {onStart, onPartial, onEnd});
    this.noise = .005; this.ring = []; this.frames = null; this.loud = 0;
  }
  get inSpeech() { return this.frames !== null; }
  reset() { this.frames = null; this.loud = 0; this.ring = []; }
  feed(frame, mode = 'listen') {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms = Math.sqrt(sum / frame.length);
    if (!this.inSpeech) {
      this.ring.push(frame);
      if (this.ring.length > Go2Utterances.PRE_ROLL) this.ring.shift();
      const threshold = mode === 'barge' ? Math.max(.022, this.noise * 4) : Math.max(.012, this.noise * 3.5);
      if (mode !== 'mute' && rms > threshold) this.loud++;
      else {
        this.loud = 0;
        if (rms < this.noise * 3) this.noise = Math.max(.002, .95 * this.noise + .05 * rms);
      }
      if (this.loud >= (mode === 'barge' ? Go2Utterances.BARGE_FRAMES : Go2Utterances.START_FRAMES)) {
        this.frames = this.ring; this.ring = []; this.voiced = this.loud; this.quiet = 0; this.sincePartial = 0;
        this.onStart(mode);
      }
      return;
    }
    this.frames.push(frame); this.sincePartial++;
    if (rms > Math.max(.008, this.noise * 2.2)) { this.voiced++; this.quiet = 0; } else this.quiet++;
    if (this.sincePartial >= Go2Utterances.PARTIAL_EVERY) { this.sincePartial = 0; this.onPartial(this.audio(0)); }
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
Object.assign(Go2Utterances, {PRE_ROLL: 15, START_FRAMES: 3, BARGE_FRAMES: 8, END_FRAMES: 30, PARTIAL_EVERY: 50,
                              MAX_FRAMES: 1000, MIN_VOICED: 8});
window.Go2Utterances = Go2Utterances;

// Spoken commands. The microphone is cut into sentences in the page and each one is transcribed by Whisper on
// this computer (/api/stt). While the user is still talking the sentence so far is transcribed about once a
// second, for the caption and so "stop" acts without waiting for the end. A sentence the user carries on from
// after a pause, before its transcript is back, is joined to what they say next rather than dropped.
// start({single: true}) takes one command and turns the microphone off again (the mic button); start() keeps
// listening for a hands-free conversation, in which talking over Go2, or while it thinks, cuts it off.
class Go2LiveVoice {
  constructor(callbacks) {
    this.callbacks = callbacks;
    this.active = false; this.single = false; this.session = 0; this.turn = 0; this.utterance = 0;
    this.stream = null; this.audio = null; this.node = null; this.cutter = null;
    this.speaking = false; this.waiting = false; this.phase = 'off';
    this.caption = ''; this.historyCursor = 0; this.interrupting = null;
    this.timers = new Set(); this.held = []; this.lastHeard = 0; this.guardUntil = 0;
    this.partial = null; this.finalsPending = 0; this.carried = '';
  }
  get finalPending() { return this.finalsPending > 0; }
  get supported() {
    return Boolean(navigator.mediaDevices?.getUserMedia && (window.AudioContext || window.webkitAudioContext) &&
      window.AudioWorkletNode && window.fetch);
  }
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
    this.speaking = false; this.held = [];
    this.clearTimer(this.speechTimer); this.clearTimer(this.holdTimer); this.holdTimer = null;
    this.callbacks.speaker.cancel();
  }
  // keepSpeech: the microphone goes off but Go2 still answers aloud (after a single command).
  stop(detail = '', keepSpeech = false) {
    this.active = false; this.session++; this.turn++; this.utterance++;
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
    this.partial?.abort(); this.partial = null; this.finalsPending = 0; this.carried = '';
    if (keepSpeech) { this.speaking = false; this.held = []; } else this.cancelSpeech();
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
    this.caption = ''; this.held = []; this.carried = '';
    this.report('starting', 'Starting the microphone…');
    let audio, stream;
    try {
      for (let waited = 0; !this.callbacks.sttReady(); waited += 500) {
        if (this.callbacks.sttError()) throw new Error(`speech recognition is unavailable (${this.callbacks.sttError()})`);
        if (waited > 180000) throw new Error('speech recognition did not finish loading');
        this.report('starting', 'Loading speech recognition (Whisper, the first start takes a while)…');
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
      this.fail(error.name === 'NotAllowedError' ? 'Microphone access was blocked: allow it for this page in the browser.' :
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
    // While Go2 talks (a single command can start while it still answers the last one), only a louder voice counts.
    const talking = this.speaking || (this.single && this.callbacks.speaker.talking);
    const mode = talking ? 'barge' : performance.now() < this.guardUntil ? 'mute' : 'listen';
    this.cutter.feed(samples, mode);
    if (this.cutter.inSpeech) this.lastHeard = performance.now();
  }
  heard(mode) {
    this.utterance++; this.lastHeard = performance.now();
    this.partial?.abort(); this.partial = null;
    this.clearTimer(this.idleTimer);
    // Talking over Go2, or while it is still thinking, cuts in.
    if (mode === 'barge' || this.waiting || this.callbacks.isBusy()) this.interrupt();
    else this.report('listening', 'Hearing you…');
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
      if (!this.active || session !== this.session || utterance !== this.utterance || !this.cutter?.inSpeech) return;
      this.caption = text;
      // Safety: act on "stop" as soon as it is heard, without waiting for the end of the sentence.
      if (STOP_WORDS.test(text)) return this.stopWord(text);
      this.report(this.interrupting ? 'interrupting' : 'listening', this.interrupting ? 'Cutting in…' : 'Hearing you…');
    } catch (_) { /* Superseded or failed: the final transcription still follows. */ }
    finally { if (this.partial === controller) this.partial = null; }
  }
  async transcribeFinal(samples) {
    this.partial?.abort(); this.partial = null;
    if (!samples) {                                   // a click or bump, not speech
      if (this.carried && !this.finalPending) return this.flush(this.takeCarried(''));
      if (!this.finalPending) this.releaseHeld();
      if (this.single) this.awaitSpeech();
      return;
    }
    const utterance = this.utterance, session = this.session;
    this.finalsPending++;
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
    } finally { if (session === this.session) this.finalsPending = Math.max(0, this.finalsPending - 1); }
    if (!this.active || session !== this.session) return;
    if (STOP_WORDS.test(text)) return this.stopWord(text);
    if (utterance !== this.utterance) {
      // They carried on talking before this came back: it belongs with what they're saying now (or, if that
      // was only a noise and nothing else is on its way, it is sent on its own).
      if (text) this.carried = this.takeCarried(text);
      if (this.carried && !this.cutter?.inSpeech && !this.finalPending) this.flush(this.takeCarried(''));
      return;
    }
    text = this.takeCarried(text);
    if (!text) {
      this.caption = ''; this.releaseHeld();
      if (this.single) this.awaitSpeech();
      this.report('listening', this.single ? "Didn't catch that, say it again" : 'Listening');
      return;
    }
    this.flush(text);
  }
  // What was carried over, then `text`: one sentence ("in this room. and tell me" reads "in this room and tell me").
  takeCarried(text) {
    const before = this.carried;
    this.carried = '';
    if (!before || !text) return before || text;
    return /^[a-z]/.test(text) ? `${before.replace(/[.…]+$/, '')} ${text}` : `${before} ${text}`;
  }
  stopWord(text) {
    // Safety: act on "stop" immediately, even mid-sentence and while speaking.
    this.utterance++; this.cutter?.reset();
    this.partial?.abort(); this.partial = null;
    this.caption = ''; this.carried = ''; this.cancelSpeech(); this.waiting = true;
    this.callbacks.onStopWord(text);
    if (this.single) return this.stop('', true);
    this.report('listening', 'Stopping…');
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
    this.speechTimer = this.later(() => this.fail('Go2\'s voice stopped responding. Click Live to start again.'), 60000);
    speaker.say(item.epoch, item.text, item.proactive);
  }
  // Called by the speaker when everything queued (including robot updates) has been said.
  speechDone() {
    if (!this.active || !this.speaking) return;
    this.clearTimer(this.speechTimer); this.speaking = false;
    this.guardUntil = performance.now() + 300;      // the tail of Go2's own voice is not the user
    this.caption = '';
    this.report('listening', 'Listening · your turn');
  }
}
window.Go2LiveVoice = Go2LiveVoice;
