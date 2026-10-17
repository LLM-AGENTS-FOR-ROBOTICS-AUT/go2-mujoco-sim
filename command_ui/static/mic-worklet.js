'use strict';
// Microphone tap for voice commands (runs on the audio thread): hands the page 16 kHz mono samples in 20 ms blocks
// for speech recognition. Nothing is stored or played here.
class Go2MicTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;      // input samples per output sample (3 at 48 kHz)
    this.phase = 0; this.sum = 0; this.count = 0;
    this.block = new Float32Array(320); this.filled = 0;
  }
  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    for (let i = 0; i < input.length; i++) {
      // Average the input samples that fall into each output sample (a simple low-pass for downsampling).
      this.sum += input[i]; this.count++; this.phase += 1;
      if (this.phase >= this.step) {
        this.phase -= this.step;
        this.block[this.filled++] = this.sum / this.count;
        this.sum = 0; this.count = 0;
        if (this.filled === this.block.length) {
          this.port.postMessage(this.block, [this.block.buffer]);
          this.block = new Float32Array(320); this.filled = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('go2-mic-tap', Go2MicTap);
