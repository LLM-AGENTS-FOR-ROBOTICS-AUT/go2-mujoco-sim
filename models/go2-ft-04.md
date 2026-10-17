---
license: apache-2.0
base_model: google/gemma-4-E2B-it
language: en
tags:
- gguf
- robotics
- tool-calling
- unitree-go2
- ollama
---

# go2-ft-04

A fine-tune of Google's Gemma 4 E2B IT that controls a **Unitree Go2** robot dog through tool calls: it
walks, turns, sits, does tricks, looks around and finds objects when you ask in plain English, and
explains what happened. Made by AUT Project 661 (LLM Agents for Robotics).

It's the language model of the command console in
[go2-mujoco-sim](https://github.com/LLM-AGENTS-FOR-ROBOTICS-AUT/go2-mujoco-sim), which downloads this
file on its first start, checks its SHA-256 and runs it with a private Ollama process. Every tool call
the model makes passes the console's safety executor (per-call limits and LiDAR clearance checks)
before the robot moves.

| | |
|---|---|
| File | `go2-ft-04.gguf`, Q4_K_M, 3,416,119,840 bytes |
| SHA-256 | `15a875b610d8640e4d004d71e3888ba73a7647c4eddf9b91b4df95929498293d` |
| Base model | google/gemma-4-E2B-it (LoRA fine-tune, merged, then quantized to 4-bit) |
| Prompt and tools | The compact system prompt and tool definitions in the repo (`go2/agent.py`, `go2/tools.py`) |
| Result | 182 of 186 (98%) on the project's tool-calling benchmark |

## Intended use

Research and teaching with the Go2 in simulation, and supervised tests on a real Go2 EDU through the
console. Not approved for production, and not for driving a robot without the console's safety executor,
a supervisor and an emergency stop.

## Licence

Apache 2.0, as for Gemma 4 (see Google's [model card](https://ai.google.dev/gemma/docs/core/model_card_4)).
