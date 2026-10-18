# Flagged replies

Each file here is one of Go2's replies that someone flagged with the **Bad reply** button in the command
console (`command_ui/`). The console writes `<date>-<time>-<id>.json` when the button is pressed, and adds the
optional note ("What should Go2 have done?") to the same file.

Commit these files and push them, so they reach the next fine-tuning run. The training pipeline redoes each
flagged turn: it puts the simulated robot back where it was, restores the conversation before it, has the
teacher model answer the same message again (reading the note), and trains on the new turn.

A file holds:

| Field | What it is |
|---|---|
| `id`, `created` | The flag's id and when it was made |
| `model`, `prompt_style` | The model that gave the reply, and the prompt it was run with |
| `note` | What the person flagging it said Go2 should have done (may be empty) |
| `kind` | `user` (a reply to the user) or `robot_update` (Go2 reporting on its own) |
| `robot` | Where the robot was when the turn started: `target` (`sim` or `go2`), `scene`, `x`, `y`, `yaw`, `posture`, what it was `doing` and had `queued`, `armed` |
| `history` | The model's conversation before the turn |
| `input` | The message that started the turn |
| `bad_turn` | What the model did: its tool calls, their results, and its reply |
| `transcript` | The same turn as the page showed it |

They contain the conversation's text (never audio). Read one before committing it if the conversation might
have included something private.
