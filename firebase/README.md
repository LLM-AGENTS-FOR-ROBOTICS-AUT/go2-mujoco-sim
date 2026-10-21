# Firebase: a copy of every flagged reply

The command console uploads each reply flagged with the **Bad reply** button to Cloud Firestore in the
`go2-feedback` project (collection `bad_replies`), next to the file it writes in `feedback/`
([`feedback/README.md`](../feedback/README.md)). The code is [`go2/feedback_upload.py`](../go2/feedback_upload.py).

There is no secret in this repository. `web-config.json` is the project's public web config: it says which
project to talk to, not who may do what. The security rules in `firestore.rules` decide that:

| Who | Can |
|---|---|
| A console (signed in anonymously) | Create a flag; update the note of a flag it created. Never read or delete. |
| Anyone else | Nothing |
| The project's owners | Read, export and delete everything, in the Firebase console |

The anonymous sign-in and the list of flags already uploaded are kept outside the repository, in
`~/.go2-console/` (Windows: `C:\Users\<you>\.go2-console\`), or `GO2_STATE_DIR` if set.

## One-time setup (project owner)

Done once for the project, not by every teammate. The console needs nothing but the repository.

1. In the [Firebase console](https://console.firebase.google.com/project/go2-feedback): Firestore database in
   `australia-southeast1`, **Authentication > Sign-in method > Anonymous** enabled, and the project's other
   owners added under **Project settings > Users and permissions**.
2. Install the Firebase CLI (needs [Node.js](https://nodejs.org) LTS), sign in, and deploy the rules from this
   folder:

   ```
   npm install -g firebase-tools
   firebase login
   cd firebase
   firebase deploy --only firestore
   ```

3. Check it from the repository root:

   ```
   .venv/bin/python -m go2.feedback_upload --check          # Windows: .venv\Scripts\python.exe -m go2.feedback_upload --check
   ```

   It signs in, creates and updates one test flag, and confirms that reading and writing as someone else are
   blocked. The test flag (id `00000000-000000-...`, kind `check`) can be deleted in the Firebase console.

Change the rules by editing `firestore.rules` and deploying again. `tests/test_feedback_upload.py` checks that
the rules and the uploader agree on the document's fields.

## Using it

Nothing to do: uploads are on by default. Run `python -m go2.feedback_upload` to send anything in `feedback/` that
is missing right now (the console also does this at every start). Turn uploads off with `--no-upload` on
`command_ui/server.py`, or `GO2_FEEDBACK_UPLOAD=0`. `--mock` never uploads, since its replies are scripted.
