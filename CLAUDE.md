# Two-account handoff rules

Two Claude accounts work on this repo. Either one can run out of usage at any time.
The repo is the only shared memory. Chat history is NOT shared.

## At the start of every session
1. Pull the latest from GitHub.
2. Read `HANDOFF.md` in full before you do anything else.
3. Continue from "Next steps". Do not redo work listed under "Done".

## During work
1. Work on the branch named in `HANDOFF.md`. Never push to the same branch from both accounts at once.
2. After each meaningful step, update `HANDOFF.md`, then commit and push.
3. Keep commits small. Write clear commit messages.
4. Never leave important state only in the chat. If it matters, it goes in `HANDOFF.md`.

## HANDOFF.md must always hold
- Updated: date/time (Pacific) and which account (A = personal, B = bertholomus)
- Branch
- Doing now
- Done since last handoff
- Next steps, in order
- Open questions / blockers
- Gotchas: things that broke, decisions made and why
