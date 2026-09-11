# 08: Extract persistent conversation state into Session

**What to build:** Move the state that survives between customer turns behind a single Session abstraction without changing customer-visible behavior or the production interpretation flow.

**Blocked by:** None. The implemented behavior from Tickets 01–06 is the migration baseline.

**Status:** ready-for-human

## Primary acceptance criteria

- [x] Session owns the draft order, transcript, stable line-ID counter, draft revision, review and confirmation state, submission state, and general order instructions.
- [x] FoodOrderAgent remains the public owner of a Session and continues to expose the same send behavior.
- [x] Existing deterministic menu validation, pricing, draft mutation, clarification, review, confirmation, and submission behavior is unchanged.
- [x] Existing injected interpreter, submission service, logger, and retry-policy boundaries remain compatible.
- [x] Creating two agents creates independent sessions with no shared mutable state.
- [x] No TurnProcessor, operation Outcome protocol, provider tool calls, or alternate production path is introduced.

## Tests included

- [x] First add a failing public-flow regression proving that state persists across multiple send calls while separate agents remain isolated.
- [x] Exercise an edit followed by review to prove that line IDs, revisions, instructions, totals, and confirmation eligibility survive the extraction.
- [x] Preserve controlled submission tests proving that a reviewed draft is submitted at most once and that uncertain state is not weakened.
- [x] Run the focused agent/session tests and then the full network-free suite.

## Demo

Build and edit a draft across multiple turns, review it, and show the same responses and calculated totals as before the extraction.

## Migration boundary

This is a behavior-preserving prefactor. The legacy send flow remains the only production path until Ticket 17.
