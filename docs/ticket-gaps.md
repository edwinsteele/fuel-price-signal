# Ticket gap tally

One row per fix round. Written by ticket-to-pr; see that skill for what the categories mean.

| date | issue | round | category | reason |
|---|---|---|---|---|
| 2026-10-10 | #435 | round 1 | unresolved-semantics | the ticket said "sets last_seen to the request time" but left open the timestamp precision the 410 comparison needs, and what to do when commits arrive out of order |
| 2026-10-11 | #436 | round 1 | none | the ticket specified the retry policy exactly; the reviewer disagreed with the ticket, not with the implementation |
| 2026-10-23 | #449 | round 1 | none | the ticket said to attach the handler "only if the logger has none"; reading that as any handler, and leaving out a lock, were mistakes in the code, not gaps the ticket left |
