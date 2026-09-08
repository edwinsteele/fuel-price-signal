---
name: float-reformulation-not-strictly-more-accurate
description: A 'more exact' float path trades rounding failure modes rather than removing them; check against exact rational arithmetic.
metadata:
  type: project
---

Float reformulations trade rounding-tie failure modes, they don't eliminate them: a/b < c and a < c*b round differently near boundaries, and size/(size/life) does not always recover 'life' exactly (double-rounding). A 'more exact' computation path (e.g. persisting raw fields instead of a display-rounded stamp) can be WRONG in cases where the old lossy path happened to round correctly by luck. Always sweep a wide config grid and check against exact rational arithmetic before claiming a reformulation 'removes' a tie or is 'mathematically equivalent' to what it replaces — verified in fps-o0h/PR #355 review (TankParams.full_to_empty_days: TankParams(50.0, 50.0/11) gives 10.999999999999998, not 11.0; level/size<floor vs level<floor*size disagree at size=50,daily=2.0,floor=0.14,level=7.0).
