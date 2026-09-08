---
name: prose-justifications-fail-silently
description: Prose asserting WHY something is the way it is is the claim nothing executes — attach the measurement or delete the sentence.
metadata:
  type: feedback
---

The claims that survive review are the ones nothing executes: prose asserting WHY something is the way it is. A helper's justification for existing, the guarantee a documented escape hatch offers, the population a number describes, a test's discriminating power. Tests fail loudly and types fail loudly; a justification fails silently forever, and the author is the last person able to catch it because they have already convinced themselves. PR #361 hit this three times in one review (all found by the reviewer, none by CI): '_placeholders exists because order matters here' — describe_universe sorts before calling it; 'this test fails if anyone reintroduces the second mechanism' — it does not, that form decides identically; 'three of these cases fail' — five do. The CODE was right every time. How to apply: when writing a docstring sentence of the form 'X is this way BECAUSE Y', ask what would fail if Y were false — if the answer is nothing, either delete the sentence, or attach the command/measurement that establishes it so the next reader re-derives rather than trusts. Prefer removing the fragility to documenting it (the two-mechanism coverage gate was refactored to one mechanism rather than given a warning comment). See [[test-discrimination-claims-need-mutation]] for the test-specific case, which needs a mutation run and not just reasoning.
