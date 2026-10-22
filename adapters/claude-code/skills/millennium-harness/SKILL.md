---
name: millennium-harness
description: Use for algorithmic or pure-function coding tasks with checkable outputs. Locks a fresh-context executable spec before coding, then checks the code against it (counterexample feedback). Silent on tasks it cannot check.
---
# Millennium Harness — review the spec, not the code

When a task asks for a program or pure function whose output is determined by its input
(algorithms, parsing, math, data transforms):

1. **Before coding**, lock a spec from the task text only:
   `millennium-harness spec "<task text incl. examples>" --target solution.py` (stdin program)
   or `--target path/to/file.py::function_name` (pure function).
   It prints `spec <sha> locked` or `◧ no checkable spec` (then just proceed normally).
2. Write the code at the target.
3. Before saying you are done: `millennium-harness check`.
   - `▢ agrees with N references on M inputs` → done; mention the receipt line.
   - `✗ REFUTED …` → read the smallest counterexample, fix the underlying logic (never special-case the input),
     and re-check. If you are certain the reference is wrong, explain why instead.
4. Never read or edit the locked spec (`~/.cache/millennium-harness`, `.millennium-harness/target.json`); `millennium-harness show` is for humans.

What a ▢ means: the code agreed with ≥2 independent brute-force references on every bounded input tried.
It is evidence, not proof (∎ is reserved for real proofs).
