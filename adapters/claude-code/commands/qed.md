---
description: Lock an executable spec for an algorithmic task before coding it (Millennium Harness)
---
Before writing any code for the task below, lock its specification:

1. Run `millennium-harness spec "<the full task text, including examples>" --target <solution.py | path.py::function>`.
   A separate model that never sees code writes three brute-force references and input generators; they must agree.
2. Implement the task at that target.
3. When you stop, Millennium Harness checks your code against the locked spec. If it shows a counterexample, fix the
   underlying bug (never special-case the input). If you are sure the reference is wrong, say why.

Task: $ARGUMENTS
