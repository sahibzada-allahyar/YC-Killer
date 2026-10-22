# FV task images

`lean.Dockerfile` and `dafny.Dockerfile` are templates that `bench/adapters/vericoding_fv.py` copies verbatim into each
task's `environment/Dockerfile`, next to `sl/` (a copy of the `millennium_harness` package) and the skeleton file. The layers
before `COPY sl/` are identical across tasks, so Modal builds them once and caches them.

| | Lean | Dafny |
|---|---|---|
| Toolchain | elan in `/opt/elan`, `leanprover/lean4:v4.23.0-rc2`, real binaries at `/opt/lean/bin` and `/usr/local/bin` | Dafny 4.11.0 (bundled z3) in `/opt/dafny`, `/usr/local/bin/dafny` |
| Harnesses | `opencode-ai@1.18.35`, `@anthropic-ai/claude-code@2.1.296` (npm, `/usr/local`), Node 22.12.0 | same |
| Millennium Harness | `/opt/sl/src/millennium_harness`, on `sys.path` via a site `.pth` | same |
| Agent | user `agent` (uid 1000) owns `/app`; `/opt`, `/usr/local` and `/var/sl-snap` (mode 700) are root-owned | same |

Rules these images implement (docs/research/win-plan-synthesis.md §3):

- Nothing installs at runtime. `FVBranch.install` / `ClaudeCodeFV.install` only check the baked versions and fail
  otherwise. ripgrep is baked because OpenCode would otherwise download it.
- Tools are linked into `/usr/local/bin`, because Modal drops the image `PATH` when a task sets `[agent] user`.
- The rung always runs as `python3 -I -m millennium_harness.<lang> check ...` with a clean environment
  (`bench.agents_fv.rung_cmd`). `-I` ignores the cwd and `PYTHONPATH`, so a planted `millennium_harness/` package or a PATH shim
  cannot shadow the root-owned copy.
- `/var/sl-snap` holds the same-trajectory snapshots (S0, B1, B2, C1, C2) while a trial runs. Only root can read it.

Check before any agent spend (this needs Modal): build one image of each kind, then run
`lean --version`, `dafny --version`, `opencode --version` and `claude --version` as `agent`, and
`python3 -I -m millennium_harness.lean check` on an honest reference. The egress smoke test must show that `curl` to
github.com, huggingface.co and raw.githubusercontent.com fails during the agent phase.
