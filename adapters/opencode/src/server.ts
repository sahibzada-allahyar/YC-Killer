// opencode-millennium-harness — review the spec, not the code.  (OpenCode V1 plugin, >=1.18.29 <2)
// One export only (the V1 loader treats every export as a plugin).
import { type Plugin, tool } from "@opencode-ai/plugin"
import { spawn } from "node:child_process"
import { homedir } from "node:os"

const BIN = process.env.MILLENNIUM_BIN ?? process.env.SORRYLESS_BIN ?? "millennium-harness"
const MAX_ROUNDS = 2

function run(args: string[], cwd: string, input = "", timeoutMs = 300_000): Promise<{ code: number; out: string }> {
  return new Promise((resolve) => {
    const [cmd, ...rest] = BIN.split(" ")
    const p = spawn(cmd, [...rest, ...args], { cwd, env: { ...process.env, MILLENNIUM_HOST: process.env.MILLENNIUM_HOST ?? process.env.SORRYLESS_HOST ?? "opencode" } })
    let out = ""
    p.stdout.on("data", (d) => (out += d))
    p.stderr.on("data", (d) => (out += d))
    const t = setTimeout(() => p.kill("SIGKILL"), timeoutMs)
    p.on("close", (code) => { clearTimeout(t); resolve({ code: code ?? 1, out: out.trim() }) })
    p.on("error", () => { clearTimeout(t); resolve({ code: 127, out: `${BIN} not found — install with: uv tool install millennium-harness` }) })
    p.stdin.end(input)
  })
}

const STORE = `${homedir()}/.cache/millennium-harness`

const MillenniumHarnessPlugin: Plugin = async ({ client, directory }) => {
  const inflight = new Set<string>()
  const rounds = new Map<string, number>()
  return {
    tool: {
      qed: tool({
        description:
          "Write and lock an executable specification for an algorithmic task BEFORE coding it. A separate model " +
          "that never sees code writes 3 brute-force references + input generators from the task text; they must agree. " +
          "When you finish, your code is checked against it and you get the smallest counterexample if it disagrees.",
        args: {
          task: tool.schema.string().describe("the full task statement, including any examples"),
          target: tool.schema.string().describe("solution.py (stdin program) or path.py::function_name"),
        },
        async execute(args, ctx) {
          const r = await run(["spec", args.task, "--target", args.target], ctx.directory)
          return { title: r.code === 0 ? "spec locked" : "spec failed", output: r.out, metadata: { exit: r.code } }
        },
      }),
    },
    // the locked spec is not for the agent to read or edit
    "tool.execute.before": async (input, output) => {
      const s = JSON.stringify(output.args ?? {})
      if (s.includes(STORE) || s.includes(".millennium-harness/target.json")) throw new Error("Millennium Harness: the locked spec is off-limits")
    },
    // pre-finish gate (TUI / `opencode serve`). Headless `opencode run` exits on idle — benchmarks gate in the driver.
    event: async ({ event }) => {
      if (event.type !== "session.idle") return
      const id = (event as any).properties.sessionID as string
      if (inflight.has(id)) return
      inflight.add(id)
      try {
        const s = await client.session.get({ path: { id } })
        if (s.data?.parentID) return
        const n = rounds.get(id) ?? 0
        if (n >= MAX_ROUNDS) return
        const r = await run(["gate", "--hook", "opencode"], directory)
        if (r.code !== 2) {
          if (r.code === 0) await client.tui.showToast({ body: { title: "Millennium Harness", message: "▢ agrees with the locked spec", variant: "info" } }).catch(() => {})
          return
        }
        rounds.set(id, n + 1)
        await client.tui.showToast({ body: { title: "Millennium Harness", message: "✗ counterexample found — sending it back", variant: "error" } }).catch(() => {})
        await client.session.promptAsync({ path: { id }, body: { parts: [{ type: "text", text: r.out }] } })
      } finally {
        setTimeout(() => inflight.delete(id), 2000)
      }
    },
  }
}

export default MillenniumHarnessPlugin
