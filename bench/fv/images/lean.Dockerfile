# Millennium Harness FV Lean image. Toolchain, harnesses and Millennium Harness are baked in: nothing installs at
# runtime (the agent phase can only reach the model API). Everything under /opt and /usr/local is root-owned and
# read-only; the agent runs as the non-root user `agent`, who owns /app. Build context = one task's environment/ dir
# (written by bench/adapters/vericoding_fv.py): sl/ (the millennium_harness package) + Solution.lean.
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive ELAN_HOME=/opt/elan
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils procps ripgrep \
 && rm -rf /var/lib/apt/lists/*
# Lean pinned to vericoding's lean-toolchain. The real binaries (not elan proxies, which need ELAN_HOME) are linked as
# /opt/lean and into /usr/local/bin, because Modal drops the image PATH when [agent] user is set.
RUN curl -fsSL -o /tmp/elan-init.sh https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
 && sh /tmp/elan-init.sh -y --no-modify-path --default-toolchain none && rm /tmp/elan-init.sh \
 && /opt/elan/bin/elan toolchain install leanprover/lean4:v4.23.0-rc2 && /opt/elan/bin/elan default leanprover/lean4:v4.23.0-rc2 \
 && ln -s "$(dirname "$(dirname "$(/opt/elan/bin/elan which lean)")")" /opt/lean \
 && ln -s /opt/lean/bin/lean /opt/lean/bin/lake /usr/local/bin/ \
 && chmod -R a+rX,go-w /opt/elan && lean --version
# Node 22 + exact harness versions.
RUN curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C /opt \
 && ln -s /opt/node-v22.12.0-linux-x64 /opt/node && ln -s /opt/node/bin/node /opt/node/bin/npm /opt/node/bin/npx /usr/local/bin/ \
 && npm install -g --prefix /usr/local opencode-ai@1.18.35 @anthropic-ai/claude-code@2.1.296 && npm cache clean --force \
 && opencode --version && claude --version
# Non-root agent; warm OpenCode's model catalogue (models.dev is not reachable in the agent phase).
RUN useradd -m -u 1000 -s /bin/bash agent \
 && (su agent -s /bin/bash -c "timeout 120 opencode models anthropic" > /dev/null 2>&1 || true)
# millennium_harness at /opt/sl/src, importable by `python3 -I -m millennium_harness.lean` (site .pth; -I ignores cwd
# and PYTHONPATH).
COPY sl/ /opt/sl/src/
RUN echo /opt/sl/src > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/millennium_harness.pth" \
 && chmod -R a+rX,go-w /opt/sl && install -d -m 700 /var/sl-snap && python3 -I -c "import millennium_harness.lean"
WORKDIR /app
COPY --chown=agent:agent Solution.lean /app/Solution.lean
RUN chown agent:agent /app
