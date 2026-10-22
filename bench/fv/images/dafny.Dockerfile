# Millennium Harness FV Dafny image. Same layout as lean.Dockerfile: Dafny 4.11 (bundled z3), harnesses and
# Millennium Harness baked in, root-owned and read-only; the agent runs as the non-root user `agent`, who owns /app.
# Build context = one task's environment/ dir: sl/ (the millennium_harness package) + solution.dfy.
FROM python:3.11-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive DOTNET_CLI_TELEMETRY_OPTOUT=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl unzip xz-utils libicu72 procps \
    ripgrep && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/d.zip https://github.com/dafny-lang/dafny/releases/download/v4.11.0/dafny-4.11.0-x64-ubuntu-22.04.zip \
 && unzip -q /tmp/d.zip -d /opt && rm /tmp/d.zip && chmod -R a+rX,go-w /opt/dafny \
 && ln -s /opt/dafny/dafny /usr/local/bin/dafny && dafny --version
# Node 22 + exact harness versions.
RUN curl -fsSL https://nodejs.org/dist/v22.12.0/node-v22.12.0-linux-x64.tar.xz | tar -xJ -C /opt \
 && ln -s /opt/node-v22.12.0-linux-x64 /opt/node && ln -s /opt/node/bin/node /opt/node/bin/npm /opt/node/bin/npx /usr/local/bin/ \
 && npm install -g --prefix /usr/local opencode-ai@1.18.35 @anthropic-ai/claude-code@2.1.296 && npm cache clean --force \
 && opencode --version && claude --version
# Non-root agent; warm OpenCode's model catalogue (models.dev is not reachable in the agent phase).
RUN useradd -m -u 1000 -s /bin/bash agent \
 && (su agent -s /bin/bash -c "timeout 120 opencode models anthropic" > /dev/null 2>&1 || true)
COPY sl/ /opt/sl/src/
RUN echo /opt/sl/src > "$(python3 -c 'import site; print(site.getsitepackages()[0])')/millennium_harness.pth" \
 && chmod -R a+rX,go-w /opt/sl && install -d -m 700 /var/sl-snap && python3 -I -c "import millennium_harness.dafny"
WORKDIR /app
COPY --chown=agent:agent solution.dfy /app/solution.dfy
RUN chown agent:agent /app
