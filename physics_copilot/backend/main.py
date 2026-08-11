from fastapi import FastAPI

app = FastAPI(
    title="Physics Copilot",
    description="An agent for solving physics problems.",
    version="0.0.1",
)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"message": "Physics Copilot is running."}


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
