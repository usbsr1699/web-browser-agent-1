import os
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

API = "https://api.browser-use.com/api/v4"
app = FastAPI(title="Orbit Browser Agent")


class BrowserRun(BaseModel):
    key: str = Field(min_length=8)
    task: str = Field(min_length=1, max_length=8000)
    model: str = "gemini-3.5-flash"
    sessionId: str | None = None


def auth(key: str) -> dict[str, str]:
    return {"X-Browser-Use-API-Key": key}


async def proxy(response: httpx.Response) -> dict[str, Any]:
    if response.is_error:
        try:
            detail = response.json()
        except Exception:
            detail = {"detail": response.text[:500]}
        raise HTTPException(status_code=response.status_code, detail=detail)
    return response.json()


@app.get("/")
async def home() -> FileResponse:
    return FileResponse("browser-agent.html")


@app.post("/api/browser/run")
async def create_run(body: BrowserRun) -> dict[str, Any]:
    task = (
        "Operate a real browser for the user. Complete this request: "
        f"{body.task}. You may navigate, click, type, scroll, and inspect pages. "
        "Do not submit forms, send messages, purchase anything, delete data, or take "
        "other irreversible actions unless the user's request explicitly asks for that exact action. "
        "Report what you actually did."
    )
    payload: dict[str, Any] = {
        "task": task,
        "model": body.model,
        "sessionId": body.sessionId,
        "maxCostUsd": 0.75,
    }
    if not body.sessionId:
        payload["browserSettings"] = {"record": True, "screenWidth": 1280, "screenHeight": 900}
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(f"{API}/runs", headers=auth(body.key), json=payload)
    return await proxy(response)


@app.get("/api/browser/{run_id}/status")
async def run_status(run_id: str, key: str = Query(..., min_length=8)) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{API}/runs/{run_id}/status", headers=auth(key))
    return await proxy(response)


@app.get("/api/browser/{run_id}/events")
async def run_events(run_id: str, key: str = Query(..., min_length=8), limit: int = 100, after: int = 0, include_output: bool = False) -> dict[str, Any]:
    params = {"limit": limit, "after": after, "include_output": str(include_output).lower()}
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{API}/runs/{run_id}/events", headers=auth(key), params=params)
    return await proxy(response)


@app.get("/api/browser/{run_id}")
async def run_result(run_id: str, key: str = Query(..., min_length=8)) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{API}/runs/{run_id}", headers=auth(key))
    return await proxy(response)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
