import asyncio
import base64
import json
import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from playwright.async_api import Browser, Page, async_playwright

GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
app = FastAPI(title="Orbit Free Browser Agent")
playwright_instance = None
browser: Browser | None = None
pages: dict[str, Page] = {}
runs: dict[str, dict[str, Any]] = {}


class BrowserRun(BaseModel):
    key: str = Field(min_length=8)
    task: str = Field(min_length=1, max_length=8000)
    model: str = "gemini-3.5-flash"
    sessionId: str | None = None


async def get_page(session_id: str) -> Page:
    global playwright_instance, browser
    if browser is None:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    if session_id not in pages:
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        pages[session_id] = await context.new_page()
    return pages[session_id]


def event(run_id: str, kind: str, data: dict[str, Any]) -> None:
    record = runs[run_id]
    record["events"].append({"id": len(record["events"]) + 1, "type": kind, "data": data})
    record["events"] = record["events"][-60:]


async def screenshot(run_id: str, page: Page, message: str) -> None:
    try:
        image = await page.screenshot(type="jpeg", quality=55, full_page=False, timeout=8000)
    except Exception as error:
        event(run_id, "status", {"message": f"Screenshot delayed; continuing task ({str(error)[:120]})"})
        return
    encoded = base64.b64encode(image).decode("ascii")
    event(run_id, "screenshot", {"message": message, "screenshot": f"data:image/jpeg;base64,{encoded}"})


async def plan_with_gemini(key: str, model: str, task: str) -> dict[str, Any]:
    prompt = f'''You are the planning brain for a real Playwright browser. Plan safe actions for this request:
{task}

Return JSON only in this shape:
{{"answer":"brief result summary","actions":[{{"type":"navigate","url":"https://example.com"}}]}}

Allowed actions: navigate with url; click with text or selector; fill with label, placeholder, selector, and text; press with key; scroll with amount; wait with ms. Use public HTTPS URLs. Do not invent that an action succeeded. Do not submit forms, send messages, buy anything, delete data, or take irreversible actions unless the user's request explicitly asks for that exact action.'''
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.15, "maxOutputTokens": 1200, "responseMimeType": "application/json"}}
    async with httpx.AsyncClient(timeout=45) as client:
        response = await client.post(f"{GEMINI}/{model}:generateContent", params={"key": key}, json=body)
    if response.is_error:
        try:
            detail = response.json().get("error", {}).get("message", response.text)
        except Exception:
            detail = response.text[:500]
        raise RuntimeError(detail)
    payload = response.json()
    raw = "".join(part.get("text", "") for part in payload.get("candidates", [{}])[0].get("content", {}).get("parts", []))
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"answer": raw or "Gemini returned no plan.", "actions": []}
    result["answer"] = result.get("answer", "")
    result["actions"] = result.get("actions", []) if isinstance(result.get("actions", []), list) else []
    return result


async def execute_action(page: Page, action: dict[str, Any]) -> str:
    kind = action.get("type", "").lower()
    if kind == "navigate":
        url = action.get("url", "")
        if not url.startswith(("https://", "http://")):
            raise RuntimeError("Navigation was blocked because the URL was not http or https.")
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return f"Opened {url}"
    if kind == "wait":
        await page.wait_for_timeout(min(int(action.get("ms", 1000)), 10000))
        return "Waited for the page"
    if kind == "scroll":
        await page.mouse.wheel(0, int(action.get("amount", 650)))
        return "Scrolled the page"
    if kind == "press":
        await page.keyboard.press(action.get("key", "Enter"))
        return f"Pressed {action.get('key', 'Enter')}"
    selector = action.get("selector")
    label = action.get("label")
    placeholder = action.get("placeholder")
    text = action.get("text", "")
    locator = page.locator(selector) if selector else page.get_by_label(label, exact=False) if label else page.get_by_placeholder(placeholder, exact=False) if placeholder else page.get_by_text(text, exact=False)
    locator = locator.first
    if kind == "fill":
        await locator.fill(str(action.get("value", action.get("text", ""))))
        return f"Filled {label or placeholder or selector or 'the field'}"
    if kind == "click":
        await locator.click(timeout=15000)
        return f"Clicked {text or selector or 'the target'}"
    raise RuntimeError(f"Unsupported browser action: {kind}")


async def run_task(run_id: str, body: BrowserRun, session_id: str) -> None:
    record = runs[run_id]
    try:
        event(run_id, "status", {"message": "Planning browser actions with Gemini"})
        plan = await plan_with_gemini(body.key, body.model, body.task)
        page = await get_page(session_id)
        await screenshot(run_id, page, "Remote browser ready")
        completed = []
        for action in plan["actions"][:12]:
            description = await execute_action(page, action)
            completed.append(description)
            event(run_id, "action", {"message": description})
            await screenshot(run_id, page, description)
        record["result"] = (plan["answer"] or "Browser task completed.") + (f" Completed: {'; '.join(completed)}." if completed else "")
        record["status"] = "completed"
    except Exception as error:
        record["error"] = str(error)
        record["status"] = "failed"
        event(run_id, "error", {"message": str(error)})


@app.get("/")
async def home() -> FileResponse:
    return FileResponse("browser-agent.html")


@app.post("/api/browser/run")
async def create_run(body: BrowserRun) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    session_id = body.sessionId or str(uuid.uuid4())
    runs[run_id] = {"status": "running", "result": None, "error": None, "events": []}
    asyncio.create_task(run_task(run_id, body, session_id))
    return {"id": run_id, "status": "running", "sessionId": session_id}


@app.get("/api/browser/{run_id}/status")
async def run_status(run_id: str, key: str = Query(..., min_length=8)) -> dict[str, Any]:
    if run_id not in runs:
        raise HTTPException(404, "Run not found")
    return {"status": runs[run_id]["status"]}


@app.get("/api/browser/{run_id}/events")
async def run_events(run_id: str, key: str = Query(..., min_length=8), after: int = 0, limit: int = 100, include_output: bool = False) -> dict[str, Any]:
    if run_id not in runs:
        raise HTTPException(404, "Run not found")
    events = [item for item in runs[run_id]["events"] if item["id"] > after][:limit]
    return {"events": events, "nextAfter": events[-1]["id"] if events else after, "hasMore": False}


@app.get("/api/browser/{run_id}")
async def run_result(run_id: str, key: str = Query(..., min_length=8)) -> dict[str, Any]:
    if run_id not in runs:
        raise HTTPException(404, "Run not found")
    record = runs[run_id]
    return {"status": record["status"], "result": record["result"], "error": record["error"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
