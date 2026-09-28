import asyncio
import base64
import hashlib
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
sessions: dict[str, dict[str, Any]] = {}
runs: dict[str, dict[str, Any]] = {}


class BrowserRun(BaseModel):
    key: str = Field(min_length=8)
    task: str = Field(min_length=1, max_length=8000)
    model: str = "gemini-3.5-flash"
    sessionId: str | None = None


class SessionRequest(BaseModel):
    key: str = Field(min_length=8)


class ControlInput(BaseModel):
    key: str = Field(min_length=8)
    type: str
    x: float | None = None
    y: float | None = None
    text: str = ""
    keyName: str = "Enter"
    url: str = ""


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


async def get_session(session_id: str) -> dict[str, Any]:
    global playwright_instance, browser
    if browser is None:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    if session_id not in sessions:
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        sessions[session_id] = {"context": context, "page": await context.new_page(), "history": [], "key": None}
    return sessions[session_id]


def event(run_id: str, kind: str, data: dict[str, Any]) -> None:
    record = runs[run_id]
    record["events"].append({"id": len(record["events"]) + 1, "type": kind, "data": data})
    record["events"] = record["events"][-100:]


async def image_data(page: Page) -> str | None:
    try:
        image = await page.screenshot(type="jpeg", quality=55, full_page=False, timeout=8000)
        return f"data:image/jpeg;base64,{base64.b64encode(image).decode('ascii')}"
    except Exception:
        return None


async def screenshot(run_id: str, page: Page, message: str) -> None:
    image = await image_data(page)
    payload: dict[str, Any] = {"message": message, "url": page.url}
    if image:
        payload["screenshot"] = image
    else:
        payload["screenshotDelayed"] = True
    event(run_id, "screenshot", payload)


async def observe(page: Page) -> dict[str, str]:
    try:
        title = await page.title()
    except Exception:
        title = ""
    try:
        text = await page.locator("body").inner_text(timeout=5000)
    except Exception:
        text = ""
    return {"url": page.url, "title": title, "text": text[:10000]}


async def ask_next(key: str, model: str, task: str, observation: dict[str, str], history: list[str], resume_note: str = "") -> dict[str, Any]:
    prompt = f'''You are Orbit, an autonomous browser agent. Work out the user's complete goal, including sensible missing substeps, and keep acting until it is done. Observe the current page after every action. If the page changes, adapt to what is actually visible. Do not claim success without checking the page.

User goal:
{task}

Current page:
URL: {observation['url']}
Title: {observation['title']}
Visible text:
{observation['text']}

Already completed:
{json.dumps(history[-12:])}

{resume_note}

Return JSON only:
{{"thought":"brief internal plan","answer":"final answer if done","done":false,"needs_login":false,"needs_confirmation":false,"message":"what the user should do if paused","action":{{"type":"navigate|click|fill|press|scroll|wait","url":"","text":"","selector":"","label":"","placeholder":"","value":"","keyName":"Enter","amount":500,"ms":1000}}}}

Rules:
- Use one action at a time. Infer URLs, buttons, fields, scrolling, and verification steps yourself.
- Use navigate, click, fill, press, scroll, or wait only. Prefer labels, placeholders, and visible text over brittle selectors.
- If a sign-in, password, one-time code, CAPTCHA, payment, or private credential is needed, set needs_login=true and pause. Never ask for secrets in chat.
- If the user has not explicitly asked to submit, send, purchase, delete, or make an irreversible change, set needs_confirmation=true before that action.
- If the requested result is visible or verified, set done=true and provide a concise answer.
- Do not invent an action or URL.'''
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.15, "maxOutputTokens": 1400, "responseMimeType": "application/json"}}
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
        decision = json.loads(raw)
    except json.JSONDecodeError:
        decision = {"done": True, "answer": raw or "The agent returned no next step."}
    action = decision.get("action")
    if isinstance(action, list):
        action = action[0] if action else None
    decision["action"] = action if isinstance(action, dict) else None
    decision["done"] = bool(decision.get("done"))
    decision["needs_login"] = bool(decision.get("needs_login"))
    decision["needs_confirmation"] = bool(decision.get("needs_confirmation"))
    return decision


async def execute_action(page: Page, action: dict[str, Any]) -> str:
    kind = str(action.get("type", "")).lower()
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
        await page.keyboard.press(action.get("keyName", "Enter"))
        return f"Pressed {action.get('keyName', 'Enter')}"
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


async def run_task(run_id: str, body: BrowserRun, session_id: str, resume_note: str = "") -> None:
    record = runs[run_id]
    session = await get_session(session_id)
    page: Page = session["page"]
    try:
        event(run_id, "status", {"message": "Reading the page and deciding the next step"})
        for step in range(20):
            observation = await observe(page)
            event(run_id, "observe", {"message": f"Step {step + 1}: {observation['title'] or observation['url']}", "url": observation["url"]})
            decision = await ask_next(body.key, body.model, body.task, observation, session["history"], resume_note if step == 0 else "")
            resume_note = ""
            if decision["needs_login"] or decision["needs_confirmation"]:
                kind = "login_required" if decision["needs_login"] else "confirmation_required"
                record["status"] = "waiting_user"
                record["waiting_kind"] = kind
                record["waiting_message"] = decision.get("message") or ("Sign in on the remote browser, then continue." if kind == "login_required" else "Confirm this action in the app to continue.")
                event(run_id, kind, {"message": record["waiting_message"], "sessionId": session_id})
                return
            if decision["done"] or not decision["action"]:
                record["result"] = decision.get("answer") or "The requested browser task is complete."
                record["status"] = "completed"
                event(run_id, "complete", {"message": record["result"]})
                await screenshot(run_id, page, "Task complete")
                return
            description = await execute_action(page, decision["action"])
            session["history"].append(description)
            event(run_id, "action", {"message": description})
            await screenshot(run_id, page, description)
        record["result"] = "I reached the safe step limit. The current browser page is ready for a follow-up."
        record["status"] = "completed"
    except Exception as error:
        record["error"] = str(error)
        record["status"] = "failed"
        event(run_id, "error", {"message": str(error)})


@app.get("/")
async def home() -> FileResponse:
    return FileResponse("browser-agent.html")


@app.post("/api/browser/session")
async def create_session(body: SessionRequest) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    session = await get_session(session_id)
    session["key"] = key_fingerprint(body.key)
    image = await image_data(session["page"])
    return {"sessionId": session_id, "url": session["page"].url, "screenshot": image}


@app.post("/api/browser/{session_id}/input")
async def browser_input(session_id: str, body: ControlInput) -> dict[str, Any]:
    session = sessions.get(session_id)
    if not session or session.get("key") != key_fingerprint(body.key):
        raise HTTPException(403, "This browser session is not available.")
    page: Page = session["page"]
    if body.type == "navigate":
        if not body.url.startswith(("https://", "http://")):
            raise HTTPException(400, "Only http and https addresses are allowed.")
        await page.goto(body.url, wait_until="domcontentloaded", timeout=30000)
    elif body.type == "back":
        await page.go_back(wait_until="domcontentloaded", timeout=30000)
    elif body.type == "forward":
        await page.go_forward(wait_until="domcontentloaded", timeout=30000)
    elif body.type == "reload":
        await page.reload(wait_until="domcontentloaded", timeout=30000)
    elif body.type == "click":
        await page.mouse.click(body.x or 0, body.y or 0)
    elif body.type == "type":
        await page.keyboard.insert_text(body.text)
    elif body.type == "key":
        await page.keyboard.press(body.keyName or "Enter")
    elif body.type == "scroll":
        await page.mouse.wheel(0, body.y or 650)
    else:
        raise HTTPException(400, "Unsupported browser control.")
    return {"url": page.url, "screenshot": await image_data(page)}


@app.post("/api/browser/run")
async def create_run(body: BrowserRun) -> dict[str, Any]:
    session_id = body.sessionId or str(uuid.uuid4())
    session = await get_session(session_id)
    if session.get("key") and session["key"] != key_fingerprint(body.key):
        raise HTTPException(403, "This browser session belongs to another key.")
    session["key"] = key_fingerprint(body.key)
    run_id = str(uuid.uuid4())
    runs[run_id] = {"status": "running", "result": None, "error": None, "events": [], "sessionId": session_id, "body": body}
    asyncio.create_task(run_task(run_id, body, session_id))
    return {"id": run_id, "status": "running", "sessionId": session_id}


@app.post("/api/browser/{run_id}/resume")
async def resume_run(run_id: str, body: SessionRequest) -> dict[str, Any]:
    record = runs.get(run_id)
    if not record or record["status"] != "waiting_user":
        raise HTTPException(409, "This browser run is not waiting for input.")
    original: BrowserRun = record["body"]
    if original.key != body.key:
        raise HTTPException(403, "This browser run belongs to another key.")
    record["status"] = "running"
    event(run_id, "status", {"message": "Continuing after user input"})
    asyncio.create_task(run_task(run_id, original, record["sessionId"], "The user has completed the requested sign-in or confirmation. Continue the original task."))
    return {"status": "running"}


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
    return {"status": record["status"], "result": record["result"], "error": record["error"], "waitingKind": record.get("waiting_kind"), "waitingMessage": record.get("waiting_message")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
