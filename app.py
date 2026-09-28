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
PROFILE_ROOT = os.getenv("ORBIT_PROFILE_DIR", "/app/.orbit-profiles")
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
    tabId: str | None = None
    profileId: str | None = None


class SessionRequest(BaseModel):
    key: str = Field(min_length=8)
    profileId: str | None = None


class ControlInput(BaseModel):
    key: str = Field(min_length=8)
    type: str
    tabId: str | None = None
    x: float | None = None
    y: float | None = None
    text: str = ""
    keyName: str = "Enter"
    url: str = ""
    profileId: str | None = None


def key_fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def profile_path(key: str, profile_id: str | None) -> str:
    identity = f"{key}:{profile_id or 'default'}"
    return os.path.join(PROFILE_ROOT, f"{key_fingerprint(identity)}.json")


async def get_session(session_id: str, key: str | None = None, profile_id: str | None = None) -> dict[str, Any]:
    global playwright_instance, browser
    if browser is None:
        playwright_instance = await async_playwright().start()
        browser = await playwright_instance.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
    if session_id not in sessions:
        context = await browser.new_context(viewport={"width": 1280, "height": 900})
        saved_profile = profile_path(key, profile_id) if key else ""
        if saved_profile and os.path.exists(saved_profile):
            await context.close()
            try:
                context = await browser.new_context(viewport={"width": 1280, "height": 900}, storage_state=saved_profile)
            except Exception:
                context = await browser.new_context(viewport={"width": 1280, "height": 900})
        tab_id = str(uuid.uuid4())
        sessions[session_id] = {"context": context, "tabs": {tab_id: await context.new_page()}, "active": tab_id, "history": [], "key": key_fingerprint(key) if key else None, "profile_path": saved_profile, "profile_id": profile_id}
    return sessions[session_id]


async def save_profile(session: dict[str, Any]) -> None:
    path = session.get("profile_path")
    if not path:
        return
    try:
        os.makedirs(PROFILE_ROOT, exist_ok=True)
        await session["context"].storage_state(path=path)
    except Exception:
        pass


def active_tab(session: dict[str, Any], tab_id: str | None = None) -> tuple[str, Page]:
    chosen = tab_id or session["active"]
    if chosen not in session["tabs"]:
        chosen = session["active"]
    session["active"] = chosen
    return chosen, session["tabs"][chosen]


async def tab_list(session: dict[str, Any]) -> list[dict[str, Any]]:
    tabs = []
    for tab_id, page in list(session["tabs"].items()):
        if page.is_closed():
            session["tabs"].pop(tab_id, None)
            continue
        try:
            title = await page.title()
        except Exception:
            title = ""
        tabs.append({"id": tab_id, "title": title[:32] or "New tab", "url": page.url, "active": tab_id == session["active"]})
    if not tabs:
        tab_id = str(uuid.uuid4())
        session["tabs"][tab_id] = await session["context"].new_page()
        session["active"] = tab_id
        return await tab_list(session)
    return tabs


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


async def screenshot(run_id: str, session: dict[str, Any], page: Page, message: str) -> None:
    active_id, _ = active_tab(session)
    payload: dict[str, Any] = {"message": message, "url": page.url, "tabId": active_id, "tabs": await tab_list(session)}
    image = await image_data(page)
    if image:
        payload["screenshot"] = image
    else:
        payload["screenshotDelayed"] = True
    event(run_id, "screenshot", payload)


async def observe(session: dict[str, Any]) -> dict[str, Any]:
    tab_id, page = active_tab(session)
    try:
        title = await page.title()
    except Exception:
        title = ""
    try:
        text = await page.locator("body").inner_text(timeout=5000)
    except Exception:
        text = ""
    try:
        frames = await page.locator("iframe").evaluate_all("els => els.map(el => el.src || '').join(' ')")
    except Exception:
        frames = ""
    return {"tabId": tab_id, "url": page.url, "title": title, "text": text[:10000], "frames": frames, "tabs": await tab_list(session)}


def human_check(observation: dict[str, Any]) -> bool:
    signal = f"{observation['url']} {observation['title']} {observation['text']} {observation.get('frames', '')}".lower()
    markers = ("captcha", "recaptcha", "hcaptcha", "turnstile", "i'm not a robot", "im not a robot", "verify you are human", "human verification", "press and hold")
    return any(marker in signal for marker in markers)


def action_summary(action: dict[str, Any]) -> str:
    kind = str(action.get("type", "action")).lower()
    if kind == "fill":
        target = action.get("label") or action.get("placeholder") or action.get("selector")
        return f"Typing into {target or 'a field'}"
    target = action.get("label") or action.get("selector") or action.get("text") or action.get("url")
    if kind == "click":
        return f"Clicking {target or 'a target'}"
    if kind == "navigate":
        return f"Opening {target or 'a page'}"
    if kind == "new_tab":
        return "Opening a new tab"
    if kind == "switch_tab":
        return "Switching to another tab"
    if kind == "close_tab":
        return "Closing a tab"
    if kind == "scroll":
        return "Scrolling the page"
    if kind == "press":
        return f"Pressing {action.get('keyName', 'Enter')}"
    if kind == "wait":
        return "Waiting for the page"
    return f"Running {kind}"


async def ask_next(key: str, model: str, task: str, observation: dict[str, Any], history: list[str], resume_note: str = "") -> dict[str, Any]:
    prompt = f'''You are Orbit, an autonomous browser agent. Work out the user's complete goal, including sensible missing substeps, and keep acting until it is done. Observe the current page after every action. If the page changes, adapt to what is actually visible. Do not claim success without checking.

User goal:
{task}

Active tab {observation['tabId']}:
URL: {observation['url']}
Title: {observation['title']}
Visible text:
{observation['text']}

Open tabs:
{json.dumps(observation['tabs'])}

Already completed:
{json.dumps(history[-12:])}

{resume_note}

Return JSON only:
{{"thought":"brief internal plan","answer":"final answer if done","done":false,"needs_login":false,"needs_confirmation":false,"message":"what the user should do if paused","action":{{"type":"navigate|new_tab|switch_tab|close_tab|click|fill|press|scroll|wait","tabId":"","url":"","text":"","selector":"","label":"","placeholder":"","value":"","keyName":"Enter","amount":500,"ms":1000}}}}

Rules:
- Use one action at a time. Infer URLs, buttons, fields, scrolling, tab changes, and verification steps yourself.
- Use new_tab to open a separate page, switch_tab to work in another open page, and close_tab only when useful.
- Prefer labels, placeholders, and visible text over brittle selectors.
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


async def execute_action(session: dict[str, Any], action: dict[str, Any]) -> str:
    kind = str(action.get("type", "")).lower()
    if kind == "new_tab":
        page = await session["context"].new_page()
        tab_id = str(uuid.uuid4())
        session["tabs"][tab_id] = page
        session["active"] = tab_id
        url = action.get("url", "")
        if url:
            if not url.startswith(("https://", "http://")):
                raise RuntimeError("The new tab URL was not http or https.")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return f"Opened a new tab{f' for {url}' if url else ''}"
    if kind == "switch_tab":
        tab_id = action.get("tabId")
        if tab_id not in session["tabs"]:
            raise RuntimeError("That tab is no longer open.")
        session["active"] = tab_id
        return "Switched to another tab"
    if kind == "close_tab":
        tab_id = action.get("tabId") or session["active"]
        if len(session["tabs"]) <= 1:
            raise RuntimeError("The last tab cannot be closed.")
        await session["tabs"][tab_id].close()
        session["tabs"].pop(tab_id, None)
        session["active"] = next(iter(session["tabs"]))
        return "Closed a tab"
    _, page = active_tab(session, action.get("tabId"))
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
    session = await get_session(session_id, body.key, body.profileId)
    if body.tabId and body.tabId in session["tabs"]:
        session["active"] = body.tabId
    try:
        event(run_id, "status", {"message": "Reading the page and deciding the next step", "tabs": await tab_list(session)})
        for step in range(20):
            observation = await observe(session)
            event(run_id, "observe", {"message": f"Step {step + 1}: {observation['title'] or observation['url']}", "url": observation["url"], "tabs": observation["tabs"]})
            if human_check(observation):
                record["status"] = "waiting_user"
                record["waiting_kind"] = "human_check"
                record["waiting_message"] = "A website is asking for a human verification. Complete it in the live browser, then continue."
                event(run_id, "human_check", {"message": record["waiting_message"], "sessionId": session_id, "tabs": await tab_list(session)})
                return
            event(run_id, "thinking", {"message": "Choosing the next browser step", "step": step + 1, "tabs": observation["tabs"]})
            decision = await ask_next(body.key, body.model, body.task, observation, session["history"], resume_note if step == 0 else "")
            resume_note = ""
            if decision["needs_login"] or decision["needs_confirmation"]:
                kind = "login_required" if decision["needs_login"] else "confirmation_required"
                record["status"] = "waiting_user"
                record["waiting_kind"] = kind
                record["waiting_message"] = decision.get("message") or ("Sign in on the remote browser, then continue." if kind == "login_required" else "Confirm this action in the app to continue.")
                event(run_id, kind, {"message": record["waiting_message"], "sessionId": session_id, "tabs": await tab_list(session)})
                return
            if decision["done"] or not decision["action"]:
                record["result"] = decision.get("answer") or "The requested browser task is complete."
                record["status"] = "completed"
                event(run_id, "complete", {"message": record["result"]})
                _, page = active_tab(session)
                await screenshot(run_id, session, page, "Task complete")
                await save_profile(session)
                return
            event(run_id, "acting", {"message": action_summary(decision["action"]), "tabs": await tab_list(session)})
            description = await execute_action(session, decision["action"])
            session["history"].append(description)
            event(run_id, "action", {"message": description, "tabs": await tab_list(session)})
            _, page = active_tab(session)
            await screenshot(run_id, session, page, description)
            await save_profile(session)
        record["result"] = "I reached the safe step limit. The current browser tabs are ready for a follow-up."
        record["status"] = "completed"
    except Exception as error:
        record["error"] = str(error)
        record["status"] = "failed"
        event(run_id, "error", {"message": str(error)})


def verify_session(session_id: str, key: str) -> dict[str, Any]:
    session = sessions.get(session_id)
    if not session or session.get("key") != key_fingerprint(key):
        raise HTTPException(403, "This browser session is not available.")
    return session


def verify_run(run_id: str, key: str) -> dict[str, Any]:
    record = runs.get(run_id)
    if not record:
        raise HTTPException(404, "Run not found")
    if key_fingerprint(record["body"].key) != key_fingerprint(key):
        raise HTTPException(403, "This browser run belongs to another key.")
    return record


@app.get("/")
async def home() -> FileResponse:
    return FileResponse("browser-agent.html")


@app.post("/api/browser/session")
async def create_session(body: SessionRequest) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    session = await get_session(session_id, body.key, body.profileId)
    session["key"] = key_fingerprint(body.key)
    tab_id, page = active_tab(session)
    return {"sessionId": session_id, "activeTabId": tab_id, "tabs": await tab_list(session), "url": page.url, "screenshot": await image_data(page)}


@app.post("/api/browser/{session_id}/input")
async def browser_input(session_id: str, body: ControlInput) -> dict[str, Any]:
    session = verify_session(session_id, body.key)
    if body.type == "new_tab":
        page = await session["context"].new_page()
        tab_id = str(uuid.uuid4())
        session["tabs"][tab_id] = page
        session["active"] = tab_id
        if body.url:
            await page.goto(body.url, wait_until="domcontentloaded", timeout=30000)
    elif body.type == "switch_tab":
        if body.tabId not in session["tabs"]:
            raise HTTPException(404, "Tab not found")
        session["active"] = body.tabId
    elif body.type == "close_tab":
        if len(session["tabs"]) <= 1:
            raise HTTPException(409, "The last tab cannot be closed.")
        tab_id = body.tabId or session["active"]
        await session["tabs"][tab_id].close()
        session["tabs"].pop(tab_id, None)
        session["active"] = next(iter(session["tabs"]))
    else:
        _, page = active_tab(session, body.tabId)
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
    tab_id, page = active_tab(session)
    await save_profile(session)
    return {"activeTabId": tab_id, "tabs": await tab_list(session), "url": page.url, "screenshot": await image_data(page)}


@app.post("/api/browser/run")
async def create_run(body: BrowserRun) -> dict[str, Any]:
    session_id = body.sessionId or str(uuid.uuid4())
    session = await get_session(session_id, body.key, body.profileId)
    if session.get("key") and session["key"] != key_fingerprint(body.key):
        raise HTTPException(403, "This browser session belongs to another key.")
    session["key"] = key_fingerprint(body.key)
    run_id = str(uuid.uuid4())
    runs[run_id] = {"status": "running", "result": None, "error": None, "events": [], "sessionId": session_id, "body": body}
    asyncio.create_task(run_task(run_id, body, session_id))
    return {"id": run_id, "status": "running", "sessionId": session_id}


@app.post("/api/browser/{run_id}/resume")
async def resume_run(run_id: str, body: SessionRequest) -> dict[str, Any]:
    record = verify_run(run_id, body.key)
    if record["status"] != "waiting_user":
        raise HTTPException(409, "This browser run is not waiting for input.")
    record["status"] = "running"
    event(run_id, "status", {"message": "Continuing after user input"})
    asyncio.create_task(run_task(run_id, record["body"], record["sessionId"], "The user has completed the requested sign-in or confirmation. Continue the original task."))
    return {"status": "running"}


@app.get("/api/browser/{run_id}/status")
async def run_status(run_id: str, key: str = Query(..., min_length=8)) -> dict[str, Any]:
    return {"status": verify_run(run_id, key)["status"]}


@app.get("/api/browser/{run_id}/events")
async def run_events(run_id: str, key: str = Query(..., min_length=8), after: int = 0, limit: int = 100, include_output: bool = False) -> dict[str, Any]:
    record = verify_run(run_id, key)
    events = [item for item in record["events"] if item["id"] > after][:limit]
    return {"events": events, "nextAfter": events[-1]["id"] if events else after, "hasMore": False}


@app.get("/api/browser/{run_id}")
async def run_result(run_id: str, key: str = Query(..., min_length=8)) -> dict[str, Any]:
    record = verify_run(run_id, key)
    return {"status": record["status"], "result": record["result"], "error": record["error"], "waitingKind": record.get("waiting_kind"), "waitingMessage": record.get("waiting_message")}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
