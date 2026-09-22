from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .app import Application, build_application


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    thread_id: str = Field(default="research-chat", min_length=1, max_length=128)


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_dotenv()
    application.state.research = build_application()
    yield
    application.state.research.close()


app = FastAPI(title="Memory-Aware Research Agent", lifespan=lifespan)


def _research() -> Application:
    return app.state.research


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict[str, str]:
    answer = _research().agent.ask(payload.question, thread_id=payload.thread_id)
    return {"answer": answer, "thread_id": payload.thread_id}


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD


def run() -> None:
    import uvicorn

    uvicorn.run("research_memory_agent.web:app", host="127.0.0.1", port=8010, reload=False)


DASHBOARD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Memory-Aware Research Agent</title><style>
:root{--bg:#090d18;--panel:#12182a;--line:#29314a;--muted:#9ca8c7;--ink:#f4f7ff;--accent:#7d8cff}
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at top,#1b2450 0,transparent 42%),var(--bg);color:var(--ink);font:15px Inter,system-ui,sans-serif}
.shell{width:min(900px,calc(100% - 32px));margin:auto;padding:54px 0}.eyebrow{color:#aeb8ff;font-weight:800;letter-spacing:.12em;text-transform:uppercase;font-size:12px}h1{font-size:clamp(34px,6vw,58px);line-height:1.04;margin:12px 0}.sub{color:var(--muted);font-size:17px;line-height:1.6;max-width:700px}.panel{margin-top:32px;background:rgba(18,24,42,.94);border:1px solid var(--line);border-radius:20px;padding:24px;box-shadow:0 24px 70px #0005}.row{display:flex;gap:12px;align-items:end}.thread{width:210px}label{display:block;color:var(--muted);font-size:12px;font-weight:700;margin:0 0 7px}input,textarea{width:100%;border:1px solid var(--line);border-radius:12px;background:#0b1020;color:var(--ink);padding:13px;font:inherit}textarea{min-height:150px;resize:vertical}.ask{flex:1}.actions{display:flex;justify-content:space-between;align-items:center;margin-top:14px;color:var(--muted);font-size:12px}button{border:0;border-radius:11px;background:linear-gradient(135deg,#7181ff,#8d6ff7);color:white;padding:12px 20px;font-weight:800;cursor:pointer}.answer{display:none;margin-top:22px;padding:20px;border-radius:14px;background:#0c1223;border:1px solid var(--line);white-space:pre-wrap;line-height:1.65}.answer.show{display:block}.status{color:var(--muted);margin-bottom:8px;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.08em}@media(max-width:650px){.row{display:block}.thread{width:100%;margin-bottom:14px}.shell{padding:30px 0}}
</style></head><body><main class="shell"><div class="eyebrow">Seven-memory research system</div><h1>Ask. Research. Remember.</h1><p class="sub">Talk directly to the memory-aware agent. It retrieves relevant conversation, knowledge, workflow, entity, and summary memories—and can use research tools when needed.</p><section class="panel"><div class="row"><div class="thread"><label for="thread">Conversation ID</label><input id="thread" value="research-chat"></div><div class="ask"><label for="question">What would you like to research?</label><textarea id="question" placeholder="Example: Find recent research about long-term memory for AI agents and explain the main approaches."></textarea></div></div><div class="actions"><span>Your conversation is remembered under this ID.</span><button id="send" onclick="ask()">Ask agent →</button></div><div id="answer" class="answer"><div id="status" class="status"></div><div id="text"></div></div></section></main><script>
async function ask(){const q=document.querySelector('#question').value.trim();if(!q)return;const box=document.querySelector('#answer'),status=document.querySelector('#status'),text=document.querySelector('#text'),button=document.querySelector('#send');box.classList.add('show');status.textContent='Agent is thinking';text.textContent='Retrieving relevant memories and selecting tools…';button.disabled=true;try{const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({question:q,thread_id:document.querySelector('#thread').value.trim()||'research-chat'})});const data=await r.json();if(!r.ok)throw new Error(data.detail||'The request failed');status.textContent='Answer';text.textContent=data.answer}catch(error){status.textContent='Error';text.textContent=error.message}finally{button.disabled=false}}
document.querySelector('#question').addEventListener('keydown',event=>{if(event.ctrlKey&&event.key==='Enter')ask()});
</script></body></html>"""
