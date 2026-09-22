from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .app import Application, build_application


class IngestRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    source: str = Field(min_length=1, max_length=2_000)
    pages: list[str] = Field(min_length=1, max_length=2_000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    thread_id: str = Field(default="evidence-ui", max_length=128)


class ArxivIngestRequest(BaseModel):
    arxiv_id: str = Field(min_length=1, max_length=100)
    title: str = Field(default="", max_length=300)


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_dotenv()
    application.state.research = build_application()
    yield
    application.state.research.close()


app = FastAPI(title="Evidence Research Workspace", lifespan=lifespan)


def _research() -> Application:
    return app.state.research


@app.get("/api/traces")
def traces(limit: int = 12) -> list[dict[str, Any]]:
    return _research().store.recent_evidence_traces(limit=limit)


@app.post("/api/ingest")
def ingest(payload: IngestRequest) -> dict[str, Any]:
    return _research().evidence.ingest_pages(payload.pages, title=payload.title, source=payload.source)


@app.post("/api/ingest/arxiv")
def ingest_arxiv(payload: ArxivIngestRequest) -> dict[str, Any]:
    return _research().evidence.ingest_arxiv(payload.arxiv_id, title=payload.title)


@app.post("/api/ask")
def ask(payload: QuestionRequest) -> dict[str, Any]:
    return _research().evidence.answer(payload.question, thread_id=payload.thread_id)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return DASHBOARD


def run() -> None:
    import uvicorn

    uvicorn.run("research_memory_agent.web:app", host="127.0.0.1", port=8010, reload=False)


DASHBOARD = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evidence Research Workspace</title><style>
:root{--bg:#0c1020;--panel:#151b31;--line:#293253;--muted:#9ba8c7;--ink:#f3f6ff;--accent:#7c8cff;--good:#57d7a4;--bad:#ff8c99}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at top right,#1d2650,transparent 40%),var(--bg);color:var(--ink);font:15px Inter,system-ui,sans-serif}.shell{max-width:1160px;margin:auto;padding:44px 24px}h1{font-size:34px;margin:0 0 8px}h2{font-size:17px;margin:0 0 16px}.sub{color:var(--muted);margin-bottom:28px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:rgba(21,27,49,.9);border:1px solid var(--line);border-radius:16px;padding:20px}label{display:block;color:var(--muted);font-size:12px;margin:12px 0 6px}input,textarea{width:100%;background:#0d1326;border:1px solid var(--line);border-radius:9px;color:var(--ink);padding:11px;font:inherit}textarea{min-height:120px}button{margin-top:14px;background:var(--accent);border:0;border-radius:9px;padding:11px 15px;color:white;font-weight:700;cursor:pointer}.answer{white-space:pre-wrap;line-height:1.55}.trace{border-top:1px solid var(--line);padding:14px 0}.pill{display:inline-block;border-radius:99px;padding:4px 8px;font-size:11px;font-weight:700;background:#23305c;color:#cdd5ff}.ok{background:#123d35;color:var(--good)}.bad{background:#4a2631;color:var(--bad)}.small{font-size:12px;color:var(--muted)}@media(max-width:760px){.grid{grid-template-columns:1fr}.shell{padding:25px 15px}}</style></head><body><main class="shell"><h1>Evidence Research Workspace</h1><div class="sub">Chunked evidence · hybrid retrieval · citations · claim verification · durable traces</div><div class="grid"><section class="panel"><h2>1. Ingest evidence</h2><label>Document title</label><input id="title" value="Demo research note"><label>Source URL or label</label><input id="source" value="local-demo"><label>Pages — separate pages using PAGE BREAK</label><textarea id="pages">Long-term agent memory stores useful information outside the active prompt. Retrieval systems select only relevant evidence before an answer is generated.\nPAGE BREAK\nHybrid retrieval combines semantic vector similarity with exact keyword search. A verifier should check each cited claim against its evidence.</textarea><button onclick="ingest()">Chunk and index</button><div id="ingest" class="small"></div></section><section class="panel"><h2>2. Ask a grounded question</h2><label>Question</label><textarea id="question">What does hybrid retrieval combine?</textarea><button onclick="ask()">Retrieve, answer, verify</button><div id="result" class="answer"></div></section></div><section class="panel" style="margin-top:18px"><h2>Evidence traces</h2><div id="traces" class="small">Loading…</div></section></main><script>
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function ingest(){let pages=document.querySelector('#pages').value.split('\\nPAGE BREAK\\n');let body={title:document.querySelector('#title').value,source:document.querySelector('#source').value,pages};let r=await fetch('/api/ingest',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});let x=await r.json();document.querySelector('#ingest').innerHTML=r.ok?`<span class="pill ok">Indexed ${x.chunks} chunks across ${x.pages} pages</span>`:`<span class="pill bad">${esc(JSON.stringify(x))}</span>`;}
async function ask(){let out=document.querySelector('#result');out.textContent='Retrieving and verifying…';let body={question:document.querySelector('#question').value};let r=await fetch('/api/ask',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(body)});let x=await r.json();if(!r.ok){out.textContent=JSON.stringify(x);return}let state=x.all_claims_supported?'ok':'bad';out.innerHTML=`<p>${esc(x.answer)}</p><span class="pill ${state}">${x.all_claims_supported?'All claims verified':'Verification needs review'}</span><p class="small">${x.evidence.length} evidence chunks · trace ${esc(x.trace_id)}</p>`;load();}
async function load(){let r=await fetch('/api/traces');let ts=await r.json();let out=document.querySelector('#traces');out.innerHTML=ts.length?ts.map(t=>{let evidence=t.retrieval.map(e=>`<li><b>${esc(e.citation)}</b> · score ${e.score}<br>${esc(e.preview)}</li>`).join('');let checks=t.verification.map(v=>`<li><span class="pill ${v.supported?'ok':'bad'}">${v.supported?'supported':'rejected'}</span> ${esc(v.claim)}<br><span class="small">${esc(v.reason)}</span></li>`).join('');return `<div class="trace"><b>${esc(t.question)}</b><p>${esc(t.answer)}</p><div class="small">${t.retrieval.length} chunks retrieved · ${t.verification.filter(v=>v.supported).length}/${t.verification.length} claims supported · ${new Date(t.created_at).toLocaleString()}</div><details><summary>Retrieved memory</summary><ol>${evidence}</ol></details><details><summary>Claim verification</summary><ol>${checks}</ol></details></div>`}).join(''):'No evidence traces yet.'}load();
</script></body></html>"""
