<div align="center">

# Research Memory-Aware Agent

### A research assistant that remembers conversations, evidence, entities, workflows, summaries, tools, and every tool execution.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Oracle](https://img.shields.io/badge/Oracle-Vector_Search-F80000?style=flat-square&logo=oracle&logoColor=white)](https://www.oracle.com/database/ai-vector-search/)
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI-111111?style=flat-square&logo=openai&logoColor=white)](https://platform.openai.com/)
[![License](https://img.shields.io/badge/License-MIT-6e56cf?style=flat-square)](LICENSE)

</div>

## What it is

Most chatbots only know what fits inside their current prompt. This project gives a research agent durable memory outside the model.

Before answering, the agent retrieves the most useful parts of its history. During research it can search arXiv, read papers, save evidence, and expand older summaries. After answering, it stores the conversation, extracted entities, tool results, and successful workflow for future sessions.

The central idea is simple:

> Store everything durably, retrieve only what is relevant, and never confuse an LLM's context window with long-term memory.

This repository is a production-oriented Python rewrite of the original educational notebook. It replaces the unavailable course-specific `helper.py` with explicit, inspectable modules.

---

## Why this is more than chat history

The agent uses seven memory categories, each with a different job:

| Memory | Purpose | Retrieval |
| --- | --- | --- |
| **Conversation** | Recent user and assistant turns for continuity | Exact `thread_id`, chronological |
| **Semantic** | Papers, passages, and verified research evidence | Vector similarity |
| **Workflow** | Tool sequences and approaches that worked or failed | Vector similarity to the new goal |
| **Toolbox** | Tool names, descriptions, and JSON schemas | Semantic tool selection |
| **Entity** | People, papers, organizations, systems, and concepts | Vector similarity |
| **Summary** | Compressed older conversations with source references | Thread-scoped vector retrieval |
| **Tool log** | Complete arguments, outputs, status, and errors | Exact relational audit log |

Conversation and tool logs are relational because their order and identity matter. The other five stores use Oracle vectors because they are retrieved by meaning.

---

## Architecture

```mermaid
flowchart TD
    USER[Research question] --> CTX[Context assembler]
    ORA[(Oracle Database)] --> CTX
    CTX --> LLM[LLM decision]
    LLM -->|answer| WRITE[Memory write-back]
    LLM -->|tool call| REG[Semantic tool registry]
    REG --> TOOL[arXiv / paper / memory tool]
    TOOL --> LOG[Full tool audit log]
    LOG --> LLM
    WRITE --> ORA
    LOG --> ORA

    subgraph Seven memory types
      CONV[Conversation]
      KB[Semantic knowledge]
      WF[Workflow]
      TB[Toolbox]
      ENT[Entity]
      SUM[Summary]
      TL[Tool log]
    end

    ORA --- CONV
    ORA --- KB
    ORA --- WF
    ORA --- TB
    ORA --- ENT
    ORA --- SUM
    ORA --- TL
```

### One request, step by step

1. Retrieve recent conversation by thread.
2. Search semantic, workflow, entity, and summary memory using the current question.
3. Estimate context usage.
4. If the memory context crosses the configured threshold, summarize older conversation units and keep source references.
5. Add the current question after compaction so it is never summarized away.
6. Select up to five tools by comparing the question with embedded tool descriptions.
7. Ask the LLM for either an answer or a structured tool call.
8. Execute only registered tools and return bounded results to the model.
9. Save the complete tool result in Oracle even when the prompt receives a truncated version.
10. Store the answer, extracted entities, and tool workflow for future requests.

---

## Deterministic and agent-triggered memory

The system uses both patterns:

- **Deterministic memory:** the runtime automatically retrieves recent conversation, entities, relevant knowledge, workflows, and summaries before every model decision.
- **Agent-triggered memory:** the model can call tools such as `expand_summary` when compressed context does not contain enough detail.

Memory is therefore not loaded only once at the beginning of a chat. A fresh bounded context is assembled for every request.

---

## Context engineering

The context window is protected in three ways:

1. **Selective retrieval** — only the closest memories are loaded.
2. **Automatic offloading** — older conversation is summarized when usage crosses the configured threshold.
3. **Tool-result bounding** — full outputs stay in Oracle while only a limited prefix is returned to the LLM.

Summarization does not delete the original conversation. Each summary stores its source conversation IDs, allowing the system to audit what was compressed and expand details just in time.

---

## Semantic tool selection

Every registered tool has a name, description, and JSON parameter schema. That description is embedded into Toolbox Memory.

For each question, the agent searches Toolbox Memory and sends only the most relevant tools to the LLM. A paper question can surface arXiv tools, while a context-compaction request can surface summary tools.

This reduces prompt size and prevents the model from choosing among a large list of irrelevant functions.

Included tools:

- `search_arxiv`
- `fetch_arxiv_paper`
- `ingest_arxiv_paper`
- `save_to_knowledge_base`
- `summarize_and_store`
- `expand_summary`

---

## Project structure

```text
src/research_memory_agent/
├── agent.py          # bounded LLM/tool loop and write-back
├── app.py            # dependency assembly
├── cli.py            # command-line interface
├── config.py         # environment configuration
├── context.py        # retrieval, budgeting, and offloading
├── embeddings.py     # local Sentence Transformer adapter
├── llm.py            # OpenAI chat, summary, and entity adapter
├── memory.py         # seven-memory application API
├── models.py         # memory and tool contracts
├── prompts.py        # agent and extraction instructions
├── storage.py        # Oracle relational/vector persistence
└── tools.py          # registry and research tools

scripts/demo.py       # original five-turn notebook demonstration
tests/                # dependency-light unit tests
```

---

## Requirements

- Python 3.11+
- Oracle Database with AI Vector Search support
- An Oracle user allowed to create tables
- An OpenAI API key
- Internet access for arXiv tools and first-time embedding-model download

The default embedding model is `sentence-transformers/paraphrase-mpnet-base-v2`, which produces 768-dimensional vectors locally.

---

## Installation

```bash
git clone https://github.com/ravenscoat/Research-memeory-aware-agent.git
cd Research-memeory-aware-agent

python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Configure `.env`:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_MODEL=gpt-5-mini

ORACLE_USER=VECTOR
ORACLE_PASSWORD=your-password
ORACLE_DSN=127.0.0.1:1521/FREEPDB1
```

Secrets are loaded from the environment and must not be committed.

---

## Initialize the memory tables

```bash
research-agent --init-db
```

This operation is non-destructive and creates missing tables. To deliberately drop and recreate all seven tables:

```bash
research-agent --reset-db
```

`--reset-db` permanently deletes existing memory and should only be used for a controlled demo or test environment.

---

## Use the agent

Interactive mode:

```bash
research-agent --thread research-demo
```

One question:

```bash
research-agent --thread research-demo "Find the MemGPT paper and explain its main contribution"
```

Continue using the same thread ID to preserve conversational continuity:

```bash
research-agent --thread research-demo "Save that paper to the knowledge base"
research-agent --thread research-demo "What were its main takeaways?"
```

Run the original five-turn demonstration:

```bash
python scripts/demo.py
```

The last demo question asks, “What was my first question?” after the conversation has been summarized. It demonstrates summary retrieval and just-in-time expansion.

---

## Reliability and safety choices

- The tool loop has a strict maximum iteration count.
- Only registered Python functions can be executed.
- Tool arguments must be valid JSON objects.
- Complete tool results are durably logged before being truncated for context.
- Tool failures return evidence to the model instead of silently disappearing.
- Entity extraction is optional enrichment and cannot prevent the main answer.
- Table identifiers are fixed and validated; user text is passed as SQL bind variables.
- Database reset is explicit and never happens during normal startup.
- Tool definitions use stable IDs, so restarting the application does not create duplicate toolbox records.

---

## Test

```bash
pytest -q
ruff check .
```

The unit suite does not require Oracle or an OpenAI key. A live end-to-end run requires both services.

---

## Original notebook scenario

The design is demonstrated through one continuous research thread:

1. Find the MemGPT paper.
2. Save “the paper,” testing conversational reference resolution.
3. Ask for takeaways from semantic memory.
4. Summarize the conversation with a tool.
5. Recover the first question from compressed history.

This sequence demonstrates acquisition, continuity, semantic recall, compression, and just-in-time expansion.

---

## Roadmap

- Chunk long papers before embedding and preserve page-level citations
- Add provider-neutral LLM and embedding interfaces
- Add reranking and hybrid keyword/vector retrieval
- Add memory deduplication, supersession, confidence, and retention policies
- Add Langfuse/OpenTelemetry tracing and evaluation datasets
- Add a FastAPI service and lightweight research workspace UI

---

## Author

**Shehroz Ali** — Applied AI Engineer focused on agentic systems, persistent memory, Voice AI, and retrieval.

- [LinkedIn](https://www.linkedin.com/in/shehroz-ali-802952258)
- [GitHub](https://github.com/ravenscoat)

## License

MIT — see [LICENSE](LICENSE).
