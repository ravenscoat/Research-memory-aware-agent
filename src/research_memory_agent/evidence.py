from __future__ import annotations

import io
import re
import urllib.request
import uuid
from dataclasses import asdict
from typing import Any

from pypdf import PdfReader

from .models import ClaimVerification, EvidenceChunk, MemoryType


def chunk_page(text: str, *, size: int = 1_200, overlap: int = 180) -> list[str]:
    """Split one page into overlap-preserving chunks without cutting words where possible."""
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = clean.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        chunks.append(clean[start:end].strip())
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


class EvidenceGroundedResearch:
    """Chunk, retrieve, answer, verify, and trace research evidence."""

    def __init__(self, *, store: Any, model: Any) -> None:
        self.store = store
        self.model = model

    def ingest_pdf(
        self, pdf: bytes, *, title: str, source: str, document_id: str | None = None
    ) -> dict[str, Any]:
        reader = PdfReader(io.BytesIO(pdf))
        pages = [page.extract_text() or "" for page in reader.pages]
        return self.ingest_pages(pages, title=title, source=source, document_id=document_id)

    def ingest_arxiv(self, arxiv_id: str, *, title: str = "") -> dict[str, Any]:
        safe_id = arxiv_id.strip().removeprefix("https://arxiv.org/abs/").removeprefix(
            "http://arxiv.org/abs/"
        )
        if not safe_id or any(not (char.isalnum() or char in ".v/-") for char in safe_id):
            raise ValueError("Invalid arXiv ID")
        source = f"https://arxiv.org/abs/{safe_id}"
        request = urllib.request.Request(
            f"https://arxiv.org/pdf/{safe_id}",
            headers={"User-Agent": "research-memory-agent/0.2"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            pdf = response.read()
        return self.ingest_pdf(
            pdf,
            title=title or f"arXiv:{safe_id}",
            source=source,
            document_id=f"arxiv:{safe_id}",
        )

    def ingest_pages(
        self, pages: list[str], *, title: str, source: str, document_id: str | None = None
    ) -> dict[str, Any]:
        document_id = document_id or str(uuid.uuid4())
        superseded = self.store.deactivate_document_chunks(document_id)
        chunk_ids: list[str] = []
        for page_number, page_text in enumerate(pages, start=1):
            for chunk_index, text in enumerate(chunk_page(page_text)):
                chunk_ids.append(
                    self.store.append_vector(
                        MemoryType.SEMANTIC,
                        text,
                        metadata={
                            "kind": "evidence_chunk",
                            "document_id": document_id,
                            "title": title,
                            "source": source,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                        },
                    )
                )
        return {
            "document_id": document_id,
            "pages": len(pages),
            "chunks": len(chunk_ids),
            "superseded_chunks": superseded,
            "chunk_ids": chunk_ids,
        }

    def retrieve(self, question: str, *, limit: int = 6) -> list[EvidenceChunk]:
        matches = self.store.search_hybrid_evidence(question, limit=limit)
        evidence = []
        for item in matches:
            meta = item.metadata
            if meta.get("kind") != "evidence_chunk":
                continue
            evidence.append(
                EvidenceChunk(
                    id=item.id,
                    document_id=str(meta["document_id"]),
                    title=str(meta["title"]),
                    source=str(meta["source"]),
                    page_number=int(meta["page_number"]),
                    chunk_index=int(meta["chunk_index"]),
                    text=item.text,
                    score=float(item.score or 0),
                )
            )
        return evidence

    def answer(self, question: str, *, thread_id: str = "evidence") -> dict[str, Any]:
        evidence = self.retrieve(question)
        if not evidence:
            answer = "I do not have indexed evidence for that question."
            verification: list[ClaimVerification] = []
        else:
            context = "\n\n".join(
                f"{chunk.citation}\n{chunk.text}" for chunk in evidence
            )
            answer = self._generate_answer(question, context)
            verification = self.verify(answer, evidence)
            if any(not item.supported for item in verification):
                failures = "\n".join(
                    f"- {item.claim}: {item.reason}" for item in verification if not item.supported
                )
                answer = self._generate_answer(question, context, failures=failures)
                verification = self.verify(answer, evidence)
            if any(not item.supported for item in verification):
                supported_claims = [item.claim for item in verification if item.supported]
                answer = " ".join(supported_claims) or (
                    "The retrieved evidence was insufficient to produce a fully verified answer."
                )
                verification = self.verify(answer, evidence) if supported_claims else []

        retrieval = [
            {"id": item.id, "citation": item.citation, "source": item.source,
             "score": round(item.score, 4), "preview": item.text[:260]}
            for item in evidence
        ]
        verification_data = [asdict(item) for item in verification]
        trace_id = self.store.write_evidence_trace(
            thread_id=thread_id, question=question, retrieval=retrieval,
            answer=answer, verification=verification_data,
        )
        return {
            "trace_id": trace_id,
            "answer": answer,
            "evidence": retrieval,
            "verification": verification_data,
            "all_claims_supported": bool(verification_data) and all(item["supported"] for item in verification_data),
        }

    def _generate_answer(self, question: str, context: str, *, failures: str = "") -> str:
        repair = ""
        if failures:
            repair = (
                "\nA previous draft failed verification for these claims:\n"
                + failures
                + "\nRewrite it so every factual sentence has its own exact citation."
            )
        message = self.model.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the supplied evidence. Every factual sentence must end with "
                        "one or more exact evidence citations in square brackets. Keep the answer concise. "
                        "If the evidence does not support an answer, say so. Do not use outside knowledge."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nEvidence:\n{context}{repair}",
                },
            ]
        )
        return message.content.strip() or "I could not produce a grounded answer."

    def verify(self, answer: str, evidence: list[EvidenceChunk]) -> list[ClaimVerification]:
        by_citation = {item.citation: item for item in evidence}
        # Do not split inside page citations such as "p. 2"; ordinary sentence starts use capitals.
        claims = [item.strip() for item in re.split(r"(?<=[.!?])(?=\s+[A-Z])", answer) if item.strip()]
        verified: list[ClaimVerification] = []
        for claim in claims:
            citations = re.findall(r"\[[^\]]+\]", claim)
            citation = citations[0] if citations else None
            source = by_citation.get(citation or "")
            if source is None:
                verified.append(ClaimVerification(claim, citation, False, "Missing or unknown evidence citation."))
                continue
            words = {word.lower() for word in re.findall(r"[A-Za-z0-9]{4,}", claim)}
            support_words = {word.lower() for word in re.findall(r"[A-Za-z0-9]{4,}", source.text)}
            overlap = len(words & support_words) / max(1, len(words))
            verified.append(
                ClaimVerification(
                    claim, citation, overlap >= 0.15,
                    f"Lexical evidence overlap: {overlap:.2f}.",
                )
            )
        return verified
