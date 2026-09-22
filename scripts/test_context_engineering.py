"""Live check for automatic context offloading and lossless summary references."""

import uuid

from research_memory_agent.app import build_application
from research_memory_agent.context import ContextAssembler

thread_id = "context-check-" + uuid.uuid4().hex[:8]
question = "CURRENT-QUESTION-MUST-SURVIVE: What facts were retained?"
app = build_application()
try:
    for index in range(8):
        role = "user" if index % 2 == 0 else "assistant"
        app.store.append_conversation(
            thread_id,
            role,
            f"SOURCE-MARKER-{index} " + (f"context block {index} " * 90),
        )

    assembler = ContextAssembler(app.agent.memory, token_limit=600, offload_threshold=0.50)
    bundle = assembler.build(question, thread_id)
    assert bundle.offloaded is True
    assert bundle.usage_ratio <= 0.50
    assert bundle.text.startswith("# Question\n" + question)

    with app.store.cursor() as cursor:
        cursor.execute(
            "SELECT count(*) FROM research_conversational_memory "
            "WHERE thread_id = %s AND summarized = true",
            (thread_id,),
        )
        summarized_messages = cursor.fetchone()[0]
        cursor.execute(
            "SELECT id, source_ids FROM research_summary_memory "
            "WHERE thread_id = %s ORDER BY created_at DESC LIMIT 1",
            (thread_id,),
        )
        summary_id, source_ids = cursor.fetchone()

    expanded = app.agent.memory.expand_summary(str(summary_id))
    assert summarized_messages == 8
    assert len(source_ids) == 8
    assert "SOURCE-MARKER-0" in expanded and "SOURCE-MARKER-7" in expanded
    print("OFFLOADED=" + str(bundle.offloaded))
    print("QUESTION_PRESERVED=true")
    print("SUMMARIZED_MESSAGES=" + str(summarized_messages))
    print("SUMMARY_SOURCE_IDS=" + str(len(source_ids)))
    print("SOURCE_EXPANSION=true")
    print("POST_COMPRESSION_TOKENS=" + str(bundle.estimated_tokens))
    print("MEMORY_USAGE_RATIO=" + str(round(bundle.usage_ratio, 3)))
finally:
    app.close()
