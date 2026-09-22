from research_memory_agent.llm import OllamaChatModel


def test_tool_arguments_are_converted_to_ollama_objects():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search_arxiv",
                        "arguments": '{"query":"agent memory","max_results":1}',
                    },
                }
            ],
        }
    ]

    converted = OllamaChatModel._ollama_messages(messages)

    assert converted[0]["tool_calls"][0]["function"]["arguments"] == {
        "query": "agent memory",
        "max_results": 1,
    }
    assert isinstance(messages[0]["tool_calls"][0]["function"]["arguments"], str)
