import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--mcp-server-url", type=str, help="URL of the MCP server for document retrieval")
    args = parser.parse_args()
    
    # Set MCP_SERVER_URL environment variable if provided via CLI
    if args.mcp_server_url:
        import os
        os.environ["MCP_SERVER_URL"] = args.mcp_server_url
    
    skill = AgentSkill(
        id="nfcorpus_retrieval",
        name="NFCorpus Biomedical Document Retrieval",
        description="A purple agent that retrieves relevant biomedical documents from the NFCorpus database using intelligent LLM-based search with MCP tools",
        tags=["retrieval", "biomedical", "nfcorpus", "vector-search", "purple-agent"],
        examples=[
            'Input: {"query": "calcium and bone health", "top_k": 5} → Output: {"doc_ids": ["MED-123", "MED-456", "MED-789", "MED-234", "MED-567"]}',
            'Input: {"query": "diabetes treatment options", "top_k": 3} → Output: {"doc_ids": ["MED-890", "MED-345", "MED-678"]}'
        ]
    )

    agent_card = AgentCard(
        name="NFCorpus Retrieval Agent",
        description="Purple agent that uses PydanticAI with GPT-4o to intelligently retrieve biomedical documents from NFCorpus database via MCP tools. Accepts biomedical queries and returns ranked document IDs.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
