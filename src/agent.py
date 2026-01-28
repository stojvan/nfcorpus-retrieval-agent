import os
import json
from typing import List
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent, RunContext
import httpx

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart
from a2a.utils import get_message_text, new_agent_text_message


class QueryRequest(BaseModel):
    query: str = Field(description="Search query text")
    top_k: int = Field(default=5, description="Number of documents to retrieve")


class RetrievalResponse(BaseModel):
    doc_ids: List[str] = Field(description="List of retrieved document IDs in ranked order")


class DocumentRanking(BaseModel):
    """Structured output for LLM agent - ranked list of document IDs."""
    doc_ids: List[str] = Field(description="List of document IDs in ranked order (most relevant first)")


class MCPDeps(BaseModel):
    """Dependencies for MCP client access."""
    model_config = {"arbitrary_types_allowed": True}
    
    mcp_url: str
    http_client: httpx.AsyncClient


SYSTEM_PROMPT = """You are a biomedical document retrieval expert. Your task is to find the most relevant 
documents from the NFCorpus database for a given query.

You have access to a search tool that queries a vector database of biomedical articles.
You can:
1. Reformulate the query if needed for better results
2. Make multiple searches with different query variations
3. Analyze the returned documents and their relevance scores
4. Decide on the final ranking of documents

Your output must be a list of document IDs in ranked order (most relevant first).
Return exactly the number of documents requested (top_k), or fewer if insufficient matches.
Only return the document IDs as a JSON array, nothing else."""


class Agent:
    def __init__(self):
        self.mcp_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
        self.llm_model = os.getenv("LLM_MODEL", "openai:gpt-4.1")
        
        self.llm_agent = PydanticAgent(
            model=self.llm_model,
            system_prompt=SYSTEM_PROMPT,
            deps_type=MCPDeps,
            output_type=DocumentRanking,
        )
        
        @self.llm_agent.tool
        async def search_nfcorpus(ctx: RunContext[MCPDeps], query: str, top_k: int = 5) -> list[dict]:
            """Search NFCorpus biomedical database for relevant documents.
            
            Args:
                query: Search query text
                top_k: Number of documents to retrieve
                
            Returns:
                List of documents with doc_id, score, title, and text
            """
            try:
                response = await ctx.deps.http_client.post(
                    f"{ctx.deps.mcp_url}/search_nfcorpus",
                    json={"query": query, "top_k": top_k},
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("results", [])
            except Exception as e:
                raise RuntimeError(f"MCP search failed: {e}")

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Execute LLM-powered document retrieval.

        Args:
            message: The incoming message containing QueryRequest
            updater: Report progress and results
        """
        try:
            input_text = get_message_text(message)
            
            await updater.update_status(
                TaskState.working, 
                new_agent_text_message("Parsing query request...")
            )
            
            request = QueryRequest.model_validate_json(input_text)
            
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(f"Searching for: {request.query}")
            )
            
            async with httpx.AsyncClient() as http_client:
                deps = MCPDeps(mcp_url=self.mcp_url, http_client=http_client)
                
                result = await self.llm_agent.run(
                    f"Find the top {request.top_k} most relevant documents for this query: {request.query}",
                    deps=deps
                )
                
                # Extract document IDs from structured output
                doc_ids = result.output.doc_ids if result.output else []
                
                response = RetrievalResponse(doc_ids=doc_ids[:request.top_k])
            
            await updater.add_artifact(
                parts=[Part(root=DataPart(data=response.model_dump()))],
                name="retrieval_results",
            )
            
        except json.JSONDecodeError as e:
            await updater.failed(
                new_agent_text_message(f"Invalid request format: {e}")
            )
        except Exception as e:
            await updater.failed(
                new_agent_text_message(f"Retrieval failed: {e}")
            )
