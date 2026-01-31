import os
import logging
import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent as PydanticAgent, RunContext
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from schemas import QueryRequest, RetrievalResponse

logger = logging.getLogger(__name__)

class MCPSearchResult(BaseModel):
    """Result from MCP search_nfcorpus tool."""
    doc_id: str
    score: float


class MCPSearchResponse(BaseModel):
    """Response from MCP server."""
    results: list[MCPSearchResult]


class AgentDeps(BaseModel):
    """Dependencies for PydanticAI agent."""
    model_config = {"arbitrary_types_allowed": True}
    
    mcp_server_url: str
    http_client: httpx.AsyncClient


class Agent:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.mcp_server_url = os.getenv("MCP_SERVER_URL", "http://green-agent:8000")
        logger.info(f"Agent initialized with MCP_SERVER_URL: {self.mcp_server_url}")
        
        self.pydantic_agent = PydanticAgent(
            "openai:gpt-4o",
            deps_type=AgentDeps,
            output_type=RetrievalResponse,
            system_prompt="""You are an intelligent biomedical document retrieval agent.
Your task is to analyze biomedical queries and retrieve the most relevant documents from the NFCorpus database.

You have access to a vector search tool that can find documents based on semantic similarity.
Use the search_nfcorpus tool to query the database with appropriate search terms.

Analyze the query, decide on the best search strategy, and return the most relevant document IDs."""
        )
        
        @self.pydantic_agent.tool
        async def search_nfcorpus(ctx: RunContext[AgentDeps], query: str, top_k: int = 5) -> dict:
            """Search the NFCorpus database for relevant biomedical documents.
            
            Args:
                ctx: Runtime context with dependencies
                query: The search query for finding relevant documents
                top_k: Number of top results to return (default: 5)
                
            Returns:
                Dictionary with search results containing doc_ids and scores
            """
            mcp_url = f"{ctx.deps.mcp_server_url}/search_nfcorpus"
            logger.info(f"Calling MCP server at {mcp_url} with query='{query}', top_k={top_k}")
            try:
                response = await ctx.deps.http_client.post(
                    mcp_url,
                    json={
                        "query": query,
                        "top_k": top_k
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
                logger.info(f"MCP server response: {data}")
                
                mcp_response = MCPSearchResponse.model_validate(data)
                
                return {
                    "results": [
                        {"doc_id": r.doc_id, "score": r.score}
                        for r in mcp_response.results
                    ]
                }
            except Exception as e:
                logger.error(f"MCP search failed: {str(e)}")
                return {"error": f"MCP search failed: {str(e)}", "results": []}

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Process biomedical query and return retrieved document IDs.
        
        Args:
            message: The incoming message containing a query request
            updater: Report progress and results
        """
        input_text = get_message_text(message)
        
        try:
            query_request = QueryRequest.model_validate_json(input_text)
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid query format: {e}"))
            return
        
        await updater.update_status(
            TaskState.working, 
            new_agent_text_message(f"Searching NFCorpus for: {query_request.query}")
        )
        
        async with httpx.AsyncClient() as http_client:
            deps = AgentDeps(
                mcp_server_url=self.mcp_server_url,
                http_client=http_client
            )
            
            try:
                result = await self.pydantic_agent.run(
                    f"Find the top {query_request.top_k} most relevant documents for this biomedical query: {query_request.query}",
                    deps=deps
                )
                
                retrieval_response = result.output
                
                await updater.add_artifact(
                    parts=[
                        Part(root=DataPart(data={
                            "doc_ids": retrieval_response.doc_ids
                        }))
                    ],
                    name="Retrieved Documents"
                )
                
            except Exception as e:
                await updater.reject(
                    new_agent_text_message(f"Error during retrieval: {str(e)}")
                )