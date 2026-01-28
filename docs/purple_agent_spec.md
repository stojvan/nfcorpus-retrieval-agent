# Purple Agent Specification

## Overview

This document specifies the interface that purple agents (retrieval systems) must implement to be evaluated by the NFCorpus Retrieval Evaluator green agent.

## Communication Protocol

Purple agents communicate with the green agent using the **A2A (Agent-to-Agent) protocol**. The green agent sends queries to the purple agent and expects document IDs in response.

## Request Format

The green agent sends a JSON message with the following structure:

```json
{
  "query": "What are the effects of calcium on bone health?",
  "top_k": 5
}
```

### Request Schema (Pydantic)

```python
class QueryRequest(BaseModel):
    query: str = Field(description="Search query text")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
```

### Fields

- **query** (string, required): The search query text
- **top_k** (integer, optional, default: 5): Number of documents to retrieve

## Response Format

Purple agents must respond with a JSON message containing a list of document IDs:

```json
{
  "doc_ids": ["MED-123", "MED-456", "MED-789", "MED-234", "MED-567"]
}
```

### Response Schema (Pydantic)

```python
class RetrievalResponse(BaseModel):
    doc_ids: List[str] = Field(description="List of retrieved document IDs in ranked order")
```

### Fields

- **doc_ids** (array of strings, required): List of NFCorpus document IDs in ranked order (most relevant first)

### Important Notes

1. **Document IDs must be valid NFCorpus IDs** (e.g., "MED-123", "MED-456")
2. **Order matters**: Documents should be ranked by relevance (most relevant first)
3. **Return exactly `top_k` documents** (or fewer if not enough documents match)
4. **No duplicates**: Each document ID should appear at most once

## MCP Server Access

Purple agents can access the NFCorpus vector database through the MCP server.

### MCP Server Configuration

- **URL**: `http://mcp-server:8000` (Docker Compose internal) or `http://localhost:8000` (external)
- **Protocol**: MCP (Model Context Protocol)

### Available Tools

#### search_nfcorpus

Search NFCorpus biomedical documents using vector similarity.

**Tool Call:**
```json
{
  "tool": "search_nfcorpus",
  "parameters": {
    "query": "calcium and bone health",
    "top_k": 5
  }
}
```

**Tool Response:**
```json
{
  "results": [
    {
      "doc_id": "MED-123",
      "score": 0.856,
      "title": "Effects of Calcium Supplementation on Bone Density",
      "text": "This study examines the relationship between calcium intake..."
    },
    {
      "doc_id": "MED-456",
      "score": 0.823,
      "title": "Bone Health in Postmenopausal Women",
      "text": "Postmenopausal women are at increased risk..."
    }
  ]
}
```

## Example Purple Agent Implementation

Here's a minimal example of a purple agent:

```python
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, DataPart
from pydantic import BaseModel
from typing import List
import json

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class RetrievalResponse(BaseModel):
    doc_ids: List[str]

class PurpleAgent:
    def __init__(self):
        # Initialize MCP client connection
        self.mcp_client = MCPClient("http://mcp-server:8000")
    
    async def run(self, message: Message, updater: TaskUpdater):
        # Parse incoming query request
        message_text = get_message_text(message)
        request = QueryRequest.model_validate_json(message_text)
        
        # Call MCP search tool
        mcp_response = await self.mcp_client.call_tool(
            "search_nfcorpus",
            {"query": request.query, "top_k": request.top_k}
        )
        
        # Extract document IDs
        doc_ids = [result["doc_id"] for result in mcp_response["results"]]
        
        # Return response
        response = RetrievalResponse(doc_ids=doc_ids)
        await updater.add_artifact(
            parts=[Part(root=DataPart(data=response.model_dump()))],
            name="retrieval_results"
        )
```

## Evaluation Process

1. **Green agent samples queries** from NFCorpus test set using configured random seed
2. **For each query**, green agent sends QueryRequest to purple agent
3. **Purple agent retrieves documents** and returns document IDs
4. **Green agent calculates NDCG@5** by comparing retrieved documents to ground truth
5. **Green agent reports aggregate metrics** (mean, median, std, min, max NDCG@5)

## Evaluation Metrics

The primary metric is **NDCG@5** (Normalized Discounted Cumulative Gain at rank 5):

- **Range**: 0.0 to 1.0
- **Higher is better**: 1.0 = perfect ranking
- **Considers**: Both relevance and ranking position
- **Relevance levels**: 0 (not relevant), 1 (partially relevant), 2 (highly relevant)

## Testing Your Purple Agent

### Local Testing with Docker Compose

1. Start the infrastructure:
```bash
docker-compose up -d qdrant mcp-server
```

2. Start your purple agent:
```bash
# In your purple agent repository
docker run -p 9010:9010 my-purple-agent
```

3. Send a test evaluation request to the green agent:
```bash
curl -X POST http://localhost:9009/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "participants": {
      "retrieval_agent": "http://host.docker.internal:9010"
    },
    "config": {
      "num_queries": 10,
      "top_k": 5,
      "random_seed": 42
    }
  }'
```

### Testing MCP Server Directly

```bash
# Test search functionality
curl -X POST http://localhost:8000/search_nfcorpus \
  -H "Content-Type: application/json" \
  -d '{"query": "calcium and bone health", "top_k": 5}'
```

## Requirements Checklist

- [ ] Implements A2A protocol server
- [ ] Accepts QueryRequest format
- [ ] Returns RetrievalResponse format with valid NFCorpus document IDs
- [ ] Documents are ranked by relevance
- [ ] Handles errors gracefully
- [ ] Responds within reasonable timeout (< 60 seconds per query)
- [ ] Can be deployed as Docker container
- [ ] Exposes agent card at root endpoint

## Common Pitfalls

1. **Invalid document IDs**: Ensure IDs match NFCorpus format (e.g., "MED-123")
2. **Wrong response format**: Must return `{"doc_ids": [...]}`, not just a list
3. **Unranked results**: Documents should be ordered by relevance
4. **Timeout issues**: Optimize retrieval to respond quickly
5. **MCP connection errors**: Ensure MCP server URL is correct

## Support

For questions or issues:
- Review the green agent repository: https://github.com/your-org/nfcorpus-retrieval-evaluator
- Check MCP server documentation: `mcp_server/README.md`
- See example request/response in `docs/examples/`

## Dataset Information

- **Dataset**: NFCorpus (Nutrition Facts Corpus)
- **Domain**: Biomedical/nutrition
- **Documents**: ~3,633 biomedical articles
- **Test Queries**: 323 queries
- **Relevance Judgments**: 3-level scale (0, 1, 2)
- **Source**: BEIR benchmark (https://huggingface.co/datasets/BeIR/nfcorpus)
