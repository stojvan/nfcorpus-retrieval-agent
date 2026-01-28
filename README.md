# NFCorpus Retrieval Agent

An LLM-powered [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) purple agent that retrieves and ranks biomedical documents from the NFCorpus dataset.

## Overview

This agent uses **PydanticAI** with an LLM (GPT-4.1) to provide intelligent document retrieval:
- The LLM receives biomedical queries and decides how to search the vector database
- The LLM has access to MCP tools (via function calling) to query the NFCorpus database
- The LLM analyzes search results and decides on final document selection and ranking
- This enables intelligent query reformulation, multi-step retrieval, and relevance reasoning

## Architecture

```
User Query → A2A Server → Agent → PydanticAI + LLM → MCP Tools → Vector DB
                                        ↓
                                  Ranked Results
```

**Components:**
- **A2A Server** (`src/server.py`) - Handles agent-to-agent communication
- **Agent** (`src/agent.py`) - LLM-powered retrieval logic with PydanticAI
- **MCP Tools** - Interface to NFCorpus vector database
- **LLM** - GPT-4 for intelligent query processing and ranking

## Project Structure

```
src/
├─ server.py      # A2A server setup and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # LLM-powered retrieval agent implementation
└─ messenger.py   # A2A messaging utilities
tests/
└─ test_agent.py  # A2A conformance and retrieval tests
Dockerfile        # Docker configuration
pyproject.toml    # Python dependencies (includes PydanticAI, OpenAI)
docs/
└─ purple_agent_spec.md  # Purple agent specification
```

## Environment Variables

- **`OPENAI_API_KEY`** (required) - OpenAI API key for GPT-4 access
- **`MCP_SERVER_URL`** (optional) - MCP server URL (default: `http://mcp-server:8000`)
- **`LLM_MODEL`** (optional) - LLM model to use (default: `openai:gpt-4`)

## Request/Response Format

### Request (QueryRequest)
```json
{
  "query": "What are the effects of calcium on bone health?",
  "top_k": 5
}
```

### Response (RetrievalResponse)
```json
{
  "doc_ids": ["MED-123", "MED-456", "MED-789", "MED-234", "MED-567"]
}
```

Document IDs are returned in ranked order (most relevant first).

## Running Locally

```bash
# Install dependencies
uv sync

# Set required environment variables
export OPENAI_API_KEY="your-api-key-here"
export MCP_SERVER_URL="http://localhost:8000"  # Optional, defaults to http://mcp-server:8000

# Run the server
uv run src/server.py --port 9010
```

The agent will be available at `http://localhost:9010`

## Running with Docker

```bash
# Build the image
docker build -t nfcorpus-retrieval-agent .

# Run the container
docker run -p 9010:9010 \
  -e OPENAI_API_KEY="your-api-key-here" \
  -e MCP_SERVER_URL="http://host.docker.internal:8000" \
  nfcorpus-retrieval-agent
```

## Testing

### A2A Conformance Tests

Run A2A conformance tests to verify the agent follows the protocol:

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9010
```

### Testing with MCP Server

To test the full retrieval pipeline, you need the MCP server running:

```bash
# Start MCP server infrastructure (from green agent repo)
docker-compose up -d qdrant mcp-server

# Test MCP server directly
curl -X POST http://localhost:8000/search_nfcorpus \
  -H "Content-Type: application/json" \
  -d '{"query": "calcium and bone health", "top_k": 5}'

# Run your agent with MCP server
export OPENAI_API_KEY="your-api-key-here"
export MCP_SERVER_URL="http://localhost:8000"
uv run src/server.py --port 9010
```

### Integration Testing with Green Agent

Send an evaluation request to the green agent:

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

## Dataset Information

- **Dataset**: NFCorpus (Nutrition Facts Corpus)
- **Domain**: Biomedical/nutrition
- **Documents**: ~3,633 biomedical articles
- **Test Queries**: 323 queries
- **Relevance Judgments**: 3-level scale (0, 1, 2)
- **Source**: [BEIR benchmark](https://huggingface.co/datasets/BeIR/nfcorpus)

## Evaluation Metrics

The green agent evaluates retrieval performance using **NDCG@5** (Normalized Discounted Cumulative Gain at rank 5):
- **Range**: 0.0 to 1.0 (higher is better)
- **Considers**: Both relevance and ranking position
- **Relevance levels**: 0 (not relevant), 1 (partially relevant), 2 (highly relevant)

## Cost Considerations

This agent uses OpenAI's GPT-4 API, which incurs costs per query:
- Each query involves LLM calls for query processing and result analysis
- The LLM may make multiple tool calls to search the database
- Consider implementing caching or rate limiting for production use
- Monitor API usage in your OpenAI dashboard

## How It Works

1. **Query Reception**: Agent receives a `QueryRequest` via A2A protocol
2. **LLM Processing**: PydanticAI agent with GPT-4 analyzes the query
3. **Tool Calling**: LLM calls `search_nfcorpus` MCP tool (potentially multiple times)
4. **Result Analysis**: LLM examines returned documents and their relevance scores
5. **Ranking Decision**: LLM decides final document ranking based on relevance
6. **Response**: Agent returns `RetrievalResponse` with ranked document IDs

The LLM can:
- Reformulate queries for better results
- Make multiple searches with different variations
- Analyze document titles, text, and scores
- Apply domain knowledge to improve ranking

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image to GitHub Container Registry.

**Important**: Add `OPENAI_API_KEY` in Settings → Secrets and variables → Actions → Repository secrets for CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/nfcorpus-retrieval-agent:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/nfcorpus-retrieval-agent:1.0.0
ghcr.io/<your-username>/nfcorpus-retrieval-agent:1
```

## References

- **Purple Agent Spec**: [`docs/purple_agent_spec.md`](docs/purple_agent_spec.md)
- **A2A Protocol**: https://a2a-protocol.org/latest/
- **PydanticAI**: https://ai.pydantic.dev/
- **NFCorpus Dataset**: https://huggingface.co/datasets/BeIR/nfcorpus

## License

See LICENSE file for details.
