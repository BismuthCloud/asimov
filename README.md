[![PyPI - Version](https://img.shields.io/pypi/v/asimov_agents.svg)](https://pypi.org/project/asimov_agents)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/asimov_agents.svg)](https://pypi.org/project/asimov_agents)

# Asimov Agents

A Python framework for building AI agent systems with robust task management, inference capabilities, and caching.

## System Overview

Asimov Agents is composed of three main components:

1. **Task Graph System**
   - Manages task execution flow and dependencies
   - Supports different task states (WAITING, EXECUTING, COMPLETE, FAILED, PARTIAL)
   - Uses Pydantic models for robust data validation
   - Unique task identification via UUIDs

2. **Inference Clients**
   - Supports multiple LLM providers:
     - Anthropic Claude (via API)
     - AWS Bedrock
   - Features:
     - Streaming responses
     - Tool/function calling capabilities
     - Token usage tracking
     - OpenTelemetry instrumentation
     - Prompt caching support

3. **Caching System**
   - Abstract Cache interface with Redis implementation
   - Features:
     - Key-value storage with JSON serialization
     - Prefix/suffix namespacing
     - Pub/sub messaging via mailboxes
     - Bulk operations (get_all, clear)
     - Async interface

## Component Interactions

### Task Management
- Tasks are created and tracked using the `Task` class
- Each task has:
  - Unique ID
  - Type and objective
  - Parameters dictionary
  - Status tracking
  - Result/error storage

### Inference Pipeline
1. Messages are formatted with appropriate roles (SYSTEM, USER, ASSISTANT, TOOL_RESULT)
2. Inference clients handle:
   - Message formatting
   - API communication
   - Response streaming
   - Token accounting
   - Error handling

### Caching Layer
- Redis cache provides:
  - Fast key-value storage
  - Message queuing
  - Namespace management
  - Atomic operations

## Setup and Configuration

### Redis Cache Setup
```python
cache = RedisCache(
    host="localhost",  # Redis host
    port=6379,        # Redis port
    db=0,             # Database number
    password=None,    # Optional password
    default_prefix="" # Optional key prefix
)
```

### Inference Client Setup
```python
# Anthropic Client
client = AnthropicInferenceClient(
    model="claude-3",
    api_key="your-api-key",
    api_url="https://api.anthropic.com/v1/messages"
)

# AWS Bedrock Client
client = BedrockInferenceClient(
    model="anthropic.claude-3",
    region_name="us-east-1"
)
```

### Task Creation
```python
task = Task(
    type="processing",
    objective="Process data",
    params={"input": "data"}
)
```

## Performance Considerations

### Caching
- Use appropriate key prefixes/suffixes for namespace isolation
- Consider timeout settings for blocking operations
- Monitor Redis memory usage
- Use raw mode when bypassing JSON serialization

### Inference
- Token usage is tracked automatically
- Streaming reduces time-to-first-token
- Tool calls support iteration limits
- Prompt caching can improve response times

### Task Management
- Tasks support partial failure states
- Use UUIDs for guaranteed uniqueness
- Status transitions are atomic

## Development

### Running Tests
```bash
pytest tests/
```

### Required Dependencies
- Redis server
- Python 3.7+
- See requirements.txt for Python packages

## License

[Add your license information here]
