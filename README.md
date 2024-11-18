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

### Graph System Architecture
- **Module Types**
  - `PLANNER`: Strategic task planning modules
  - `EXECUTOR`: Task execution modules
  - `DISCRIMINATOR`: Decision making modules
  - `OBSERVER`: Monitoring and logging modules
  - `SUBGRAPH`: Nested graph structure modules
  - `FLOW_CONTROL`: Execution flow control modules

- **Node Configuration**
  ```python
  node_config = NodeConfig(
      parallel=True,              # Enable parallel execution
      condition="task.ready",     # Conditional execution
      retry_on_failure=True,      # Enable retry mechanism
      max_retries=3,             # Maximum retry attempts
      max_visits=5,              # Maximum node visits
      inputs=["data"],           # Required inputs
      outputs=["result"]         # Expected outputs
  )
  ```

- **Flow Control Features**
  - Conditional branching based on task state
  - Dynamic node addition during execution
  - Dependency chain management
  - Automatic cleanup of completed nodes
  - Execution state tracking and recovery

- **Snapshot System**
  - State preservation modes:
    - `NEVER`: No snapshots
    - `ONCE`: Single snapshot
    - `ALWAYS`: Continuous snapshots
  - Captures:
    - Agent state
    - Cache contents
    - Task status
    - Execution history
  - Configurable storage location via `ASIMOV_SNAPSHOT`

- **Error Handling**
  - Automatic retry mechanisms
  - Partial completion states
  - Failed chain tracking
  - Detailed error reporting
  - Timeout management

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

### Task and Graph Setup
```python
# Create a task
task = Task(
    type="processing",
    objective="Process data",
    params={"input": "data"}
)

# Create nodes with different module types
planner_node = Node(
    name="planner",
    type=ModuleType.PLANNER,
    modules=[PlannerModule()],
    node_config=NodeConfig(
        parallel=False,
        max_retries=3
    )
)

executor_node = Node(
    name="executor",
    type=ModuleType.EXECUTOR,
    modules=[ExecutorModule()],
    dependencies=["planner"]
)

flow_control = Node(
    name="flow_control",
    type=ModuleType.FLOW_CONTROL,
    modules=[FlowControlModule(
        flow_config=FlowControlConfig(
            decisions=[
                FlowDecision(
                    next_node="executor",
                    condition="task.ready == true"
                )
            ],
            default="planner"
        )
    )]
)

# Set up the agent
agent = Agent(
    cache=RedisCache(),
    max_concurrent_tasks=5,
    max_total_iterations=100
)

# Add nodes to the agent
agent.add_multiple_nodes([planner_node, executor_node, flow_control])

# Run the task
await agent.run_task(task)
```

## Advanced Features

### Middleware System
```python
class LoggingMiddleware(Middleware):
    async def process(self, data: Dict[str, Any], cache: Cache) -> Dict[str, Any]:
        print(f"Processing data: {data}")
        return data

node = Node(
    name="executor",
    type=ModuleType.EXECUTOR,
    modules=[ExecutorModule()],
    config=ModuleConfig(
        middlewares=[LoggingMiddleware()],
        timeout=30.0
    )
)
```

### Execution State Management
- Tracks execution history
- Supports execution plan compilation
- Enables dynamic plan modification
- Provides state restoration capabilities
```python
# Access execution state
current_plan = agent.execution_state.current_plan
execution_history = agent.execution_state.execution_history
total_iterations = agent.execution_state.total_iterations

# Compile execution plans
full_plan = agent.compile_execution_plan()
partial_plan = agent.compile_execution_plan_from("specific_node")

# Restore from snapshot
await agent.run_from_snapshot(snapshot_dir)
```

### OpenTelemetry Integration
- Automatic span creation for nodes
- Execution tracking
- Performance monitoring
- Error tracing
```python
node = Node(
    name="traced_node",
    type=ModuleType.EXECUTOR,
    modules=[ExecutorModule()],
    trace=True  # Enable OpenTelemetry tracing
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
