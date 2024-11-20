[![PyPI - Version](https://img.shields.io/pypi/v/asimov_agents.svg)](https://pypi.org/project/asimov_agents)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/asimov_agents.svg)](https://pypi.org/project/asimov_agents)

# Asimov

Asimov is a powerful graph-based inference and caching system with Redis support. It provides a flexible framework for building and executing inference workflows while efficiently managing data through Redis caching.

## Features

- Graph-based workflow execution
- Multiple inference client implementations (Anthropic, OpenAI, Vertex AI, Bedrock)
- Redis-backed caching system with mock implementation for testing
- Asynchronous operation support
- Mailbox system for message passing between components
- Comprehensive test coverage

## Installation

```bash
pip install asimov_agents
```

### Requirements

- Python 3.7+
- Redis server (for production use)
- Optional: Various AI service API keys depending on which inference clients you use

## Quick Start

Here's a basic example of setting up and using Asimov:

```python
from asimov.caches.redis_cache import RedisCache
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

# Initialize cache
cache = RedisCache(host='localhost', port=6379)

# Create inference client
client = InferenceClient(cache=cache)

# Use the system
messages = [
    ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=ChatRole.USER, content="Hello!")
]
response = await client.get_generation(messages)
```

## Core Components

### Cache System

The cache system provides a Redis-backed storage solution with support for both simple key-value operations and more complex messaging patterns:

```python
# Basic cache operations
await cache.set("key", {"value": 42})
result = await cache.get("key")

# Mailbox functionality
await cache.publish_to_mailbox("mailbox_id", {"message": "hello"})
message = await cache.get_message("mailbox_id", timeout=1)
```

### Graph Module

The graph module allows you to create and execute complex workflows using different types of modules:

```python
from asimov.graph import AgentModule, ModuleType

class CustomModule(AgentModule):
    name = "custom_module"
    type = ModuleType.EXECUTOR
    
    async def process(self, cache, semaphore):
        # Your custom processing logic here
        return {"status": "success", "result": "processing complete"}

# Execute module
module = CustomModule()
result = await module.run(cache, semaphore)
```

Available module types:
- PLANNER: For modules that plan or coordinate workflows
- EXECUTOR: For modules that perform specific tasks
- DISCRIMINATOR: For modules that make decisions
- OBSERVER: For modules that monitor or log
- SUBGRAPH: For composite modules containing other modules
- FLOW_CONTROL: For modules that control workflow execution

### Inference Clients

Asimov supports multiple inference clients for different AI services:

- Anthropic (Claude)
- OpenAI
- Vertex AI
- Bedrock

Each client provides a consistent interface for making inference requests:

```python
# Example with Anthropic client
from asimov.services.inference_clients import AnthropicInferenceClient

client = AnthropicInferenceClient(
    model="claude-2",
    api_key="your-api-key"
)

response = await client.get_generation(messages)

# Streaming responses
async for chunk in client.connect_and_listen(messages):
    print(chunk)
```

## Testing

Asimov provides a mock Redis implementation for testing:

```python
from asimov.caches.mock_redis_cache import MockRedisCache

# Use mock cache in tests
cache = MockRedisCache()
await cache.set("test_key", "test_value")

# Clean up after tests
await cache.clear()
await cache.close()
```

## Performance Considerations

### Redis Caching Benefits
- Reduces API calls to inference services
- Provides fast access to frequently used data
- Supports distributed caching across multiple instances

### Graph Execution Strategy
- Parallel execution of independent modules
- Efficient message passing through Redis mailboxes
- Configurable retry and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement your changes
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/asimov.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for all public functions and classes
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Redis team for their excellent caching solution
- Various AI service providers for their APIs
- The open source community for their contributions
Asimov is a powerful graph-based inference and caching system with Redis support. It provides a flexible framework for building and executing inference workflows while efficiently managing data through Redis caching.

## Features

- Graph-based workflow execution
- Multiple inference client implementations (Anthropic, OpenAI, Vertex AI, Bedrock)
- Redis-backed caching system with mock implementation for testing
- Asynchronous operation support
- Mailbox system for message passing between components
- Comprehensive test coverage

## Installation

```bash
pip install asimov_agents
```

### Requirements

- Python 3.7+
- Redis server (for production use)
- Optional: Various AI service API keys depending on which inference clients you use

## Quick Start

Here's a basic example of setting up and using Asimov:

```python
from asimov.caches.redis_cache import RedisCache
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

# Initialize cache
cache = RedisCache(host='localhost', port=6379)

# Create inference client
client = InferenceClient(cache=cache)

# Use the system
messages = [
    ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=ChatRole.USER, content="Hello!")
]
response = await client.get_generation(messages)
```

## Core Components

### Cache System

The cache system provides a Redis-backed storage solution with support for both simple key-value operations and more complex messaging patterns:

```python
# Basic cache operations
await cache.set("key", {"value": 42})
result = await cache.get("key")

# Mailbox functionality
await cache.publish_to_mailbox("mailbox_id", {"message": "hello"})
message = await cache.get_message("mailbox_id", timeout=1)
```

### Graph Module

The graph module allows you to create and execute complex workflows using different types of modules:

```python
from asimov.graph import AgentModule, ModuleType

class CustomModule(AgentModule):
    name = "custom_module"
    type = ModuleType.EXECUTOR
    
    async def process(self, cache, semaphore):
        # Your custom processing logic here
        return {"status": "success", "result": "processing complete"}

# Execute module
module = CustomModule()
result = await module.run(cache, semaphore)
```

Available module types:
- PLANNER: For modules that plan or coordinate workflows
- EXECUTOR: For modules that perform specific tasks
- DISCRIMINATOR: For modules that make decisions
- OBSERVER: For modules that monitor or log
- SUBGRAPH: For composite modules containing other modules
- FLOW_CONTROL: For modules that control workflow execution

### Inference Clients

Asimov supports multiple inference clients for different AI services:

- Anthropic (Claude)
- OpenAI
- Vertex AI
- Bedrock

Each client provides a consistent interface for making inference requests:

```python
# Example with Anthropic client
from asimov.services.inference_clients import AnthropicInferenceClient

client = AnthropicInferenceClient(
    model="claude-2",
    api_key="your-api-key"
)

response = await client.get_generation(messages)

# Streaming responses
async for chunk in client.connect_and_listen(messages):
    print(chunk)
```

## Testing

Asimov provides a mock Redis implementation for testing:

```python
from asimov.caches.mock_redis_cache import MockRedisCache

# Use mock cache in tests
cache = MockRedisCache()
await cache.set("test_key", "test_value")

# Clean up after tests
await cache.clear()
await cache.close()
```

## Performance Considerations

### Redis Caching Benefits
- Reduces API calls to inference services
- Provides fast access to frequently used data
- Supports distributed caching across multiple instances

### Graph Execution Strategy
- Parallel execution of independent modules
- Efficient message passing through Redis mailboxes
- Configurable retry and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement your changes
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/asimov.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for all public functions and classes
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Redis team for their excellent caching solution
- Various AI service providers for their APIs
- The open source community for their contributions
Asimov is a powerful graph-based inference and caching system with Redis support. It provides a flexible framework for building and executing inference workflows while efficiently managing data through Redis caching.

## Features

- Graph-based workflow execution
- Multiple inference client implementations (Anthropic, OpenAI, Vertex AI, Bedrock)
- Redis-backed caching system with mock implementation for testing
- Asynchronous operation support
- Mailbox system for message passing between components
- Comprehensive test coverage

## Installation

```bash
pip install asimov_agents
```

### Requirements

- Python 3.7+
- Redis server (for production use)
- Optional: Various AI service API keys depending on which inference clients you use

## Quick Start

Here's a basic example of setting up and using Asimov:

```python
from asimov.caches.redis_cache import RedisCache
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

# Initialize cache
cache = RedisCache(host='localhost', port=6379)

# Create inference client
client = InferenceClient(cache=cache)

# Use the system
messages = [
    ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=ChatRole.USER, content="Hello!")
]
response = await client.get_generation(messages)
```

## Core Components

### Cache System

The cache system provides a Redis-backed storage solution with support for both simple key-value operations and more complex messaging patterns:

```python
# Basic cache operations
await cache.set("key", {"value": 42})
result = await cache.get("key")

# Mailbox functionality
await cache.publish_to_mailbox("mailbox_id", {"message": "hello"})
message = await cache.get_message("mailbox_id", timeout=1)
```

### Graph Module

The graph module allows you to create and execute complex workflows using different types of modules:

```python
from asimov.graph import AgentModule, ModuleType

class CustomModule(AgentModule):
    name = "custom_module"
    type = ModuleType.EXECUTOR
    
    async def process(self, cache, semaphore):
        # Your custom processing logic here
        return {"status": "success", "result": "processing complete"}

# Execute module
module = CustomModule()
result = await module.run(cache, semaphore)
```

Available module types:
- PLANNER: For modules that plan or coordinate workflows
- EXECUTOR: For modules that perform specific tasks
- DISCRIMINATOR: For modules that make decisions
- OBSERVER: For modules that monitor or log
- SUBGRAPH: For composite modules containing other modules
- FLOW_CONTROL: For modules that control workflow execution

### Inference Clients

Asimov supports multiple inference clients for different AI services:

- Anthropic (Claude)
- OpenAI
- Vertex AI
- Bedrock

Each client provides a consistent interface for making inference requests:

```python
# Example with Anthropic client
from asimov.services.inference_clients import AnthropicInferenceClient

client = AnthropicInferenceClient(
    model="claude-2",
    api_key="your-api-key"
)

response = await client.get_generation(messages)

# Streaming responses
async for chunk in client.connect_and_listen(messages):
    print(chunk)
```

## Testing

Asimov provides a mock Redis implementation for testing:

```python
from asimov.caches.mock_redis_cache import MockRedisCache

# Use mock cache in tests
cache = MockRedisCache()
await cache.set("test_key", "test_value")

# Clean up after tests
await cache.clear()
await cache.close()
```

## Performance Considerations

### Redis Caching Benefits
- Reduces API calls to inference services
- Provides fast access to frequently used data
- Supports distributed caching across multiple instances

### Graph Execution Strategy
- Parallel execution of independent modules
- Efficient message passing through Redis mailboxes
- Configurable retry and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement your changes
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/asimov.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for all public functions and classes
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Redis team for their excellent caching solution
- Various AI service providers for their APIs
- The open source community for their contributions

Asimov is a powerful graph-based inference and caching system with Redis support. It provides a flexible framework for building and executing inference workflows while efficiently managing data through Redis caching.

## Features

- Graph-based workflow execution
- Multiple inference client implementations (Anthropic, OpenAI, Vertex AI, Bedrock)
- Redis-backed caching system with mock implementation for testing
- Asynchronous operation support
- Mailbox system for message passing between components
- Comprehensive test coverage

## Installation

```bash
pip install asimov_agents
```

### Requirements

- Python 3.7+
- Redis server (for production use)
- Optional: Various AI service API keys depending on which inference clients you use

## Quick Start

Here's a basic example of setting up and using Asimov:


pip install asimov_agents
```

### Requirements

- Python 3.7+
- Redis server (for production use)
- Optional: Various AI service API keys depending on which inference clients you use

## Quick Start

Here's a basic example of setting up and using Asimov:

```python
from asimov.caches.redis_cache import RedisCache
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

# Initialize cache
cache = RedisCache(host='localhost', port=6379)

# Create inference client
client = InferenceClient(cache=cache)

# Use the system
messages = [
    ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=ChatRole.USER, content="Hello!")
]
response = await client.get_generation(messages)
```

## Core Components

### Cache System

The cache system provides a Redis-backed storage solution with support for both simple key-value operations and more complex messaging patterns:

```python
# Basic cache operations
await cache.set("key", {"value": 42})
result = await cache.get("key")

# Mailbox functionality
await cache.publish_to_mailbox("mailbox_id", {"message": "hello"})
message = await cache.get_message("mailbox_id", timeout=1)
```

### Graph Module

The graph module allows you to create and execute complex workflows using different types of modules:

```python
from asimov.graph import AgentModule, ModuleType

class CustomModule(AgentModule):
    name = "custom_module"
    type = ModuleType.EXECUTOR
    
    async def process(self, cache, semaphore):
        # Your custom processing logic here
        return {"status": "success", "result": "processing complete"}

# Execute module
module = CustomModule()
result = await module.run(cache, semaphore)
```

Available module types:
- PLANNER: For modules that plan or coordinate workflows
- EXECUTOR: For modules that perform specific tasks
- DISCRIMINATOR: For modules that make decisions
- OBSERVER: For modules that monitor or log
- SUBGRAPH: For composite modules containing other modules
- FLOW_CONTROL: For modules that control workflow execution

### Inference Clients

Asimov supports multiple inference clients for different AI services:

- Anthropic (Claude)
- OpenAI
- Vertex AI
- Bedrock

Each client provides a consistent interface for making inference requests:

```python
# Example with Anthropic client
from asimov.services.inference_clients import AnthropicInferenceClient

client = AnthropicInferenceClient(
    model="claude-2",
    api_key="your-api-key"
)

response = await client.get_generation(messages)

# Streaming responses
async for chunk in client.connect_and_listen(messages):
    print(chunk)
```

## Testing

Asimov provides a mock Redis implementation for testing:

```python
from asimov.caches.mock_redis_cache import MockRedisCache

# Use mock cache in tests
cache = MockRedisCache()
await cache.set("test_key", "test_value")

# Clean up after tests
await cache.clear()
await cache.close()
```

## Performance Considerations

### Redis Caching Benefits
- Reduces API calls to inference services
- Provides fast access to frequently used data
- Supports distributed caching across multiple instances

### Graph Execution Strategy
- Parallel execution of independent modules
- Efficient message passing through Redis mailboxes
- Configurable retry and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement your changes
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/asimov.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for all public functions and classes
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Redis team for their excellent caching solution
- Various AI service providers for their APIs
- The open source community for their contributions

Asimov is a powerful graph-based inference and caching system with Redis support. It provides a flexible framework for building and executing inference workflows while efficiently managing data through Redis caching.

## Features

- Graph-based workflow execution
- Multiple inference client implementations (Anthropic, OpenAI, Vertex AI, Bedrock)
- Redis-backed caching system with mock implementation for testing
- Asynchronous operation support
- Mailbox system for message passing between components
- Comprehensive test coverage

## Installation

```bash
pip install asimov_agents
```

### Requirements

- Python 3.7+
- Redis server (for production use)
- Optional: Various AI service API keys depending on which inference clients you use

## Quick Start

Here's a basic example of setting up and using Asimov:

```python
from asimov.caches.redis_cache import RedisCache
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

# Initialize cache
cache = RedisCache(host='localhost', port=6379)

# Create inference client
client = InferenceClient(cache=cache)

# Use the system
messages = [
    ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
    ChatMessage(role=ChatRole.USER, content="Hello!")
]
response = await client.get_generation(messages)
```

## Core Components

### Cache System

The cache system provides a Redis-backed storage solution with support for both simple key-value operations and more complex messaging patterns:

```python
# Basic cache operations
await cache.set("key", {"value": 42})
result = await cache.get("key")

# Mailbox functionality
await cache.publish_to_mailbox("mailbox_id", {"message": "hello"})
message = await cache.get_message("mailbox_id", timeout=1)
```

### Graph Module

The graph module allows you to create and execute complex workflows using different types of modules:

```python
from asimov.graph import AgentModule, ModuleType

class CustomModule(AgentModule):
    name = "custom_module"
    type = ModuleType.EXECUTOR
    
    async def process(self, cache, semaphore):
        # Your custom processing logic here
        return {"status": "success", "result": "processing complete"}

# Execute module
module = CustomModule()
result = await module.run(cache, semaphore)
```

Available module types:
- PLANNER: For modules that plan or coordinate workflows
- EXECUTOR: For modules that perform specific tasks
- DISCRIMINATOR: For modules that make decisions
- OBSERVER: For modules that monitor or log
- SUBGRAPH: For composite modules containing other modules
- FLOW_CONTROL: For modules that control workflow execution

### Inference Clients

Asimov supports multiple inference clients for different AI services:

- Anthropic (Claude)
- OpenAI
- Vertex AI
- Bedrock

Each client provides a consistent interface for making inference requests:

```python
# Example with Anthropic client
from asimov.services.inference_clients import AnthropicInferenceClient

client = AnthropicInferenceClient(
    model="claude-2",
    api_key="your-api-key"
)

response = await client.get_generation(messages)

# Streaming responses
async for chunk in client.connect_and_listen(messages):
    print(chunk)
```

## Testing

Asimov provides a mock Redis implementation for testing:

```python
from asimov.caches.mock_redis_cache import MockRedisCache

# Use mock cache in tests
cache = MockRedisCache()
await cache.set("test_key", "test_value")

# Clean up after tests
await cache.clear()
await cache.close()
```

## Performance Considerations

### Redis Caching Benefits
- Reduces API calls to inference services
- Provides fast access to frequently used data
- Supports distributed caching across multiple instances

### Graph Execution Strategy
- Parallel execution of independent modules
- Efficient message passing through Redis mailboxes
- Configurable retry and error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement your changes
5. Submit a pull request

### Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/asimov.git

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Include docstrings for all public functions and classes
- Write unit tests for new functionality

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The Redis team for their excellent caching solution
- Various AI service providers for their APIs
- The open source community for their contributions
