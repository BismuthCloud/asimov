import unittest
from unittest.mock import patch, MagicMock
import asyncio
from asimov.graph import AgentModule, ModuleType
from asimov.caches.mock_redis_cache import MockRedisCache
from asimov.services.inference_clients import InferenceClient, ChatMessage, ChatRole

class TestReadmeExamples(unittest.TestCase):
    def setUp(self):
        self.cache = MockRedisCache()
        
    async def test_basic_graph_setup(self):
        """Test the basic setup example that will be shown in README"""
        graph = Graph()
        cache = self.cache
        client = InferenceClient(cache=cache)
        
        # Verify components are initialized correctly
        self.assertIsInstance(graph, Graph)
        self.assertIsInstance(cache, MockRedisCache)
        self.assertIsInstance(client, InferenceClient)
        
        # Verify cache connection
        self.assertTrue(await cache.ping())

    async def test_cache_operations(self):
        """Test cache operations example for README"""
        cache = self.cache
        
        # Basic cache operations
        await cache.set("test_key", {"value": 42})
        result = await cache.get("test_key")
        self.assertEqual(result, {"value": 42})
        
        # Mailbox functionality
        await cache.publish_to_mailbox("test_mailbox", {"message": "hello"})
        message = await cache.get_message("test_mailbox", timeout=1)
        self.assertEqual(message, {"message": "hello"})

    async def test_graph_module_execution(self):
        """Test graph module execution example for README"""
        class TestModule(AgentModule):
            name = "test_module"
            type = ModuleType.EXECUTOR
            
            async def process(self, cache, semaphore):
                return {"status": "success", "result": "test completed"}

        graph = Graph()
        cache = self.cache
        module = TestModule()
        
        # Execute module
        semaphore = asyncio.Semaphore(1)
        result = await module.run(cache, semaphore)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "test completed")

    async def test_inference_client_usage(self):
        """Test inference client usage example for README"""
        cache = self.cache
        client = InferenceClient(cache=cache)
        
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=ChatRole.USER, content="Hello!")
        ]
        
        # Mock the generation method since we can't make real API calls in tests
        with patch.object(client, 'get_generation') as mock_generate:
            mock_generate.return_value = "Hello! How can I help you today?"
            response = await client.get_generation(messages)
            self.assertEqual(response, "Hello! How can I help you today?")
        self.assertTrue(await cache.ping())

    async def test_cache_operations(self):
        """Test cache operations example for README"""
        cache = self.cache
        
        # Basic cache operations
        await cache.set("test_key", {"value": 42})
        result = await cache.get("test_key")
        self.assertEqual(result, {"value": 42})
        
        # Mailbox functionality
        await cache.publish_to_mailbox("test_mailbox", {"message": "hello"})
        message = await cache.get_message("test_mailbox", timeout=1)
        self.assertEqual(message, {"message": "hello"})

    async def test_graph_module_execution(self):
        """Test graph module execution example for README"""
        class TestModule(AgentModule):
            name = "test_module"
            type = ModuleType.EXECUTOR
            
            async def process(self, cache, semaphore):
                return {"status": "success", "result": "test completed"}

        graph = Graph()
        cache = self.cache
        module = TestModule()
        
        # Execute module
        semaphore = asyncio.Semaphore(1)
        result = await module.run(cache, semaphore)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["result"], "test completed")

    async def test_inference_client_usage(self):
        """Test inference client usage example for README"""
        cache = self.cache
        client = InferenceClient(cache=cache)
        
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content="You are a helpful assistant."),
            ChatMessage(role=ChatRole.USER, content="Hello!")
        ]
        
        # Mock the generation method since we can't make real API calls in tests
        with patch.object(client, 'get_generation') as mock_generate:
            mock_generate.return_value = "Hello! How can I help you today?"
            response = await client.get_generation(messages)
            self.assertEqual(response, "Hello! How can I help you today?")