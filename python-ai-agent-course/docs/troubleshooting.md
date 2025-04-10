# Troubleshooting Guide

This guide helps you identify and resolve common issues when developing and deploying AI agents.

## Common Issues and Solutions

### 1. Agent Not Learning

**Symptoms:**
- Agent's performance not improving
- Knowledge base not updating
- Same decisions being made repeatedly

**Solutions:**
1. Check learning rate:
```python
# Adjust learning rate
agent = SimpleAIAgent(name="TestAgent", learning_rate=0.01)  # Try different values
```

2. Verify feedback mechanism:
```python
# Ensure feedback is being properly processed
feedback = {
    'context': 'learning_session',
    'action': 'study',
    'success': 0.9
}
agent.learn(feedback)
print(agent.get_knowledge())  # Check if knowledge was updated
```

3. Implement logging:
```python
import logging

class LoggingAgent(SimpleAIAgent):
    def learn(self, feedback: Dict[str, Any]) -> None:
        logging.info(f"Received feedback: {feedback}")
        super().learn(feedback)
        logging.info(f"Updated knowledge: {self.get_knowledge()}")
```

### 2. Memory Issues

**Symptoms:**
- Memory growing too large
- Slow performance
- Memory errors

**Solutions:**
1. Implement memory limits:
```python
class MemoryLimitedAgent(SimpleAIAgent):
    def __init__(self, name: str, max_memory: int = 1000):
        super().__init__(name)
        self.max_memory = max_memory
        
    def perceive(self, environment_state: Dict[str, Any]) -> None:
        super().perceive(environment_state)
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]
```

2. Use efficient data structures:
```python
from collections import deque

class EfficientMemoryAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.memory = deque(maxlen=1000)
```

### 3. Performance Problems

**Symptoms:**
- Slow decision making
- High CPU usage
- Long response times

**Solutions:**
1. Implement caching:
```python
from functools import lru_cache

class CachingAgent(SimpleAIAgent):
    @lru_cache(maxsize=1000)
    def process(self) -> Dict[str, Any]:
        return super().process()
```

2. Use parallel processing:
```python
from concurrent.futures import ThreadPoolExecutor

class ParallelAgent(SimpleAIAgent):
    def process_batch(self, states):
        with ThreadPoolExecutor() as executor:
            return list(executor.map(self.process, states))
```

### 4. Decision Making Issues

**Symptoms:**
- Poor decision quality
- Inconsistent actions
- Low confidence scores

**Solutions:**
1. Add decision validation:
```python
class ValidatingAgent(SimpleAIAgent):
    def process(self) -> Dict[str, Any]:
        decision = super().process()
        if not self._validate_decision(decision):
            decision['action'] = 'wait'
            decision['confidence'] = 0.0
        return decision
        
    def _validate_decision(self, decision: Dict[str, Any]) -> bool:
        return (
            'action' in decision and
            'confidence' in decision and
            0 <= decision['confidence'] <= 1
        )
```

2. Implement fallback strategies:
```python
class FallbackAgent(SimpleAIAgent):
    def process(self) -> Dict[str, Any]:
        try:
            return super().process()
        except Exception:
            return {
                'action': 'wait',
                'confidence': 0.0,
                'reason': 'fallback'
            }
```

## Debugging Tips

### 1. Logging

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class DebugAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.logger = logging.getLogger(name)
        
    def perceive(self, state: Dict[str, Any]) -> None:
        self.logger.debug(f"Perceiving state: {state}")
        super().perceive(state)
        
    def process(self) -> Dict[str, Any]:
        decision = super().process()
        self.logger.debug(f"Made decision: {decision}")
        return decision
```

### 2. State Inspection

```python
class InspectableAgent(SimpleAIAgent):
    def get_state(self) -> Dict[str, Any]:
        """Return current agent state for inspection"""
        return {
            'current_state': self.current_state,
            'memory_size': len(self.memory),
            'knowledge_size': len(self.knowledge_base),
            'last_action': self.memory[-1]['action'] if self.memory else None
        }
```

### 3. Performance Monitoring

```python
import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper

class MonitoredAgent(SimpleAIAgent):
    @timeit
    def process(self) -> Dict[str, Any]:
        return super().process()
```

## Error Handling

### 1. Input Validation

```python
class ValidatingAgent(SimpleAIAgent):
    def perceive(self, environment_state: Dict[str, Any]) -> None:
        if not isinstance(environment_state, dict):
            raise ValueError("Environment state must be a dictionary")
        if 'goal' not in environment_state:
            raise ValueError("Environment state must contain a 'goal'")
        super().perceive(environment_state)
```

### 2. Graceful Degradation

```python
class RobustAgent(SimpleAIAgent):
    def act(self, decision: Dict[str, Any]) -> str:
        try:
            return super().act(decision)
        except Exception as e:
            logging.error(f"Error in act: {str(e)}")
            return "Error: Action failed, entering safe mode"
```

## Testing Strategies

### 1. Unit Tests

```python
import unittest

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = SimpleAIAgent("TestAgent")
        
    def test_perception(self):
        state = {'goal': 'learn'}
        self.agent.perceive(state)
        self.assertEqual(self.agent.current_state, state)
        
    def test_decision(self):
        decision = self.agent.process()
        self.assertIn('action', decision)
        self.assertIn('confidence', decision)
```

### 2. Integration Tests

```python
class TestAgentIntegration(unittest.TestCase):
    def test_full_cycle(self):
        agent = SimpleAIAgent("TestAgent")
        state = {'goal': 'learn'}
        
        # Perception
        agent.perceive(state)
        
        # Processing
        decision = agent.process()
        
        # Action
        result = agent.act(decision)
        
        # Learning
        feedback = {
            'context': 'test',
            'action': decision['action'],
            'success': 0.9
        }
        agent.learn(feedback)
        
        # Verify results
        self.assertIsNotNone(result)
        self.assertGreater(len(agent.get_knowledge()), 0)
```

## Performance Optimization

### 1. Memory Management

```python
class OptimizedAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self._cleanup_threshold = 1000
        
    def _cleanup_memory(self):
        if len(self.memory) > self._cleanup_threshold:
            # Keep only the most recent entries
            self.memory = self.memory[-self._cleanup_threshold:]
            
    def perceive(self, state: Dict[str, Any]) -> None:
        super().perceive(state)
        self._cleanup_memory()
```

### 2. Caching Strategies

```python
from functools import lru_cache

class CachingAgent(SimpleAIAgent):
    @lru_cache(maxsize=1000)
    def _process_state(self, state_hash: int) -> Dict[str, Any]:
        return self.process()
        
    def process(self) -> Dict[str, Any]:
        state_hash = hash(str(self.current_state))
        return self._process_state(state_hash)
```

## Deployment Issues

### 1. API Integration

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

class APIAgent(SimpleAIAgent):
    @app.post("/process")
    async def process_request(self, request: Dict[str, Any]):
        try:
            self.perceive(request['state'])
            decision = self.process()
            result = self.act(decision)
            return {"result": result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
```

### 2. Monitoring

```python
from prometheus_client import Counter, Gauge

class MonitoredAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.errors = Counter('agent_errors_total', 'Number of errors')
        self.latency = Gauge('agent_latency_seconds', 'Processing latency')
        
    def process(self) -> Dict[str, Any]:
        start = time.time()
        try:
            result = super().process()
            self.latency.set(time.time() - start)
            return result
        except Exception:
            self.errors.inc()
            raise 