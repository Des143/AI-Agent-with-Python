# Advanced Topics in AI Agent Development

This document covers advanced concepts and techniques in AI agent development.

## Machine Learning Integration

### 1. Supervised Learning

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class MLAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()
        
    def train(self, X, y):
        """Train the model on historical data"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict(self, state):
        """Make predictions using the trained model"""
        state_scaled = self.scaler.transform([state])
        return self.model.predict(state_scaled)[0]
```

### 2. Deep Learning

```python
import torch
import torch.nn as nn

class DeepAgent(SimpleAIAgent):
    def __init__(self, name: str, input_size: int, hidden_size: int, output_size: int):
        super().__init__(name)
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.optimizer = torch.optim.Adam(self.network.parameters())
        
    def forward(self, state):
        """Forward pass through the network"""
        return self.network(torch.FloatTensor(state))
```

## Reinforcement Learning

### 1. Q-Learning

```python
import numpy as np

class QLearningAgent(SimpleAIAgent):
    def __init__(self, name: str, n_states: int, n_actions: int):
        super().__init__(name)
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate
        
    def choose_action(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.shape[1])
        return np.argmax(self.Q[state])
        
    def update(self, state, action, reward, next_state):
        """Update Q-values"""
        best_next_action = np.argmax(self.Q[next_state])
        self.Q[state, action] += self.alpha * (
            reward + self.gamma * self.Q[next_state, best_next_action] - 
            self.Q[state, action]
        )
```

### 2. Deep Q-Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class DQNAgent(SimpleAIAgent):
    def __init__(self, name: str, state_size: int, action_size: int):
        super().__init__(name)
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        
    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        return model
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model(torch.FloatTensor(state))
        return torch.argmax(act_values).item()
```

## Natural Language Processing

### 1. Text Understanding

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

class NLPAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        self.nlp = pipeline('text-classification')
        
    def understand_text(self, text: str) -> Dict[str, Any]:
        """Process and understand natural language input"""
        return self.nlp(text)
        
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        ner = pipeline('ner')
        return ner(text)
```

### 2. Dialogue Systems

```python
class DialogueAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.conversation_history = []
        self.context = {}
        
    def respond(self, message: str) -> str:
        """Generate a response to user input"""
        self.conversation_history.append(('user', message))
        
        # Process message and update context
        self._update_context(message)
        
        # Generate response
        response = self._generate_response()
        
        self.conversation_history.append(('agent', response))
        return response
        
    def _update_context(self, message: str):
        """Update conversation context"""
        # Implement context tracking logic
        pass
        
    def _generate_response(self) -> str:
        """Generate appropriate response"""
        # Implement response generation logic
        return "I understand your message."
```

## Multi-Agent Systems

### 1. Communication

```python
class CommunicatingAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.peers = {}
        self.messages = []
        
    def send_message(self, recipient: str, message: Dict[str, Any]):
        """Send a message to another agent"""
        if recipient in self.peers:
            self.peers[recipient].receive_message(self.name, message)
            
    def receive_message(self, sender: str, message: Dict[str, Any]):
        """Receive a message from another agent"""
        self.messages.append({
            'sender': sender,
            'message': message,
            'timestamp': np.datetime64('now')
        })
```

### 2. Coordination

```python
class CoordinatingAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.team = []
        self.roles = {}
        
    def assign_role(self, agent: str, role: str):
        """Assign a role to a team member"""
        self.roles[agent] = role
        
    def coordinate_action(self, goal: str) -> Dict[str, Any]:
        """Coordinate team actions to achieve a goal"""
        action_plan = {}
        for agent, role in self.roles.items():
            action_plan[agent] = self._determine_agent_action(role, goal)
        return action_plan
```

## Advanced Memory Systems

### 1. Long-term Memory

```python
class MemoryAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.long_term_memory = {}
        self.memory_index = {}
        
    def store_memory(self, event: Dict[str, Any], importance: float):
        """Store an event in long-term memory"""
        memory_id = str(uuid.uuid4())
        self.long_term_memory[memory_id] = {
            'event': event,
            'importance': importance,
            'timestamp': np.datetime64('now')
        }
        self._update_index(memory_id, event)
        
    def retrieve_memory(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve relevant memories"""
        relevant_memories = []
        for memory_id, memory in self.long_term_memory.items():
            if self._is_relevant(memory['event'], query):
                relevant_memories.append(memory)
        return relevant_memories
```

### 2. Episodic Memory

```python
class EpisodicAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.episodes = []
        self.current_episode = None
        
    def start_episode(self, context: Dict[str, Any]):
        """Start a new episode"""
        self.current_episode = {
            'context': context,
            'events': [],
            'start_time': np.datetime64('now')
        }
        
    def record_event(self, event: Dict[str, Any]):
        """Record an event in the current episode"""
        if self.current_episode:
            self.current_episode['events'].append({
                'event': event,
                'timestamp': np.datetime64('now')
            })
            
    def end_episode(self, outcome: Dict[str, Any]):
        """End the current episode"""
        if self.current_episode:
            self.current_episode['outcome'] = outcome
            self.current_episode['end_time'] = np.datetime64('now')
            self.episodes.append(self.current_episode)
            self.current_episode = None
```

## Performance Optimization

### 1. Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class ParallelAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
        
    def process_batch(self, states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple states in parallel"""
        futures = [self.pool.submit(self.process, state) for state in states]
        return [f.result() for f in futures]
```

### 2. Caching

```python
from functools import lru_cache

class CachingAgent(SimpleAIAgent):
    @lru_cache(maxsize=1000)
    def process_state(self, state: tuple) -> Dict[str, Any]:
        """Process state with caching"""
        # Convert state tuple back to dict
        state_dict = dict(state)
        return self.process(state_dict)
```

## Deployment Considerations

### 1. API Service

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

class APIAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        
    @app.post("/process")
    async def process_request(self, request: Dict[str, Any]):
        """Process API request"""
        self.perceive(request['state'])
        decision = self.process()
        result = self.act(decision)
        return {"result": result}
```

### 2. Monitoring

```python
import logging
from prometheus_client import start_http_server, Counter, Gauge

class MonitoredAgent(SimpleAIAgent):
    def __init__(self, name: str):
        super().__init__(name)
        self.actions_counter = Counter('agent_actions_total', 'Number of actions taken')
        self.confidence_gauge = Gauge('agent_confidence', 'Current confidence level')
        
    def act(self, decision: Dict[str, Any]) -> str:
        self.actions_counter.inc()
        self.confidence_gauge.set(decision['confidence'])
        return super().act(decision)
``` 