# API Reference

This document provides detailed information about the classes, methods, and types used in the AI agent framework.

## SimpleAIAgent Class

The main agent class that implements basic AI agent functionality.

### Constructor

```python
SimpleAIAgent(name: str, learning_rate: float = 0.1)
```

**Parameters:**
- `name` (str): The name of the agent
- `learning_rate` (float): The rate at which the agent learns (default: 0.1)

### Methods

#### perceive
```python
perceive(environment_state: Dict[str, Any]) -> None
```
Processes and stores the current environment state.

**Parameters:**
- `environment_state` (Dict[str, Any]): The current state of the environment

**Example:**
```python
agent.perceive({
    'goal': 'learn',
    'resources': ['books', 'internet']
})
```

#### process
```python
process() -> Dict[str, Any]
```
Processes the current state and makes a decision.

**Returns:**
- Dict[str, Any]: A decision containing:
  - `action` (str): The chosen action
  - `confidence` (float): Confidence in the decision

**Example:**
```python
decision = agent.process()
print(decision['action'])  # 'study'
print(decision['confidence'])  # 0.8
```

#### act
```python
act(decision: Dict[str, Any]) -> str
```
Executes the chosen action.

**Parameters:**
- `decision` (Dict[str, Any]): The decision to execute

**Returns:**
- str: A message describing the action taken

**Example:**
```python
result = agent.act(decision)
print(result)  # "Agent MyAgent performed action: study with confidence: 0.8"
```

#### learn
```python
learn(feedback: Dict[str, Any]) -> None
```
Updates the agent's knowledge based on feedback.

**Parameters:**
- `feedback` (Dict[str, Any]): Feedback containing:
  - `context` (str): The context of the feedback
  - `action` (str): The action taken
  - `success` (float): The success rate

**Example:**
```python
agent.learn({
    'context': 'learning_session',
    'action': 'study',
    'success': 0.9
})
```

#### get_memory
```python
get_memory() -> List[Dict[str, Any]]
```
Retrieves the agent's memory.

**Returns:**
- List[Dict[str, Any]]: List of memory entries

**Example:**
```python
memory = agent.get_memory()
for entry in memory:
    print(entry['state'], entry['timestamp'])
```

#### get_knowledge
```python
get_knowledge() -> Dict[str, Any]
```
Retrieves the agent's knowledge base.

**Returns:**
- Dict[str, Any]: The knowledge base

**Example:**
```python
knowledge = agent.get_knowledge()
print(knowledge['learning_session']['success_rate'])
```

## Type Definitions

### EnvironmentState
```python
EnvironmentState = Dict[str, Any]
```
Represents the state of the environment.

**Fields:**
- `goal` (str): The current goal
- `resources` (List[str]): Available resources
- `constraints` (Dict[str, Any]): Environmental constraints

### Decision
```python
Decision = Dict[str, Any]
```
Represents a decision made by the agent.

**Fields:**
- `action` (str): The chosen action
- `confidence` (float): Confidence in the decision
- `reasoning` (str): Explanation of the decision

### Feedback
```python
Feedback = Dict[str, Any]
```
Represents feedback for the agent.

**Fields:**
- `context` (str): The context of the feedback
- `action` (str): The action taken
- `success` (float): The success rate
- `details` (Dict[str, Any]): Additional feedback details

## Error Handling

### InvalidStateError
```python
class InvalidStateError(Exception):
    """Raised when the environment state is invalid"""
    pass
```

### InvalidDecisionError
```python
class InvalidDecisionError(Exception):
    """Raised when a decision is invalid"""
    pass
```

## Constants

### Default Values
```python
DEFAULT_LEARNING_RATE = 0.1
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MAX_MEMORY_SIZE = 1000
```

### Action Types
```python
ACTION_TYPES = {
    'WAIT': 'wait',
    'STUDY': 'study',
    'MOVE': 'move',
    'EXPLORE': 'explore'
}
```

## Utility Functions

### format_state
```python
format_state(state: EnvironmentState) -> str
```
Formats the environment state for display.

**Parameters:**
- `state` (EnvironmentState): The state to format

**Returns:**
- str: Formatted state string

### calculate_confidence
```python
calculate_confidence(state: EnvironmentState, action: str) -> float
```
Calculates confidence in a decision.

**Parameters:**
- `state` (EnvironmentState): Current state
- `action` (str): Proposed action

**Returns:**
- float: Confidence value between 0 and 1

## Examples

### Creating and Using an Agent
```python
from src.simple_agent import SimpleAIAgent

# Create agent
agent = SimpleAIAgent(name="LearningBot")

# Perceive environment
agent.perceive({
    'goal': 'learn',
    'resources': ['books', 'internet']
})

# Make decision
decision = agent.process()

# Take action
result = agent.act(decision)

# Learn from feedback
agent.learn({
    'context': 'learning_session',
    'action': decision['action'],
    'success': 0.9
})
```

### Handling Errors
```python
try:
    agent.perceive(invalid_state)
except InvalidStateError as e:
    print(f"Error: {str(e)}")
    # Handle error
```

### Using Memory and Knowledge
```python
# Get memory
memory = agent.get_memory()
for entry in memory[-5:]:  # Last 5 entries
    print(f"State: {entry['state']}")
    print(f"Action: {entry.get('action', 'None')}")

# Get knowledge
knowledge = agent.get_knowledge()
for context, info in knowledge.items():
    print(f"Context: {context}")
    print(f"Success Rate: {info['success_rate']}")
``` 