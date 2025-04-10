from typing import List, Dict, Any
import numpy as np

class SimpleAIAgent:
    """
    A simple AI agent that demonstrates basic agent architecture.
    This agent can:
    1. Perceive its environment
    2. Process information
    3. Make decisions
    4. Take actions
    """
    
    def __init__(self, name: str, learning_rate: float = 0.1):
        self.name = name
        self.learning_rate = learning_rate
        self.memory: List[Dict[str, Any]] = []
        self.knowledge_base: Dict[str, Any] = {}
        
    def perceive(self, environment_state: Dict[str, Any]) -> None:
        """
        Perceive the current state of the environment
        """
        self.current_state = environment_state
        self.memory.append({
            'state': environment_state,
            'timestamp': np.datetime64('now')
        })
        
    def process(self) -> Dict[str, Any]:
        """
        Process the current state and make a decision
        """
        # Simple decision making based on current state
        decision = {
            'action': 'wait',  # Default action
            'confidence': 0.0
        }
        
        if 'goal' in self.current_state:
            if self.current_state['goal'] == 'learn':
                decision['action'] = 'study'
                decision['confidence'] = 0.8
            elif self.current_state['goal'] == 'explore':
                decision['action'] = 'move'
                decision['confidence'] = 0.6
                
        return decision
    
    def act(self, decision: Dict[str, Any]) -> str:
        """
        Execute the chosen action
        """
        action = decision['action']
        confidence = decision['confidence']
        
        # Log the action
        self.memory[-1]['action'] = action
        self.memory[-1]['confidence'] = confidence
        
        return f"Agent {self.name} performed action: {action} with confidence: {confidence}"
    
    def learn(self, feedback: Dict[str, Any]) -> None:
        """
        Learn from feedback and update knowledge
        """
        if 'success' in feedback:
            # Update knowledge based on successful actions
            self.knowledge_base[feedback['context']] = {
                'action': feedback['action'],
                'success_rate': feedback['success']
            }
            
    def get_memory(self) -> List[Dict[str, Any]]:
        """
        Retrieve the agent's memory
        """
        return self.memory
    
    def get_knowledge(self) -> Dict[str, Any]:
        """
        Retrieve the agent's knowledge base
        """
        return self.knowledge_base 