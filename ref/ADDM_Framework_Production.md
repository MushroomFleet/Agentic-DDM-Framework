# Agentic DDM Framework (ADDM) - Production Version

A production-ready implementation of an agentic decision-making system based on Drift-Diffusion Model principles, using OpenRouter's Grok 4 Fast for evidence generation.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Core Implementation](#core-implementation)
- [Usage Examples](#usage-examples)
- [Testing & Validation](#testing--validation)
- [Configuration Guide](#configuration-guide)

---

## Architecture Overview

### Key Improvements from Original Design

1. **Structured Evidence Parsing**: JSON-based outputs with schema validation
2. **Fixed Drift Rate Management**: No state mutation bugs
3. **Multi-Alternative DDM**: Racing accumulators for 2+ options
4. **Robust Error Handling**: Retries, timeouts, fallbacks
5. **Parallel API Calls**: Concurrent evidence generation
6. **Validation Metrics**: A/B testing against argmax baseline
7. **Comprehensive Logging**: Structured traces with performance metrics

### Decision Flow

```
User Query → Evidence Generation (LLM) → Multi-Alternative DDM → Action Selection → Execution → Reflection
     ↓              ↓                           ↓                      ↓              ↓
  Parallel     JSON Schema              Racing Accumulators      Tool Calls     Outcome Logging
```

---

## Installation

```bash
pip install requests numpy matplotlib pydantic retry
export OPENROUTER_API_KEY="your_key_here"
```

**Dependencies**:
- `requests`: HTTP client
- `numpy`: Numerical computations
- `matplotlib`: Visualizations
- `pydantic`: Data validation
- `retry`: Resilient API calls

---

## Core Implementation

### 1. Data Models (Pydantic Schemas)

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from enum import Enum

class EvidenceQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ActionCandidate(BaseModel):
    """Structured action with evidence."""
    name: str = Field(..., description="Action description")
    evidence_score: float = Field(..., ge=-1.0, le=1.0, description="Evidence strength [-1, 1]")
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    quality: EvidenceQuality = Field(default=EvidenceQuality.MEDIUM)
    uncertainty: float = Field(default=0.3, ge=0.0, le=1.0)
    
    @validator('evidence_score')
    def validate_score(cls, v):
        if not -1.0 <= v <= 1.0:
            raise ValueError("Evidence score must be in [-1, 1]")
        return v

class PlanningResponse(BaseModel):
    """LLM planning output schema."""
    actions: List[ActionCandidate] = Field(..., min_items=2, max_items=5)
    task_analysis: str = Field(..., description="Brief task decomposition")
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)

class DDMOutcome(BaseModel):
    """DDM simulation result."""
    selected_action: str
    selected_index: int
    reaction_time: float
    confidence: float
    trajectories: Optional[List[Dict[str, Any]]] = None

class AgentResponse(BaseModel):
    """Final agent output."""
    decision: str
    action_taken: str
    reasoning: str
    metrics: Dict[str, float]
    traces: Dict[str, Any]
```

---

### 2. Multi-Alternative DDM Implementation

```python
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import time

@dataclass
class DDMConfig:
    """DDM hyperparameters."""
    base_drift: float = 1.0          # Base drift rate
    threshold: float = 1.0           # Decision boundary
    noise_sigma: float = 1.0         # Diffusion noise
    dt: float = 0.01                 # Time step (seconds)
    non_decision_time: float = 0.2   # Ter (encoding/motor time)
    starting_bias: float = 0.0       # Starting point offset
    max_time: float = 5.0            # Maximum deliberation time
    n_trials: int = 100              # Simulation trials for robustness

class MultiAlternativeDDM:
    """Racing accumulators DDM for multiple options."""
    
    def __init__(self, config: Optional[DDMConfig] = None):
        self.config = config or DDMConfig()
        self._validate_config()
    
    def _validate_config(self):
        """Sanity checks."""
        assert self.config.threshold > 0, "Threshold must be positive"
        assert self.config.dt > 0, "Time step must be positive"
        assert 0 < self.config.noise_sigma <= 2, "Noise should be (0, 2]"
    
    def simulate_decision(
        self, 
        actions: List[ActionCandidate],
        mode: str = "racing"  # "racing" or "single_trial"
    ) -> DDMOutcome:
        """
        Simulate multi-alternative decision.
        
        Args:
            actions: List of action candidates with evidence scores
            mode: "racing" for multiple trials, "single_trial" for fast single run
        
        Returns:
            DDMOutcome with selected action and metrics
        """
        if len(actions) == 0:
            raise ValueError("No actions provided")
        
        if mode == "racing":
            return self._racing_accumulators(actions)
        else:
            return self._single_trial_fast(actions)
    
    def _racing_accumulators(self, actions: List[ActionCandidate]) -> DDMOutcome:
        """
        Run racing DDM: each action has its own accumulator.
        First to threshold wins (averaged over trials).
        """
        n_actions = len(actions)
        wins = np.zeros(n_actions)
        rts = []
        all_trajectories = []
        
        for trial in range(self.config.n_trials):
            # Initialize accumulators at starting point
            accumulators = np.ones(n_actions) * (self.config.starting_bias + 0.5) * self.config.threshold
            
            # Extract drift rates (evidence scores modulate drift)
            drift_rates = np.array([
                self.config.base_drift * action.evidence_score 
                for action in actions
            ])
            
            # Simulate until one hits threshold
            t = 0.0
            trajectory = []
            
            while t < self.config.max_time:
                # Accumulate evidence with noise
                drift = drift_rates * self.config.dt
                noise = self.config.noise_sigma * np.sqrt(self.config.dt) * np.random.randn(n_actions)
                accumulators += drift + noise
                
                # Record trajectory (every 10 steps to save memory)
                if int(t / self.config.dt) % 10 == 0:
                    trajectory.append({
                        "time": t,
                        "accumulators": accumulators.copy()
                    })
                
                # Check for boundary crossing
                winners = np.where(accumulators >= self.config.threshold)[0]
                if len(winners) > 0:
                    winner_idx = winners[0]  # First to cross
                    wins[winner_idx] += 1
                    rts.append(t + self.config.non_decision_time)
                    all_trajectories.append(trajectory)
                    break
                
                t += self.config.dt
            else:
                # Timeout: pick highest accumulator
                winner_idx = np.argmax(accumulators)
                wins[winner_idx] += 1
                rts.append(self.config.max_time + self.config.non_decision_time)
                all_trajectories.append(trajectory)
        
        # Aggregate results
        selected_idx = int(np.argmax(wins))
        mean_rt = np.mean(rts)
        confidence = wins[selected_idx] / self.config.n_trials
        
        return DDMOutcome(
            selected_action=actions[selected_idx].name,
            selected_index=selected_idx,
            reaction_time=mean_rt,
            confidence=confidence,
            trajectories=all_trajectories[:10]  # Sample for visualization
        )
    
    def _single_trial_fast(self, actions: List[ActionCandidate]) -> DDMOutcome:
        """Fast single-trial simulation (for low-latency applications)."""
        n_actions = len(actions)
        accumulators = np.ones(n_actions) * 0.5 * self.config.threshold
        drift_rates = np.array([self.config.base_drift * a.evidence_score for a in actions])
        
        t = 0.0
        trajectory = []
        
        while t < self.config.max_time:
            drift = drift_rates * self.config.dt
            noise = self.config.noise_sigma * np.sqrt(self.config.dt) * np.random.randn(n_actions)
            accumulators += drift + noise
            
            if int(t / self.config.dt) % 10 == 0:
                trajectory.append({"time": t, "accumulators": accumulators.copy()})
            
            if np.max(accumulators) >= self.config.threshold:
                winner_idx = np.argmax(accumulators)
                break
            
            t += self.config.dt
        else:
            winner_idx = np.argmax(accumulators)
        
        return DDMOutcome(
            selected_action=actions[winner_idx].name,
            selected_index=winner_idx,
            reaction_time=t + self.config.non_decision_time,
            confidence=accumulators[winner_idx] / self.config.threshold,
            trajectories=[trajectory]
        )
    
    def visualize_trajectories(self, outcome: DDMOutcome, actions: List[ActionCandidate]) -> str:
        """Generate matplotlib visualization."""
        import matplotlib.pyplot as plt
        
        if not outcome.trajectories:
            return "No trajectories to visualize"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, len(actions)))
        for traj in outcome.trajectories[:10]:  # Max 10 trajectories
            times = [step["time"] for step in traj]
            for i, action in enumerate(actions):
                accs = [step["accumulators"][i] for step in traj]
                ax1.plot(times, accs, color=colors[i], alpha=0.3)
        
        # Add threshold line
        ax1.axhline(self.config.threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Evidence Accumulation', fontsize=12)
        ax1.set_title('DDM Trajectories (Racing Accumulators)', fontsize=14, fontweight='bold')
        ax1.legend([f"{a.name[:20]}" for a in actions] + ['Threshold'], loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot final evidence distribution
        if outcome.trajectories:
            final_accs = outcome.trajectories[-1][-1]["accumulators"]
            bars = ax2.bar(range(len(actions)), final_accs, color=colors)
            bars[outcome.selected_index].set_edgecolor('red')
            bars[outcome.selected_index].set_linewidth(3)
            ax2.axhline(self.config.threshold, color='black', linestyle='--', linewidth=2)
            ax2.set_xlabel('Action Index', fontsize=12)
            ax2.set_ylabel('Final Accumulator Value', fontsize=12)
            ax2.set_title('Final Evidence Distribution', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(actions)))
            ax2.set_xticklabels([f"{i}" for i in range(len(actions))])
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save to file (or return base64 for embedding)
        fig_path = f"/tmp/ddm_viz_{int(time.time())}.png"
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        return f"Visualization saved: {fig_path}\n{self._describe_visualization(outcome, actions)}"
    
    def _describe_visualization(self, outcome: DDMOutcome, actions: List[ActionCandidate]) -> str:
        """Text description of visualization."""
        desc = f"""
**DDM Visualization Summary:**
- Selected Action: {outcome.selected_action} (index {outcome.selected_index})
- Reaction Time: {outcome.reaction_time:.3f}s
- Confidence: {outcome.confidence:.2%}
- Racing Pattern: {len(actions)} parallel accumulators competing
- Winner: Action with highest evidence score ({actions[outcome.selected_index].evidence_score:.2f}) typically won fastest
"""
        return desc
```

---

### 3. Robust LLM Client with Parallel Calls

```python
import requests
import asyncio
import aiohttp
from retry import retry
from typing import List, Dict, Any, Optional
import json
import time

class OpenRouterClient:
    """Production-grade OpenRouter client with retries and parallel support."""
    
    def __init__(self, api_key: str, model: str = "x-ai/grok-4-fast"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Title": "ADDM-Agent-v2"
        }
        self.timeout = 30
        self.max_retries = 3
    
    @retry(tries=3, delay=1, backoff=2)
    def complete(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        response_format: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Synchronous completion with retries.
        
        Args:
            messages: Chat messages
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Max response tokens
            response_format: {"type": "json_object"} for JSON mode
        
        Returns:
            API response dict
        """
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {self.timeout}s")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                raise Exception("Rate limit exceeded - backoff triggered")
            elif response.status_code >= 500:
                raise Exception(f"Server error: {response.status_code}")
            else:
                raise Exception(f"HTTP error: {e}")
        except Exception as e:
            raise Exception(f"Unexpected error: {str(e)}")
    
    async def complete_async(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async completion for parallel calls."""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1500)
        }
        
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]
        
        async with aiohttp.ClientSession() as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(
                        self.base_url,
                        headers=self.headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response.raise_for_status()
                        return await response.json()
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def parallel_complete(
        self,
        prompt_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Execute multiple completions in parallel.
        
        Args:
            prompt_list: List of dicts with keys: messages, system_prompt, kwargs
        
        Returns:
            List of responses in same order
        """
        tasks = [
            self.complete_async(
                messages=prompt["messages"],
                system_prompt=prompt.get("system_prompt"),
                **prompt.get("kwargs", {})
            )
            for prompt in prompt_list
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def parse_json_response(self, response: Dict[str, Any], schema: type[BaseModel]) -> BaseModel:
        """
        Parse and validate JSON response.
        
        Args:
            response: API response dict
            schema: Pydantic model class
        
        Returns:
            Validated Pydantic instance
        """
        try:
            content = response['choices'][0]['message']['content']
            # Handle markdown JSON blocks
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            return schema(**data)
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in LLM response: {e}\nContent: {content[:200]}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")
```

---

### 4. ADDM Agent with A/B Testing

```python
from typing import Optional, Literal
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ADDM_Agent:
    """Production agentic framework with DDM decision-making."""
    
    def __init__(
        self,
        api_key: str,
        ddm_config: Optional[DDMConfig] = None,
        enable_traces: bool = True
    ):
        self.llm = OpenRouterClient(api_key)
        self.ddm = MultiAlternativeDDM(ddm_config)
        self.enable_traces = enable_traces
        self.trace_log = []
        self.metrics = {
            "decisions_made": 0,
            "total_rt": 0.0,
            "api_calls": 0,
            "errors": 0
        }
    
    def reset_metrics(self):
        """Reset performance counters."""
        self.metrics = {
            "decisions_made": 0,
            "total_rt": 0.0,
            "api_calls": 0,
            "errors": 0
        }
    
    def log_trace(self, step: str, data: Dict[str, Any]):
        """Log trace entry."""
        if self.enable_traces:
            self.trace_log.append({
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
    
    def decide_and_act(
        self,
        user_input: str,
        task_type: str = "general",
        mode: Literal["ddm", "argmax", "ab_test"] = "ddm",
        use_parallel: bool = False
    ) -> AgentResponse:
        """
        Main agentic loop with optional A/B testing.
        
        Args:
            user_input: User query
            task_type: Task category for prompt tuning
            mode: Decision mode (ddm, argmax, ab_test)
            use_parallel: Use parallel API calls for speed
        
        Returns:
            AgentResponse with decision and traces
        """
        start_time = time.time()
        self.trace_log = []  # Reset traces
        
        try:
            # Step 1: Generate action candidates with structured output
            actions = self._generate_actions(user_input, task_type, use_parallel)
            
            # Step 2: Make decision (DDM or argmax)
            if mode == "ddm":
                outcome = self._decide_ddm(actions)
            elif mode == "argmax":
                outcome = self._decide_argmax(actions)
            else:  # ab_test
                outcome = self._ab_test(actions)
            
            # Step 3: Execute action (simulate or call tools)
            execution_result = self._execute_action(outcome.selected_action, user_input)
            
            # Step 4: Generate final response
            response = self._generate_response(outcome, execution_result, actions)
            
            # Update metrics
            elapsed = time.time() - start_time
            self.metrics["decisions_made"] += 1
            self.metrics["total_rt"] += outcome.reaction_time
            
            return AgentResponse(
                decision=outcome.selected_action,
                action_taken=execution_result,
                reasoning=f"Selected action based on {mode} with RT={outcome.reaction_time:.3f}s",
                metrics={
                    "reaction_time": outcome.reaction_time,
                    "confidence": outcome.confidence,
                    "wall_time": elapsed,
                    "api_calls": self.metrics["api_calls"]
                },
                traces={step["step"]: step["data"] for step in self.trace_log}
            )
        
        except Exception as e:
            logger.error(f"Agent error: {e}")
            self.metrics["errors"] += 1
            raise
    
    def _generate_actions(
        self,
        user_input: str,
        task_type: str,
        use_parallel: bool
    ) -> List[ActionCandidate]:
        """Generate action candidates with structured LLM output."""
        
        system_prompt = f"""You are an evidence-based planning agent using Drift-Diffusion Model principles.

Your task: Generate 2-4 high-quality action candidates for the given query.

For each action, provide:
1. Clear, concrete action description
2. Evidence score (-1.0 to 1.0): How strongly evidence supports this action
   - 1.0: Overwhelming positive evidence
   - 0.0: Neutral/mixed evidence  
   - -1.0: Strong counter-evidence
3. Pros: Concrete supporting facts
4. Cons: Concrete limitations or risks
5. Quality: HIGH/MEDIUM/LOW based on evidence strength
6. Uncertainty: 0.0-1.0 (how uncertain you are about this action)

Task type: {task_type}

**Output format: Valid JSON only**
{{
  "actions": [
    {{
      "name": "Action description",
      "evidence_score": 0.8,
      "pros": ["Pro 1", "Pro 2"],
      "cons": ["Con 1"],
      "quality": "high",
      "uncertainty": 0.2
    }},
    ...
  ],
  "task_analysis": "Brief analysis of the task",
  "confidence": 0.85
}}

**Critical**: Output ONLY valid JSON. No markdown, no extra text."""

        messages = [{"role": "user", "content": f"Query: {user_input}"}]
        
        try:
            response = self.llm.complete(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=1500,
                response_format={"type": "json_object"}  # JSON mode
            )
            
            self.metrics["api_calls"] += 1
            
            planning = self.llm.parse_json_response(response, PlanningResponse)
            
            self.log_trace("LLM Planning", {
                "input": user_input,
                "num_actions": len(planning.actions),
                "actions": [a.dict() for a in planning.actions],
                "task_analysis": planning.task_analysis
            })
            
            return planning.actions
        
        except Exception as e:
            logger.error(f"Action generation failed: {e}")
            # Fallback: create default actions
            return self._fallback_actions(user_input)
    
    def _fallback_actions(self, user_input: str) -> List[ActionCandidate]:
        """Fallback when LLM fails."""
        return [
            ActionCandidate(
                name=f"Approach A for: {user_input[:30]}",
                evidence_score=0.5,
                pros=["Safe default"],
                cons=["Generic"],
                quality=EvidenceQuality.MEDIUM,
                uncertainty=0.5
            ),
            ActionCandidate(
                name=f"Approach B for: {user_input[:30]}",
                evidence_score=0.3,
                pros=["Alternative option"],
                cons=["Less evidence"],
                quality=EvidenceQuality.LOW,
                uncertainty=0.6
            )
        ]
    
    def _decide_ddm(self, actions: List[ActionCandidate]) -> DDMOutcome:
        """DDM-based decision."""
        outcome = self.ddm.simulate_decision(actions, mode="racing")
        
        self.log_trace("DDM Decision", {
            "mode": "racing",
            "selected_action": outcome.selected_action,
            "selected_index": outcome.selected_index,
            "reaction_time": outcome.reaction_time,
            "confidence": outcome.confidence,
            "config": self.ddm.config.__dict__
        })
        
        return outcome
    
    def _decide_argmax(self, actions: List[ActionCandidate]) -> DDMOutcome:
        """Baseline: simple argmax on evidence scores."""
        scores = [a.evidence_score for a in actions]
        selected_idx = int(np.argmax(scores))
        
        outcome = DDMOutcome(
            selected_action=actions[selected_idx].name,
            selected_index=selected_idx,
            reaction_time=0.0,  # Instant
            confidence=abs(actions[selected_idx].evidence_score)
        )
        
        self.log_trace("Argmax Decision", {
            "mode": "argmax",
            "selected_action": outcome.selected_action,
            "scores": scores
        })
        
        return outcome
    
    def _ab_test(self, actions: List[ActionCandidate]) -> DDMOutcome:
        """A/B test: Run both DDM and argmax, compare."""
        ddm_outcome = self._decide_ddm(actions)
        argmax_outcome = self._decide_argmax(actions)
        
        self.log_trace("A/B Test", {
            "ddm_choice": ddm_outcome.selected_action,
            "argmax_choice": argmax_outcome.selected_action,
            "agreement": ddm_outcome.selected_index == argmax_outcome.selected_index,
            "ddm_confidence": ddm_outcome.confidence,
            "ddm_rt": ddm_outcome.reaction_time
        })
        
        # For A/B test, return DDM outcome (or choose based on confidence)
        return ddm_outcome
    
    def _execute_action(self, action: str, context: str) -> str:
        """Execute selected action (simulated or real tool calls)."""
        # Placeholder: In production, call actual tools/APIs here
        execution_prompt = f"""Execute this action: {action}

Context: {context}

Provide a brief, concrete result of executing this action."""

        try:
            response = self.llm.complete(
                messages=[{"role": "user", "content": execution_prompt}],
                temperature=0.5,
                max_tokens=500
            )
            
            self.metrics["api_calls"] += 1
            
            result = response['choices'][0]['message']['content']
            
            self.log_trace("Action Execution", {
                "action": action,
                "result": result[:200]  # Truncate for logging
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return f"Simulated execution of: {action}"
    
    def _generate_response(
        self,
        outcome: DDMOutcome,
        execution_result: str,
        actions: List[ActionCandidate]
    ) -> str:
        """Format final user-facing response."""
        response = f"""**Decision Made**: {outcome.selected_action}

**Result**: {execution_result}

**Decision Metrics**:
- Reaction Time: {outcome.reaction_time:.3f}s (DDM simulation)
- Confidence: {outcome.confidence:.1%}
- Alternatives Considered: {len(actions)}

**Visualization**: {self.ddm.visualize_trajectories(outcome, actions)}
"""
        return response
```

---

## Usage Examples

### Basic Usage

```python
import os

# Initialize agent
api_key = os.getenv("OPENROUTER_API_KEY")
agent = ADDM_Agent(api_key)

# Make a decision
response = agent.decide_and_act(
    user_input="Recommend a healthy breakfast for an athlete",
    task_type="nutrition planning",
    mode="ddm"
)

print(response.decision)
print(f"Confidence: {response.metrics['confidence']:.2%}")
print(f"RT: {response.metrics['reaction_time']:.3f}s")

# View traces
for step, data in response.traces.items():
    print(f"\n--- {step} ---")
    print(data)
```

### A/B Testing DDM vs Argmax

```python
# Test on multiple queries
test_queries = [
    "Choose a programming language for web development",
    "Plan a weekend trip to Paris",
    "Decide on a machine learning model for image classification"
]

results = {"ddm_wins": 0, "argmax_wins": 0, "agreement": 0}

for query in test_queries:
    response = agent.decide_and_act(
        user_input=query,
        mode="ab_test"
    )
    
    # Extract A/B comparison from traces
    ab_data = response.traces.get("A/B Test", {})
    if ab_data.get("agreement"):
        results["agreement"] += 1
    
    # Custom metric: Higher confidence = "better"
    # (In production, use outcome quality metrics)

print(f"Agreement rate: {results['agreement']/len(test_queries):.1%}")
```

### Parallel Evidence Generation (Advanced)

```python
# For complex multi-step tasks, generate evidence in parallel
async def parallel_planning_example():
    agent = ADDM_Agent(api_key)
    
    # Create multiple sub-queries
    sub_queries = [
        "What are good breakfast proteins?",
        "What are good breakfast carbs?",
        "What timing is best for athlete breakfast?"
    ]
    
    # Prepare parallel prompts
    prompts = [
        {
            "messages": [{"role": "user", "content": q}],
            "system_prompt": "Provide evidence in JSON format.",
            "kwargs": {"response_format": {"type": "json_object"}}
        }
        for q in sub_queries
    ]
    
    # Execute in parallel (4 calls under 1 cent with Grok Fast)
    responses = await agent.llm.parallel_complete(prompts)
    
    # Aggregate evidence
    # ... (custom logic to combine into ActionCandidates)

# Run async
# asyncio.run(parallel_planning_example())
```

---

## Testing & Validation

### Unit Tests

```python
import unittest

class TestDDM(unittest.TestCase):
    def test_racing_accumulators(self):
        """Test multi-alternative DDM."""
        ddm = MultiAlternativeDDM()
        actions = [
            ActionCandidate(name="A1", evidence_score=0.8, pros=["fast"], cons=[]),
            ActionCandidate(name="A2", evidence_score=0.2, pros=["cheap"], cons=[])
        ]
        
        outcome = ddm.simulate_decision(actions, mode="racing")
        
        # High evidence action should win more often
        self.assertEqual(outcome.selected_action, "A1")
        self.assertGreater(outcome.confidence, 0.7)
    
    def test_no_drift_mutation(self):
        """Ensure drift rate doesn't accumulate across calls."""
        ddm = MultiAlternativeDDM()
        initial_v = ddm.config.base_drift
        
        actions = [ActionCandidate(name="A", evidence_score=0.5, pros=[], cons=[])]
        ddm.simulate_decision(actions)
        
        # Drift rate should be unchanged
        self.assertEqual(ddm.config.base_drift, initial_v)

class TestLLMClient(unittest.TestCase):
    def test_json_parsing(self):
        """Test structured output parsing."""
        client = OpenRouterClient(api_key="fake_key")
        
        # Mock response
        response = {
            'choices': [{
                'message': {
                    'content': '```json\n{"actions": [], "task_analysis": "test", "confidence": 0.8}\n```'
                }
            }]
        }
        
        result = client.parse_json_response(response, PlanningResponse)
        self.assertIsInstance(result, PlanningResponse)
        self.assertEqual(result.confidence, 0.8)

if __name__ == "__main__":
    unittest.main()
```

### Performance Benchmarks

```python
def benchmark_decision_speed():
    """Compare DDM modes and argmax."""
    agent = ADDM_Agent(api_key)
    
    test_query = "Choose a cloud provider for a startup"
    
    # Test DDM racing
    start = time.time()
    resp_ddm = agent.decide_and_act(test_query, mode="ddm")
    ddm_time = time.time() - start
    
    # Test argmax
    start = time.time()
    resp_argmax = agent.decide_and_act(test_query, mode="argmax")
    argmax_time = time.time() - start
    
    print(f"DDM wall time: {ddm_time:.2f}s (simulated RT: {resp_ddm.metrics['reaction_time']:.3f}s)")
    print(f"Argmax wall time: {argmax_time:.2f}s")
    print(f"Overhead: {(ddm_time - argmax_time):.2f}s")

# Run benchmark
benchmark_decision_speed()
```

---

## Configuration Guide

### DDM Hyperparameter Tuning

```python
# Conservative agent (high accuracy, slow)
conservative_config = DDMConfig(
    base_drift=0.8,
    threshold=1.5,  # Higher threshold = more evidence needed
    noise_sigma=0.8,
    n_trials=200
)

# Aggressive agent (fast, may err)
aggressive_config = DDMConfig(
    base_drift=1.5,
    threshold=0.7,  # Lower threshold = faster decisions
    noise_sigma=1.2,
    n_trials=50
)

# Use in agent
conservative_agent = ADDM_Agent(api_key, ddm_config=conservative_config)
```

### Production Deployment Checklist

- [ ] Set `OPENROUTER_API_KEY` in environment
- [ ] Configure logging (use `logging.config` for structured logs)
- [ ] Enable request tracing for debugging
- [ ] Set up monitoring (track `metrics` dict)
- [ ] Test error handling with invalid inputs
- [ ] Validate JSON schemas with diverse queries
- [ ] Benchmark DDM overhead vs. simple argmax
- [ ] Set rate limits (OpenRouter handles this, but monitor usage)
- [ ] Add caching for repeated queries (optional)
- [ ] Implement tool calling for real actions (not just LLM simulation)

### Cost Optimization

With `x-ai/grok-4-fast`:
- ~$0.0001 per 1K tokens (check OpenRouter for latest)
- Typical decision: 2-3 API calls × ~500 tokens each = ~1.5K tokens
- Cost per decision: **~$0.00015** (<$0.01 for 4 parallel calls)

**Tips**:
- Use `max_tokens` limits to control costs
- Cache action generation for similar queries
- Use `single_trial` mode for DDM (faster, 1 simulation vs 100)

---

## Advanced Extensions

### 1. Multi-Step Planning with Feedback

```python
def multi_step_task(agent: ADDM_Agent, goal: str, max_steps: int = 5):
    """Iterate decision loop until goal achieved."""
    context = goal
    steps_taken = []
    
    for i in range(max_steps):
        response = agent.decide_and_act(
            user_input=f"Step {i+1}: {context}",
            task_type="multi-step planning"
        )
        
        steps_taken.append(response.decision)
        
        # Check if goal reached (via LLM evaluation)
        if "completed" in response.action_taken.lower():
            break
        
        # Update context with outcome
        context = f"Previous: {response.action_taken}. Continue toward: {goal}"
    
    return steps_taken
```

### 2. Tool Integration

```python
class ADDM_AgentWithTools(ADDM_Agent):
    """Agent with real tool calling."""
    
    def __init__(self, api_key: str, tools: Dict[str, callable]):
        super().__init__(api_key)
        self.tools = tools  # e.g., {"search": web_search_fn, "calculator": calc_fn}
    
    def _execute_action(self, action: str, context: str) -> str:
        """Execute via actual tools."""
        # Parse action to determine tool
        for tool_name, tool_fn in self.tools.items():
            if tool_name in action.lower():
                try:
                    result = tool_fn(action)
                    return f"Tool '{tool_name}' result: {result}"
                except Exception as e:
                    return f"Tool error: {e}"
        
        # Fallback to LLM simulation
        return super()._execute_action(action, context)
```

### 3. Adaptive DDM Parameters

```python
class AdaptiveDDM(MultiAlternativeDDM):
    """DDM that adjusts parameters based on task difficulty."""
    
    def simulate_decision(self, actions: List[ActionCandidate], **kwargs):
        # Adjust threshold based on uncertainty
        avg_uncertainty = np.mean([a.uncertainty for a in actions])
        self.config.threshold = 1.0 + avg_uncertainty  # Higher uncertainty = higher threshold
        
        # Adjust drift based on quality
        high_quality = sum(1 for a in actions if a.quality == EvidenceQuality.HIGH)
        self.config.base_drift = 1.0 if high_quality > 0 else 0.7
        
        return super().simulate_decision(actions, **kwargs)
```

---

## Troubleshooting

### Common Issues

1. **"Invalid JSON in LLM response"**
   - Solution: Check that `response_format={"type": "json_object"}` is set
   - Fallback: Manual parsing with regex for JSON blocks

2. **"DDM trials timeout"**
   - Solution: Reduce `n_trials` or increase `max_time`
   - Check: Are evidence scores extreme (e.g., all near 0)?

3. **"Rate limit exceeded"**
   - Solution: OpenRouter has generous limits; check your account
   - Add exponential backoff (already in `complete_async`)

4. **"All actions have similar evidence"**
   - Expected: DDM will take longer (more trials to boundary)
   - Solution: Tune prompts to generate more discriminative evidence

---

## Summary of Fixes

| Issue | Original | Fixed |
|-------|----------|-------|
| Evidence parsing | String splitting | JSON schema validation |
| Drift mutation | `self.v +=` | Local variable |
| Multi-choice DDM | Binary only | Racing accumulators |
| Error handling | `raise_for_status()` | Retries + fallbacks |
| Validation | None | Pydantic + unit tests |
| Parallel calls | Sequential | `asyncio.gather` |
| A/B testing | Not implemented | Built-in mode |

---

## License & Citation

This framework is provided as-is for research and commercial use.

**Citation** (if used in research):
```
@software{addm_framework_2024,
  title={Agentic Drift-Diffusion Model Framework},
  author={Production Implementation},
  year={2024},
  url={https://github.com/your-repo/addm-framework}
}
```

**References**:
- Ratcliff & McKoon (2008). "The Diffusion Decision Model"
- Krajbich et al. (2010). "Visual fixations and the computation of value in simple choice"
- OpenRouter API: https://openrouter.ai/docs

---

## Next Steps

1. **Test on real tasks**: Run on your domain (finance, planning, coding)
2. **Tune hyperparameters**: Adjust DDM config for your use case
3. **Add tools**: Integrate with APIs, databases, search engines
4. **Deploy**: Wrap in FastAPI/Flask for production serving
5. **Monitor**: Track decision quality metrics (accuracy, user satisfaction)

**Questions?** Check the troubleshooting section or open an issue!
