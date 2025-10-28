# ADDM Framework - Phase 0: Project Overview

## Project Summary

The Agentic Drift-Diffusion Model (ADDM) Framework is a production-grade cognitive decision-making system that combines Drift-Diffusion Model (DDM) principles from neuroscience with Large Language Model (LLM) agents. The framework enables autonomous agents to make evidence-based decisions by simulating cognitive processes of evidence accumulation, competing options through racing accumulators, and transparent decision tracing.

**Core Innovation**: Unlike simple argmax selection (instant, noise-prone), DDM accumulates evidence over time, modeling realistic decision-making with reaction times, confidence scores, and improved accuracy under uncertainty.

**Target Applications**:
- Autonomous planning systems
- Decision support tools
- Multi-agent coordination
- Risk assessment platforms
- Strategic recommendation engines

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Input                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ADDM Agent                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Evidence Generation (LLM Client)                      │  │
│  │     - Parallel API calls to OpenRouter/Grok               │  │
│  │     - Structured JSON output (Pydantic validation)        │  │
│  │     - Action candidates with evidence scores              │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. Multi-Alternative DDM Decision                        │  │
│  │     - Racing accumulators (one per action)                │  │
│  │     - Evidence-driven drift rates                         │  │
│  │     - Stochastic simulation (noise modeling)              │  │
│  │     - Boundary threshold detection                        │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. Action Execution                                      │  │
│  │     - Tool calling (extensible)                           │  │
│  │     - LLM-based simulation                                │  │
│  │     - Result validation                                   │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                        │
│                         ▼                                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  4. Trace & Response Generation                           │  │
│  │     - Structured logging                                  │  │
│  │     - Visualization generation                            │  │
│  │     - Metrics tracking                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Response + Traces + Metrics                     │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Data Models Layer** (Pydantic)
   - Structured schemas for actions, evidence, decisions
   - Runtime validation
   - Type safety guarantees

2. **DDM Engine**
   - Racing accumulators for multi-alternative decisions
   - Configurable hyperparameters (drift, noise, threshold)
   - Trajectory tracking and visualization

3. **LLM Client**
   - OpenRouter API integration (Grok 4 Fast)
   - Parallel/async completion support
   - Retry logic with exponential backoff
   - JSON response parsing

4. **Agent Orchestrator**
   - Decision loop coordination
   - A/B testing (DDM vs argmax)
   - Trace logging system
   - Metrics aggregation

5. **Testing & Validation**
   - Unit tests for each component
   - Integration tests for full pipeline
   - Performance benchmarks
   - A/B comparison framework

## Phase Breakdown

### Phase 1: Foundation & Environment Setup
**Goal:** Establish project structure, dependencies, and development environment  
**Duration:** 2-3 hours  
**Dependencies:** None  
**Deliverables:**
- Python 3.10+ virtual environment
- All dependencies installed (`requests`, `numpy`, `matplotlib`, `pydantic`, `retry`, `aiohttp`)
- Project directory structure
- Configuration management system
- Environment variable handling (API keys)
- Basic logging setup
- Initial documentation structure

**Success Criteria:**
- [ ] Virtual environment activated
- [ ] All dependencies installed without conflicts
- [ ] API key loaded from environment
- [ ] Logging outputs to console and file
- [ ] Directory structure matches convention

---

### Phase 2: Data Models & Schemas
**Goal:** Implement Pydantic data models for type-safe, validated data flow  
**Duration:** 3-4 hours  
**Dependencies:** Phase 1  
**Deliverables:**
- `EvidenceQuality` enum
- `ActionCandidate` model with validators
- `PlanningResponse` model
- `DDMOutcome` model
- `AgentResponse` model
- `DDMConfig` dataclass
- Unit tests for all models
- Validation edge case handling

**Success Criteria:**
- [ ] All models define complete schemas
- [ ] Custom validators enforce constraints (-1 to 1 scores, etc.)
- [ ] JSON serialization/deserialization works
- [ ] Invalid data raises clear validation errors
- [ ] Unit tests achieve 95%+ coverage

---

### Phase 3: DDM Core Engine
**Goal:** Implement multi-alternative DDM with racing accumulators  
**Duration:** 6-8 hours  
**Dependencies:** Phase 2  
**Deliverables:**
- `DDMConfig` configuration management
- `MultiAlternativeDDM` class
- Racing accumulators algorithm
- Single-trial fast mode
- Trajectory recording
- Visualization generation (matplotlib)
- Unit tests for simulation logic
- Performance benchmarks

**Success Criteria:**
- [ ] Racing DDM selects action based on evidence scores
- [ ] Higher evidence actions win more frequently
- [ ] Reaction times correlate with evidence quality
- [ ] Trajectories plotted correctly
- [ ] No state mutation bugs (drift rate stable)
- [ ] Simulation runs in <1 second for 100 trials
- [ ] Visualizations render without errors

---

### Phase 4: LLM Client Layer
**Goal:** Build robust OpenRouter API client with error handling and parallel support  
**Duration:** 5-6 hours  
**Dependencies:** Phase 1  
**Deliverables:**
- `OpenRouterClient` class
- Synchronous completion with retries
- Async completion for parallel calls
- `parallel_complete` method
- JSON response parsing (with markdown stripping)
- Pydantic integration for response validation
- Timeout handling
- Rate limit detection
- Exponential backoff logic
- Unit tests with mocked API responses

**Success Criteria:**
- [ ] Successful API call returns valid response
- [ ] Failed calls retry 3 times with backoff
- [ ] Timeouts handled gracefully (30s limit)
- [ ] JSON parsing handles markdown code blocks
- [ ] Parallel calls complete in expected time
- [ ] Rate limit errors logged clearly
- [ ] Mock tests verify retry logic

---

### Phase 5: Agent Integration & Decision Loop
**Goal:** Orchestrate the complete agent pipeline from input to output  
**Duration:** 8-10 hours  
**Dependencies:** Phases 2, 3, 4  
**Deliverables:**
- `ADDM_Agent` class
- `decide_and_act` main loop
- Evidence generation pipeline
- DDM decision integration
- Argmax baseline implementation
- A/B testing mode
- Action execution (simulation + extensibility)
- Response generation
- Trace logging system
- Metrics tracking
- Integration tests

**Success Criteria:**
- [ ] Full pipeline runs end-to-end without errors
- [ ] Traces capture all steps (LLM, DDM, execution)
- [ ] Metrics accurately track RT, API calls, errors
- [ ] A/B testing compares DDM and argmax
- [ ] Response includes decision, reasoning, metrics
- [ ] Fallback actions generated when LLM fails
- [ ] Integration tests cover happy path and error cases

---

### Phase 6: Testing & Validation Framework
**Goal:** Comprehensive testing suite for reliability and correctness  
**Duration:** 4-5 hours  
**Dependencies:** Phases 2-5  
**Deliverables:**
- Unit tests for all classes
- Integration tests for agent pipeline
- Performance benchmarks
- A/B testing comparison suite
- Mock fixtures for API responses
- Test coverage reporting
- Continuous testing setup (pytest configuration)
- Edge case testing (empty actions, invalid JSON, timeouts)

**Success Criteria:**
- [ ] Unit test coverage >90%
- [ ] Integration tests pass consistently
- [ ] Performance benchmarks establish baseline
- [ ] A/B tests show DDM vs argmax differences
- [ ] All edge cases handled gracefully
- [ ] Tests run in <30 seconds total
- [ ] Coverage report generated

---

### Phase 7: Visualization & Trace System
**Goal:** Rich feedback mechanisms for interpretability and debugging  
**Duration:** 3-4 hours  
**Dependencies:** Phase 3, Phase 5  
**Deliverables:**
- DDM trajectory plotting (matplotlib)
- Final evidence distribution charts
- Trace rendering system (JSON/Markdown)
- Visualization description generation
- Image export to files
- Optional base64 embedding for web UIs
- Collapsible trace sections (future: HTML rendering)

**Success Criteria:**
- [ ] Trajectories show racing paths to thresholds
- [ ] Winner highlighted in final distribution chart
- [ ] Traces formatted as readable JSON
- [ ] Images saved to `/tmp` or outputs directory
- [ ] Descriptions provide actionable insights
- [ ] No memory leaks from matplotlib figures

---

### Phase 8: Advanced Features
**Goal:** Extensions for production use cases  
**Duration:** 6-8 hours  
**Dependencies:** Phases 5, 6  
**Deliverables:**
- Multi-step planning with feedback loops
- Tool integration pattern (extensible `_execute_action`)
- Adaptive DDM (parameters adjust to task uncertainty)
- Parallel evidence generation (async across sub-queries)
- Caching layer for repeated queries
- Rate limiting and cost tracking
- Advanced configuration options

**Success Criteria:**
- [ ] Multi-step tasks iterate toward goal completion
- [ ] Tools (search, calculator, etc.) callable from agent
- [ ] Adaptive DDM adjusts thresholds based on uncertainty
- [ ] Parallel evidence reduces latency by 2-3x
- [ ] Caching avoids redundant API calls
- [ ] Cost per decision tracked accurately

---

### Phase 9: Production Deployment
**Goal:** Deployment-ready packaging and infrastructure  
**Duration:** 4-6 hours  
**Dependencies:** All previous phases  
**Deliverables:**
- Dockerfile for containerization
- Docker Compose for local orchestration
- Environment variable management
- Production logging configuration (structured JSON)
- Monitoring hooks (Prometheus metrics)
- API endpoint wrapper (FastAPI/Flask)
- Deployment documentation
- Health check endpoint
- Graceful shutdown handling

**Success Criteria:**
- [ ] Docker image builds successfully
- [ ] Container runs agent pipeline
- [ ] Environment variables loaded from .env
- [ ] Logs output structured JSON
- [ ] Metrics exposed for monitoring
- [ ] API endpoint responds to POST requests
- [ ] Health check returns 200 OK
- [ ] Documentation covers deployment steps

---

### Phase 10: Documentation & Examples
**Goal:** Comprehensive guides for users and developers  
**Duration:** 3-4 hours  
**Dependencies:** Phases 1-9  
**Deliverables:**
- README.md with quick start
- Usage examples (basic, A/B testing, parallel)
- Configuration guide
- Troubleshooting section
- API reference
- Architecture diagrams
- Performance tuning guide
- Contributing guidelines

**Success Criteria:**
- [ ] New users can run example in <5 minutes
- [ ] Configuration options documented
- [ ] Troubleshooting covers common errors
- [ ] Examples demonstrate all features
- [ ] Diagrams clarify architecture
- [ ] Performance tips included

---

## Technology Stack

### Core Dependencies
- **Python**: 3.10+ (type hints, dataclasses)
- **Pydantic**: 2.x (data validation)
- **NumPy**: 1.24+ (numerical computation)
- **Matplotlib**: 3.7+ (visualization)
- **Requests**: 2.31+ (HTTP client)
- **Retry**: 0.9+ (resilient API calls)
- **aiohttp**: 3.9+ (async HTTP)

### Development Tools
- **pytest**: Testing framework
- **pytest-cov**: Coverage reporting
- **black**: Code formatting
- **mypy**: Type checking
- **pylint**: Linting

### Deployment
- **Docker**: Containerization
- **FastAPI** (optional): REST API wrapper
- **Prometheus** (optional): Metrics monitoring

### External Services
- **OpenRouter API**: LLM completions
- **Grok 4 Fast**: Cost-efficient reasoning model

---

## Success Criteria

### Functional Requirements
- [ ] Agent completes full decision loop (input → output)
- [ ] DDM selects actions based on evidence accumulation
- [ ] Argmax baseline provides comparison
- [ ] A/B testing demonstrates DDM advantages
- [ ] Parallel API calls reduce latency
- [ ] Error handling prevents crashes
- [ ] Traces provide full pipeline visibility

### Performance Requirements
- [ ] Decision latency: <5 seconds (including API calls)
- [ ] DDM simulation: <1 second (100 trials)
- [ ] API cost: <$0.001 per decision (Grok 4 Fast)
- [ ] Parallel calls: 4 concurrent under 1 cent
- [ ] Memory usage: <500MB per agent instance

### Quality Requirements
- [ ] Test coverage: >85%
- [ ] No critical bugs in production
- [ ] Logging captures all errors
- [ ] Documentation covers all features
- [ ] Code follows style guide (black, pylint)

### Cognitive/Scientific Validation
- [ ] DDM outperforms argmax in noisy conditions (>10% accuracy)
- [ ] Reaction times correlate with evidence quality
- [ ] Confidence scores calibrated to decision quality
- [ ] Trajectories visualize evidence accumulation

---

## Team Structure

**Recommended Composition:**
- **1 Senior Developer**: Overall architecture, DDM algorithm, integration
- **1 Mid-Level Developer**: LLM client, agent loop, testing
- **1 Junior Developer**: Data models, utilities, documentation

**Estimated Total Effort:** 40-50 hours (1-2 weeks for small team)

**Skills Required:**
- Python proficiency (type hints, async/await)
- Understanding of cognitive models (DDM basics)
- LLM API integration experience
- Testing and debugging skills
- Docker/deployment knowledge (for Phase 9)

---

## Dependencies Between Phases

```
Phase 1 (Foundation)
    ├─→ Phase 2 (Data Models)
    ├─→ Phase 4 (LLM Client)
    └─→ Phase 9 (Deployment)

Phase 2 (Data Models)
    └─→ Phase 3 (DDM Engine)
    └─→ Phase 5 (Agent)

Phase 3 (DDM Engine)
    └─→ Phase 5 (Agent)
    └─→ Phase 7 (Visualization)

Phase 4 (LLM Client)
    └─→ Phase 5 (Agent)

Phase 5 (Agent)
    ├─→ Phase 6 (Testing)
    ├─→ Phase 7 (Visualization)
    └─→ Phase 8 (Advanced Features)

Phase 6 (Testing)
    └─→ Phase 9 (Deployment)

Phase 7 (Visualization)
    └─→ Phase 10 (Documentation)

Phase 8 (Advanced Features)
    └─→ Phase 9 (Deployment)

Phase 9 (Deployment)
    └─→ Phase 10 (Documentation)
```

**Critical Path:** 1 → 2 → 3 → 5 → 6 → 9 → 10 (minimum viable product)  
**Optional Path:** 7, 8 (can be added post-MVP)

---

## Risk Assessment

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| API rate limits exceeded | High | Low | Implement exponential backoff, monitor usage |
| DDM simulation performance | Medium | Medium | Use single-trial mode for low-latency, optimize numpy ops |
| JSON parsing failures | Medium | Medium | Fallback to string parsing, validate schemas rigorously |
| State mutation bugs | High | Low | Immutable configs, thorough unit tests |
| Memory leaks in visualizations | Low | Medium | Close matplotlib figures explicitly, limit trajectory storage |

### Process Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Scope creep (too many features) | Medium | Medium | Stick to phase plan, defer non-critical features |
| Inadequate testing | High | Low | Enforce 85%+ coverage, review tests in code reviews |
| Poor documentation | Medium | Medium | Write docs as code is developed, not at the end |
| Dependency conflicts | Low | Low | Pin versions in requirements.txt, test in clean env |

---

## Getting Started

### Prerequisites
- Python 3.10+ installed
- OpenRouter API key (free tier available)
- Git for version control
- 4GB+ RAM recommended

### Quick Start
1. **Clone or create project directory**
2. **Set up environment**: `python -m venv venv && source venv/bin/activate`
3. **Install dependencies**: `pip install -r requirements.txt`
4. **Set API key**: `export OPENROUTER_API_KEY=your_key_here`
5. **Proceed to Phase 1 documentation**

---

## Next Steps

After reviewing this Phase 0 overview:

1. **Validate Architecture**: Ensure the approach meets your requirements
2. **Adjust Phase Scope**: Modify phases based on team size/timeline
3. **Begin Phase 1**: Start with foundation setup
4. **Create Substages**: If any phase is too complex, unfold it into substages
5. **Track Progress**: Check off deliverables as completed

**To proceed**: Request the detailed Phase 1 document to begin implementation.

---

## Appendix: Key Decisions & Rationale

### Why Multi-Alternative DDM?
Standard DDM is binary (2 choices). Most real decisions involve 3+ options. Racing accumulators generalize DDM to multiple alternatives, maintaining cognitive realism.

### Why Pydantic Over Dict?
Type safety catches errors at validation time, not runtime. Schemas serve as self-documentation. Serialization is built-in.

### Why Grok 4 Fast?
- Cost-efficient ($0.0001-$0.001 per 1K tokens)
- Fast inference (low latency)
- 2M token context window
- Supports JSON mode

### Why Async/Parallel?
Evidence generation for complex decisions can be parallelized (e.g., 4 sub-queries). Async reduces latency by 2-3x.

### Why A/B Testing Built-In?
Validates that DDM provides value over simple argmax. Essential for scientific rigor and production justification.

### Why Traces?
Transparency builds trust. Debugging complex agent behavior requires visibility into each step. Traces enable post-hoc analysis.

---

## References

- **DDM Theory**: Ratcliff & McKoon (2008), "The Diffusion Decision Model: Theory and Data"
- **Multi-Alternative DDM**: Krajbich et al. (2010), "Visual fixations and the computation of value in simple choice"
- **OpenRouter Docs**: https://openrouter.ai/docs
- **Pydantic Docs**: https://docs.pydantic.dev

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-27  
**Character Count**: ~14,200 (well under 45,000 limit)

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2025-10-27 | 1.0 | Initial Phase 0 overview created |

