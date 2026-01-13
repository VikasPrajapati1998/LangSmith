# LangSmith: Comprehensive Developer Guide

## Table of Contents
1. [Introduction](#introduction)
2. [What is LangSmith?](#what-is-langsmith)
3. [Why Developers Need LangSmith](#why-developers-need-langsmith)
4. [Core Features](#core-features)
5. [How LangSmith Works](#how-langsmith-works)
6. [Integration with LLM Applications](#integration-with-llm-applications)
7. [Working with RAG Systems](#working-with-rag-systems)
8. [Agent Development & Debugging](#agent-development--debugging)
9. [LangChain & LangGraph Integration](#langchain--langgraph-integration)
10. [Getting Started](#getting-started)
11. [Best Practices](#best-practices)
12. [Production Deployment](#production-deployment)
13. [Pricing & Plans](#pricing--plans)
14. [Conclusion](#conclusion)

---

## Introduction

Building reliable Large Language Model (LLM) applications is fundamentally different from traditional software development. LLMs are non-deterministic, meaning they can produce different outputs for the same input. This unpredictability, combined with complex multi-step workflows involving retrieval systems, agents, and tool usage, creates unique challenges in debugging, testing, and monitoring.

**LangSmith** is a comprehensive platform specifically designed to address these challenges. Created by the team behind LangChain, it provides end-to-end observability, evaluation, and monitoring capabilities for LLM applications, whether you're building simple chatbots or complex agentic systems.

---

## What is LangSmith?

LangSmith is a production-grade platform for building, debugging, testing, evaluating, and monitoring LLM applications. It serves as the operational backbone for AI systems, providing deep visibility into every aspect of your application's behavior.

### Key Capabilities

- **Tracing & Observability**: Complete visibility into every step of your LLM application's execution
- **Debugging Tools**: Identify and fix issues in complex chains and agents
- **Evaluation Framework**: Systematic testing and benchmarking of LLM outputs
- **Production Monitoring**: Real-time tracking of application performance and quality
- **Dataset Management**: Create, store, and version test datasets for continuous evaluation
- **Prompt Engineering**: Test, version, and optimize prompts with the Playground
- **Feedback Collection**: Gather and analyze user feedback for continuous improvement

---

## Why Developers Need LangSmith

### The Challenge with LLM Applications

Traditional debugging and monitoring tools fall short when dealing with LLM applications because:

1. **Non-Determinism**: Same input can produce different outputs, making bugs hard to reproduce
2. **Complex Workflows**: Multi-step chains involving retrieval, reasoning, and generation
3. **Hidden Failures**: Outputs may be syntactically correct but semantically wrong
4. **Token Costs**: Need to track and optimize expensive API calls
5. **Latency Issues**: Identifying bottlenecks in multi-step processes
6. **Quality Assessment**: Evaluating natural language outputs isn't straightforward

### How LangSmith Solves These Problems

LangSmith provides LLM-native solutions:

- **Hierarchical Tracing**: See every LLM call, tool invocation, and retrieval step nested logically
- **Input/Output Inspection**: Examine exact prompts, context, and responses at each step
- **Performance Metrics**: Track token usage, latency, error rates, and costs
- **Automated Evaluation**: Run systematic tests against reference datasets
- **Version Comparison**: A/B test different prompts, models, or entire workflows
- **Root Cause Analysis**: Quickly identify which component caused an issue

---

## Core Features

### 1. Tracing

Tracing is the foundation of LangSmith. Every execution of your application creates a trace - a detailed log of every operation.

**What Gets Traced:**
- LLM API calls (OpenAI, Anthropic, etc.)
- Document retrievals from vector databases
- Tool/function calls
- Prompt construction
- Output parsing
- Chain/agent reasoning steps

**Trace Structure:**
- **Runs**: Individual steps (LLM calls, tool invocations)
- **Traces**: Single execution containing a tree of runs
- **Threads**: Collection of traces representing a conversation

**Benefits:**
- Understand exactly what your application is doing
- Identify where errors occur in complex chains
- Measure latency at each step
- Debug unexpected behaviors
- Optimize token usage

### 2. Evaluation

Systematic testing framework for LLM applications:

**Evaluation Types:**
- **Offline Evaluation**: Test against curated datasets before deployment
- **Online Evaluation**: Monitor production traffic in real-time
- **Human Evaluation**: Annotation queues for expert review
- **Automated Evaluation**: Heuristic, LLM-as-judge, and custom evaluators

**Evaluation Metrics:**
- Correctness/Accuracy
- Relevance
- Faithfulness (no hallucinations)
- Groundedness (based on retrieved context)
- Completeness
- Helpfulness
- Custom domain-specific criteria

### 3. Datasets

Centralized dataset management for testing:

- Create datasets from production traces
- Version control for datasets
- Share datasets across teams
- Synthetic data generation
- Reference answers for comparison
- Continuous dataset enrichment from user feedback

### 4. Playground

Interactive prompt engineering environment:

- Test prompts with different models
- Compare outputs side-by-side
- Adjust parameters (temperature, max tokens, etc.)
- Version and save successful prompts
- Share prompts via LangChain Hub

### 5. Monitoring

Production-ready monitoring dashboard:

- Real-time performance metrics
- Error rate tracking
- Cost analysis
- Usage patterns
- Custom alerts
- Drift detection
- Integration with CloudWatch, Grafana, etc.

---

## How LangSmith Works

### Architecture Overview

LangSmith uses a lightweight SDK that instruments your application code. When enabled, it automatically captures execution data without adding latency to your application.

```
Your Application → LangSmith SDK → Async Trace Collector → LangSmith Platform
                                                          ↓
                                                    Analysis & Visualization
```

### Data Flow

1. **Instrumentation**: Add LangSmith SDK to your application
2. **Execution**: Your application runs normally
3. **Capture**: SDK asynchronously logs execution data
4. **Processing**: Data is processed and structured
5. **Storage**: Traces stored in LangSmith platform
6. **Analysis**: View, query, and analyze via UI or API

### Key Architecture Principles

- **Zero Latency**: Async processing doesn't slow your application
- **Fault Tolerant**: LangSmith issues don't affect your app
- **Framework Agnostic**: Works with any LLM framework, not just LangChain
- **Scalable**: Handles high-volume production workloads
- **Secure**: Data encryption, role-based access control

---

## Integration with LLM Applications

### Generic LLM Applications

LangSmith works with any LLM application, regardless of framework:

**Using the `@traceable` Decorator (Python):**

```python
import os
from langsmith import traceable
from openai import OpenAI

# Set environment variables
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_TRACING"] = "true"

client = OpenAI()

@traceable
def generate_summary(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the following text"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# This call is automatically traced
result = generate_summary("Long article text here...")
```

**Wrapping OpenAI Client:**

```python
from langsmith import wrappers

# Automatically trace all OpenAI calls
wrapped_client = wrappers.wrap_openai(client)
```

### Custom Metadata and Tags

Organize traces with metadata for better filtering and analysis:

```python
@traceable(
    tags=["production", "customer-support", "v2.0"],
    metadata={
        "user_tier": "premium",
        "feature": "summarization",
        "model_version": "gpt-4-turbo"
    }
)
def my_function(input_data):
    # Your logic here
    pass
```

---

## Working with RAG Systems

Retrieval-Augmented Generation (RAG) systems are particularly complex, involving multiple components. LangSmith excels at debugging and evaluating RAG pipelines.

### RAG Components Traced

1. **Query Processing**: Input validation and transformation
2. **Embedding Generation**: Converting query to vector
3. **Vector Search**: Retrieving relevant documents
4. **Context Assembly**: Combining retrieved documents
5. **Prompt Construction**: Building final prompt with context
6. **LLM Generation**: Generating answer
7. **Post-Processing**: Formatting and validation

### RAG-Specific Evaluation Metrics

**Retrieval Quality:**
- **Context Precision**: Are retrieved documents relevant?
- **Context Recall**: Are all relevant documents retrieved?
- **Retrieval Relevance**: Quality of retrieval mechanism

**Generation Quality:**
- **Faithfulness**: Is answer based only on retrieved context?
- **Answer Relevance**: Does answer address the question?
- **Groundedness**: No hallucinations beyond provided context
- **Completeness**: Does answer fully address the query?

### Example RAG Evaluation

```python
from langsmith import Client, traceable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

client = Client()

@traceable
def retrieve_docs(question: str):
    # Retrieval logic
    return retriever.get_relevant_documents(question)

@traceable
def rag_pipeline(question: str):
    docs = retrieve_docs(question)
    context = "\n".join([doc.page_content for doc in docs])
    
    llm = ChatOpenAI(model="gpt-4")
    response = llm.invoke(f"Context: {context}\n\nQuestion: {question}")
    
    return response.content

# Create evaluation dataset
dataset_name = "rag-qa-dataset"
examples = [
    {
        "inputs": {"question": "What is RAG?"},
        "outputs": {"answer": "RAG combines retrieval and generation..."}
    }
]

dataset = client.create_dataset(dataset_name, examples=examples)

# Run evaluation
def correctness_evaluator(run, example):
    # Custom evaluation logic
    return {"score": similarity_score(run.outputs, example.outputs)}

results = client.evaluate(
    rag_pipeline,
    data=dataset_name,
    evaluators=[correctness_evaluator],
    experiment_prefix="rag-v1"
)
```

### Debugging RAG Issues

Common RAG problems LangSmith helps diagnose:

1. **Poor Retrieval**: See exactly which documents were retrieved
2. **Irrelevant Context**: Identify when vector search returns wrong documents
3. **Hallucinations**: Compare generated answer with retrieved context
4. **Chunking Issues**: Analyze if document splits are optimal
5. **Embedding Quality**: Test different embedding models
6. **Prompt Effectiveness**: Experiment with context formatting

---

## Agent Development & Debugging

Agents are autonomous systems that reason, plan, and use tools. They're notoriously difficult to debug due to their multi-step, dynamic nature. LangSmith provides specialized tools for agent development.

### Challenges with Agents

- **Long Execution Times**: Agents can run for minutes or hours
- **Complex Reasoning**: Multiple steps of planning and reflection
- **Tool Usage**: Calling external APIs and functions
- **State Management**: Maintaining context across steps
- **Non-linear Flow**: Agents may loop, backtrack, or branch

### Deep Agent Debugging

LangSmith offers specific features for "deep agents":

**1. Polly - AI-Powered Analysis**

Polly is an in-app AI assistant that analyzes your agent traces:

- Understands full agent trajectories
- Identifies failure patterns
- Suggests prompt improvements
- Explains unexpected behaviors
- Recommends architectural changes

**2. LangSmith Fetch CLI**

For developers who prefer working in the terminal:

```bash
# Install
pip install langsmith-fetch

# Fetch most recent trace
langsmith-fetch traces --project-uuid <uuid> --format json

# Get last 30 minutes of activity
langsmith-fetch traces --last-n-minutes 30

# Fetch threads for analysis
langsmith-fetch threads ./agent-data --limit 50

# Pipe to coding agents
langsmith-fetch traces | claude-code "analyze this trace"
```

**3. Visual Agent Debugging**

- Hierarchical trace visualization
- Step-by-step execution flow
- Tool call inspection
- Memory state tracking
- Decision point analysis

### Agent Evaluation

Evaluating agent performance requires different metrics:

- **Task Success Rate**: Did agent complete the objective?
- **Efficiency**: Number of steps to completion
- **Tool Usage**: Are tools used appropriately?
- **Cost**: Token usage and API costs
- **Latency**: Time to completion
- **Error Recovery**: How agents handle failures

### Example Agent Testing

```python
from langsmith import Client

client = Client()

def test_agent(inputs: dict):
    return agent.invoke(inputs["task"])

# Create agent test dataset
agent_tests = [
    {
        "inputs": {"task": "Find and summarize latest AI research"},
        "outputs": {"success": True, "steps": 5}
    }
]

# Run evaluation
results = client.evaluate(
    test_agent,
    data="agent-test-suite",
    evaluators=[
        task_success_evaluator,
        efficiency_evaluator,
        cost_evaluator
    ]
)
```

---

## LangChain & LangGraph Integration

LangSmith integrates seamlessly with LangChain and LangGraph, providing zero-config tracing.

### LangChain Integration

**Automatic Tracing:**

```python
import os

# Enable tracing with environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All LangChain components automatically traced
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("Translate {text} to {language}")

# This chain execution is fully traced
chain = prompt | llm
result = chain.invoke({"text": "Hello", "language": "French"})
```

**Features:**
- No code changes required
- Automatic prompt capture
- Chain visualization
- Token tracking
- Error logging

### LangGraph Integration

LangGraph is a framework for building stateful, multi-agent applications. LangSmith provides deep integration:

**Automatic State Tracking:**

```python
from langgraph.graph import StateGraph
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Define graph
workflow = StateGraph(state_schema)

# Add nodes
workflow.add_node("research", research_agent)
workflow.add_node("write", writing_agent)

# LangSmith automatically traces:
# - State transitions
# - Node executions
# - Edge decisions
# - Agent reasoning
```

**LangGraph Studio Integration:**

LangGraph Studio is a visual IDE for agent development, fully integrated with LangSmith:

- Visual graph representation
- Real-time execution visualization
- State inspection and editing
- Breakpoint debugging
- Direct deployment to LangGraph Platform

**Production Features:**
- **Durable Execution**: Agents resume after failures
- **Human-in-the-Loop**: Pause for human approval
- **Memory Management**: Short and long-term memory
- **Token Streaming**: Real-time output display

---

## Getting Started

### Quick Setup (5 Minutes)

**1. Create Account:**
Visit [smith.langchain.com](https://smith.langchain.com) and sign up

**2. Get API Key:**
Navigate to Settings → API Keys → Create API Key

**3. Install SDK:**

```bash
# Python
pip install langsmith

# JavaScript/TypeScript
npm install langsmith
```

**4. Configure Environment:**

```bash
export LANGSMITH_API_KEY="your-api-key"
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="my-first-project"
```

**5. Start Tracing:**

```python
from langsmith import traceable

@traceable
def my_llm_function(user_input: str):
    # Your LLM logic
    return response

# That's it! Your function is now traced
```

### Project Organization

**Best Practices:**
- Create separate projects for development, staging, production
- Use tags to categorize traces (feature, user tier, etc.)
- Add metadata for filtering and analysis
- Set up custom project names per environment

```python
from langsmith import tracing_context

with tracing_context(
    project_name="production-chatbot",
    tags=["v2.0", "customer-support"],
    metadata={"environment": "prod", "region": "us-east"}
):
    result = my_function(input_data)
```

---

## Best Practices

### 1. Tracing Strategy

**Instrument Early:**
- Add tracing from the start of development
- Wrap all LLM calls and key functions
- Tag appropriately for easy filtering

**Consistent Naming:**
```python
config = {
    "tags": [
        "env:production",      # Environment
        "model:gpt-4",         # Model
        "feature:qa-bot",      # Feature
        "version:2.1.0"        # Version
    ],
    "metadata": {
        "user_id": "user_123",
        "session_id": "sess_456",
        "ab_test": "variant_b"
    }
}
```

### 2. Dataset Management

**Build Quality Datasets:**
- Start with hand-crafted examples
- Add edge cases and failure modes
- Include diverse query types
- Continuously enrich from production

**Dataset Sources:**
```python
# From production traces
client.create_dataset_from_runs(
    dataset_name="production-failures",
    run_ids=[run_id for run_id in failed_runs],
    description="Real failures from production"
)

# Manual examples
examples = [
    {"inputs": {...}, "outputs": {...}},
]
client.create_dataset(dataset_name, examples=examples)
```

### 3. Evaluation Workflow

**Continuous Evaluation:**
```python
# Run evaluations in CI/CD
def test_model_quality():
    results = client.evaluate(
        my_function,
        data="regression-test-suite",
        evaluators=[accuracy, relevance, cost],
    )
    
    # Assert minimum quality thresholds
    assert results.metrics["accuracy"] > 0.85
    assert results.metrics["avg_cost"] < 0.05
```

**Multi-Metric Approach:**
- Combine automated and human evaluation
- Use LLM-as-judge for subjective criteria
- Implement heuristic checks for objective criteria
- Periodic human spot-checks for calibration

### 4. Production Monitoring

**Set Up Alerts:**
- Error rate thresholds
- Latency spikes
- Cost anomalies
- Quality degradation

**Monitor Key Metrics:**
- Token usage and costs
- Latency (p50, p95, p99)
- Error rates
- User feedback scores
- Time-to-first-token

### 5. Cost Management

**Track and Optimize:**
```python
config = {
    "metadata": {
        "expected_cost_cents": 5,
        "cost_center": "customer-support",
        "budget_category": "ai-operations"
    }
}

# LangSmith tracks actual costs
# Compare expected vs actual
# Identify expensive operations
```

### 6. Team Collaboration

**Annotation Queues:**
- Route problematic outputs to experts
- Collect structured feedback
- Build golden datasets
- Calibrate automated evaluators

**Sharing Best Practices:**
- Share prompts via LangChain Hub
- Document evaluation criteria
- Version control datasets
- Regular team reviews of traces

---

## Production Deployment

### Deployment Checklist

**Pre-Production:**
- [ ] Comprehensive evaluation suite
- [ ] Performance benchmarking
- [ ] Cost estimation and budgeting
- [ ] Error handling and fallbacks
- [ ] Monitoring and alerting setup
- [ ] Security and compliance review

**Production Configuration:**

```python
# Separate production project
os.environ["LANGSMITH_PROJECT"] = "production"

# Optional: Sample traces to reduce volume
os.environ["LANGSMITH_SAMPLING_RATE"] = "0.1"  # 10% sampling

# Configure alerts
# (via LangSmith UI or API)
```

### Monitoring Strategy

**Real-Time Monitoring:**
- Dashboard for key metrics
- Automated alerts for anomalies
- Integration with existing tools (Datadog, PagerDuty)

**Feedback Loops:**
```python
from langsmith import Client

client = Client()

# Collect user feedback
client.create_feedback(
    run_id=run.id,
    key="user_rating",
    score=4.5,
    comment="Helpful response"
)

# Use feedback to improve
# - Add to evaluation datasets
# - Identify improvement areas
# - Track satisfaction trends
```

### CI/CD Integration

**Automated Quality Gates:**

```yaml
# .github/workflows/evaluate.yml
name: LLM Quality Tests

on: [push, pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Evaluations
        env:
          LANGSMITH_API_KEY: ${{ secrets.LANGSMITH_API_KEY }}
        run: |
          python -m pytest tests/llm_tests.py
          # Fails if quality metrics below threshold
```

### Self-Hosting (Enterprise)

For organizations requiring data sovereignty:

- Deploy on your Kubernetes cluster (AWS, GCP, Azure)
- Data never leaves your environment
- Same features as cloud version
- Enterprise support included

---

## Pricing & Plans

### Free Tier
- Unlimited projects
- 1 million trace spans
- Core evaluation features
- Perfect for prototyping and small apps

### Pro Plan ($249/month)
- Unlimited trace spans
- 5GB processed data
- 50,000 evaluation scores
- Production deployment support
- Priority support

### Enterprise
- High-volume applications
- Dedicated infrastructure
- Self-hosting options
- Custom integrations
- SLA guarantees
- Advanced security features

**Cost Optimization:**
- Sample traces in production (e.g., 10% sampling)
- Archive old traces
- Focus monitoring on critical paths
- Use evaluation efficiently

---

## Conclusion

### Why LangSmith is Essential

Building reliable LLM applications requires more than just good code. You need:

1. **Visibility**: Understand what your application is actually doing
2. **Quality Assurance**: Systematically test and evaluate outputs
3. **Debugging Tools**: Quickly identify and fix issues
4. **Performance Monitoring**: Track costs, latency, and quality in production
5. **Continuous Improvement**: Iterate based on real data and feedback

LangSmith provides all of these in a single, integrated platform.

### When to Use LangSmith

**Essential For:**
- Production LLM applications
- RAG systems with complex retrieval
- Multi-agent systems
- Applications requiring high reliability
- Teams needing collaboration tools
- Cost-sensitive deployments

**Especially Valuable When:**
- Building beyond simple chatbots
- Deploying to users/customers
- Debugging non-deterministic behavior
- Optimizing for quality and cost
- Scaling from prototype to production

### Key Takeaways

1. **Start Early**: Instrument your application from day one
2. **Test Systematically**: Build evaluation datasets and run regular tests
3. **Monitor Continuously**: Track performance in production
4. **Iterate Based on Data**: Use traces and feedback to improve
5. **Framework Agnostic**: Works with any LLM framework
6. **Zero Latency**: Doesn't slow down your application
7. **Production Ready**: Scales from prototype to enterprise

### Getting Help

- **Documentation**: [docs.langchain.com/langsmith](https://docs.langchain.com/langsmith)
- **Support**: [support.langchain.com](https://support.langchain.com)
- **Community**: Discord, GitHub Discussions
- **Tutorials**: LangSmith Cookbook, YouTube videos
- **Blog**: [blog.langchain.com](https://blog.langchain.com)

### Final Thoughts

The difference between a demo and a production LLM application is systematic evaluation and monitoring. LangSmith allows you to check the output of LLMs against built-in criteria and provides powerful evaluation and monitoring tools. Whether you're using LangChain, LangGraph, or building custom LLM applications, LangSmith gives you the observability and confidence needed to ship reliable AI products.

Start building with confidence. Start with LangSmith.

---

## Quick Reference

### Essential Commands

```bash
# Install
pip install langsmith

# Configure
export LANGSMITH_API_KEY="your-key"
export LANGSMITH_TRACING="true"
export LANGSMITH_PROJECT="my-project"

# Fetch CLI
pip install langsmith-fetch
langsmith-fetch traces --project-uuid <uuid>
```

### Essential Code Patterns

```python
# Basic tracing
from langsmith import traceable

@traceable
def my_function(input):
    return output

# With metadata
@traceable(tags=["prod"], metadata={"version": "1.0"})
def my_function(input):
    return output

# Context manager
from langsmith import tracing_context

with tracing_context(project_name="test"):
    result = function()

# Create dataset
from langsmith import Client
client = Client()
client.create_dataset("my-dataset", examples=[...])

# Run evaluation
client.evaluate(
    target_function,
    data="my-dataset",
    evaluators=[evaluator1, evaluator2]
)
```

### Common Use Cases

| Use Case | Primary Feature | Key Benefit |
|----------|----------------|-------------|
| Debugging chains | Tracing | See every step |
| Testing quality | Evaluation | Systematic testing |
| RAG optimization | Metrics | Measure retrieval & generation |
| Agent development | Deep debugging | Understand reasoning |
| Production monitoring | Alerts | Catch issues early |
| Cost optimization | Token tracking | Reduce spending |
| Team collaboration | Datasets & Hub | Share knowledge |

---

**Ready to build production-ready LLM applications? Sign up at [smith.langchain.com](https://smith.langchain.com) and start tracing in 5 minutes.**