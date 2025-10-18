# OpenMind Wrangler: AI Agents for Data Preparation in the AI 3.0 Era

This is the repository for the exploratory project OpenMind Wranglers, inspired by the accelerating progress of AI 3.0 models and their potential to automate complex data engineering workflows for multi-study neuroimaging and clinical datasets.

## Project Aim

The goal of this project is to explore whether AI agents (primarily LLM-based) can meaningfully assist in dataset wrangling tasks that precede model inference or training — tasks that are time-consuming yet essential to reproducible, large-scale neuroimaging research.

While the BIDS standard ensures interoperability on the metadata level, many preprocessing steps — such as volume normalization, quality control, outlier detection, or label encoding — remain manual or semi-automated. These steps form a bottleneck in data-driven science, especially when aggregating datasets across multiple studies.

In this proof-of-concept, we aim to determine whether a coordinated system of AI agents can reliably execute these operations and produce AI-ready dataset collections, similar to those hosted on platforms like HuggingFace Datasets: OpenMind, with minimal human intervention.

## The Vision

If successful, OpenMind Wrangler will serve as a foundation for a general-purpose AI data engineering assistant, capable of producing multi-study datasets that are immediately usable for machine learning and statistical analysis — without manual wrangling.

## Rough Work Plan

The project will proceed along four parallel tracks: agent design, dataset exploration, wrangling task automation, and evaluation.

### 1. Agent Design

We will prototype a multi-agent system where each agent handles a specific wrangling subtask (e.g., normalization, QC, encoding). Agents will communicate via a shared memory and task queue.

Goals:

Design a modular agentic framework (e.g., LangChain, CrewAI, or AutoGen)

Define prompt templates and tools for each subtask

Implement logging and reasoning traceability

Resources:

LangChain Agent Docs

OpenAI Function Calling Guide

### 2. Dataset Exploration

OpenNeuro

Tasks:

Retrieve metadata and BIDS structures

Generate schema summaries and compatibility maps

Identify data types for downstream analysis

### 3. Wrangling Task Automation

The central track: enabling LLM-driven automation of the following operations:

Volume normalization (using NiBabel / ANTsPy)

Quality control report generation

Outlier detection (statistical and visual)

Label harmonization and encoding

Data documentation generation (Markdown / JSON-LD)

The focus is not on achieving perfection, but on evaluating how well an AI agent can assist or autonomously perform these tasks, given contextual metadata and goals.

### 4. Evaluation and Benchmarking

The evaluation phase will assess the efficiency, accuracy, and robustness of agentic wrangling workflows compared to traditional, human-coded pipelines.

Metrics will include:

Time saved per dataset compared to manual pipelines

Accuracy of normalization / encoding

Error detection rate (false positives / negatives in QC)

Consistency across heterogeneous datasets

We will also benchmark against a simple scripted baseline (e.g., manual nipype or pandas pipeline).

Milestones

✅ Literature & tooling review on AI-assisted data wrangling

⚙️ Prototype LLM agent framework for dataset preprocessing

🧠 Evaluation on 2–3 public BIDS datasets

📊 Quantitative + qualitative benchmarking report


Generative augmentation: Can the agent propose synthetic data to fill gaps or balance classes?

Teams and Participants
