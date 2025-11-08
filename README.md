# 🧠 Helix Navigator — AI-driven Biomedical Knowledge Graph System

An educational and research framework that integrates **Large Language Models (LLMs)**, **LangGraph**, and **Neo4j knowledge graphs** to explore reasoning across genes, proteins, diseases, and drugs.

This project demonstrates how structured biomedical data can be combined with generative reasoning to build explainable, multi-step AI workflows.

---

## 🧩 Newly Added Features

### 1. Conversation Memory
The system now tracks prior user–agent interactions to preserve context across multiple questions.
- Stores conversation history in `WorkflowState["history"]`
- Enables context-aware prompt building using previous turns
- Can be reset manually with `reset_memory()`

**Why it matters:**
This allows the agent to remember entities (e.g., “TP53”, “BRCA1”) or discussion topics (e.g., “drug resistance”) across turns, creating a more coherent, human-like dialogue.

---

### 2. Reflection Step
A new **Reflection Node** (`reflect_answer`) was added to the LangGraph workflow.
- Runs after the main reasoning chain
- Prompts the model to **critically review** its own reasoning
- Appends a reflection summary under `🪞 Reflection:` in the final output

**Why it matters:**
Encourages self-correction and transparency in AI reasoning—useful for biomedical queries where accuracy and interpretability are key.

---

### 3. Reasoning Trace Persistence
The model now maintains a `reasoning_trace` variable across interactions.
- Stores condensed summaries of reasoning steps from previous runs
- Reintegrates these traces into subsequent prompts
- Supports consistent multi-turn reasoning

**Why it matters:**
Preserves the logical flow across runs, helping the AI recall prior inference paths when analyzing related biomedical questions.
