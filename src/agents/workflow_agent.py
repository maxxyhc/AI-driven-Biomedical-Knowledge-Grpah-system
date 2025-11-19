"""
LangGraph workflow agent for biomedical knowledge graphs.

5-step workflow: Classify → Extract → Generate → Execute → Format
"""

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

from anthropic import Anthropic
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from .graph_interface import GraphInterface


class WorkflowState(TypedDict):
    """State that flows through the workflow steps."""

    user_question: str
    question_type: Optional[str]
    entities: Optional[List[str]]
    cypher_query: Optional[str]
    results: Optional[List[Dict]]
    #new features store conversation memory
    history: Optional[list[dict[str, str]]]
    reasoning_steps: Optional[List[Dict[str, str]]]
    reasoning_trace: Optional[List[str]]
    final_answer: Optional[str]
    error: Optional[str]


class WorkflowAgent:
    """LangGraph workflow agent for biomedical knowledge graphs."""

    # Class constants
    MODEL_NAME = "claude-sonnet-4-20250514"
    DEFAULT_MAX_TOKENS = 200

    # Default schema query
    SCHEMA_QUERY = (
        "MATCH (n) RETURN labels(n) as node_type, count(n) as count "
        "ORDER BY count DESC LIMIT 10"
    )

    def __init__(self, graph_interface: GraphInterface, anthropic_api_key: str):
        self.graph_db = graph_interface
        self.anthropic = Anthropic(api_key=anthropic_api_key)
        self.schema = self.graph_db.get_schema_info()
        self.property_values = self._get_key_property_values()
        self.workflow = self._create_workflow()


    def _get_key_property_values(self) -> Dict[str, List[Any]]:
        """Get property values dynamically from all nodes and relationships.

        This method discovers all available properties in the database schema and
        collects sample values for each property. This enables the LLM to generate
        more accurate queries by knowing what property values actually exist.

        Returns:
            Dict mapping property names to lists of sample values from the database
        """
        values = {}
        try:
            # Discover and collect property values from all node types in the database
            # This replaces hardcoded property lists with dynamic schema discovery
            for node_label in self.schema.get("node_labels", []):
                # Get all properties that exist for this node type
                node_props = self.schema.get("node_properties", {}).get(node_label, [])
                for prop_name in node_props:
                    # Avoid duplicate property names (same property might exist
                    # on multiple node types)
                    if prop_name not in values:
                        # Query the database for actual property values
                        # (limited to 20 for performance)
                        prop_values = self.graph_db.get_property_values(
                            node_label, prop_name
                        )
                        # Only store properties that have actual values in the database
                        if prop_values:
                            values[prop_name] = prop_values

            # Discover and collect property values from all relationship types
            # This ensures we capture relationship-specific properties like
            # confidence, weight, etc.
            for rel_type in self.schema.get("relationship_types", []):
                # GraphInterface expects 'REL_' prefix for relationship queries
                rel_label = f"REL_{rel_type}"
                # Get all properties that exist for this relationship type
                rel_props = self.schema.get("relationship_properties", {}).get(
                    rel_type, []
                )
                for prop_name in rel_props:
                    # Skip if we already have this property from a node type
                    if prop_name not in values:
                        try:
                            # Query relationship properties using the REL_ prefix
                            # convention
                            prop_values = self.graph_db.get_property_values(
                                rel_label, prop_name
                            )
                            # Only store if the relationship actually has values
                            # for this property
                            if prop_values:
                                values[prop_name] = prop_values
                        except Exception:
                            # Some relationships might not have certain properties -
                            # skip gracefully
                            continue

        except Exception:
            pass
        return values

    def _get_llm_response(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Get response from LLM and handle content extraction."""
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS

        try:
            response = self.anthropic.messages.create(
                model=self.MODEL_NAME,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0]
            return content.text.strip() if hasattr(content, "text") else str(content)
        except Exception as e:
            raise RuntimeError(f"LLM response failed: {str(e)}")

    def _create_workflow(self) -> Any:
        """Create the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("classify", self.classify_question)
        workflow.add_node("extract", self.extract_entities)
        workflow.add_node("generate", self.generate_query)
        workflow.add_node("execute", self.execute_query)
        workflow.add_node("format", self.format_answer)
        workflow.add_node("reflect", self.reflect_answer) # adding new node

        workflow.add_edge("classify", "extract")
        workflow.add_edge("extract", "generate")
        workflow.add_edge("generate", "execute")
        workflow.add_edge("execute", "format")
        workflow.add_edge("format", "reflect")  # let format go into reflect
        workflow.add_edge("reflect", END)

        workflow.set_entry_point("classify")
        return workflow.compile()

    def _build_classification_prompt(self, question: str) -> str:
        """Build classification prompt with consistent formatting."""
        return f"""Classify this biomedical question. Choose one:
- gene_disease: genes and diseases
- drug_treatment: drugs and treatments
- protein_function: proteins and functions
- general_db: database exploration
- general_knowledge: biomedical concepts

Question: {question}

Respond with just the type."""

    def classify_question(self, state: WorkflowState) -> WorkflowState:
        """Classify the biomedical question type using an LLM.

        Uses LLM-based classification instead of hardcoded keyword matching for
        more flexible and accurate question type detection. This enables the agent
        to handle nuanced questions that don't fit simple keyword patterns.
        """
        try:
            # Build classification prompt with available question types
            prompt = self._build_classification_prompt(state["user_question"])
            # Use minimal tokens since we only need a single classification word
            
            #add a prompt for reasoning thinking
            reasoning_prompt = f"Think step-by-step before classifying:\n{prompt}"

            reasoning = self._get_llm_response(reasoning_prompt, max_tokens=50)
            # use last sentence as classification result
            classification = reasoning.split("\n")[-1].strip()
            state["question_type"] = classification

            # record the reasoning steps
            if "reasoning_steps" not in state or state["reasoning_steps"] is None:
                state["reasoning_steps"] = []
            state["reasoning_steps"].append({
                "step": "classify",
                "thought": reasoning
            })

        except Exception as e:
            # If classification fails, record error but continue with safe fallback
            state["error"] = f"Classification failed: {str(e)}"
            # Default to general knowledge to avoid database queries with
            # malformed inputs
            state["question_type"] = "general_knowledge"
        return state

    def extract_entities(self, state: WorkflowState) -> WorkflowState:
        """Extract biomedical entities from the question.

        Uses the database schema to guide entity extraction, ensuring that
        only entities that can actually be found in the database are
        extracted.
        This improves query generation accuracy by providing relevant context.
        """
        # Skip entity extraction for questions that don't need
        # database-specific entities
        question_type = state.get("question_type")
        if question_type in ["general_db", "general_knowledge"]:
            # General questions don't need specific entity extraction
            state["entities"] = []
            return state

        # Build dynamic property information from actual database content
        # This replaces hardcoded examples with real data from the database
        property_info = []
        for prop_name, values in self.property_values.items():
            if values:  # Only show properties with actual values in database
                # Show first 3 values as representative examples for the LLM
                sample_values = ", ".join(str(v) for v in values[:3])
                property_info.append(f"- {prop_name}: {sample_values}")

        entity_types_str = ", ".join(self.schema.get("node_labels", []))
        relationship_types_str = ", ".join(self.schema.get("relationship_types", []))

        prompt = (
            f"""Extract biomedical terms and concepts from this question """
            f"""based on the database schema:

Available entity types: {entity_types_str}
Available relationships: {relationship_types_str}

Available property values in database:
{chr(10).join(property_info) if property_info else "- No property values available"}

Question: {state['user_question']}

Extract ALL relevant terms including:
- Specific entity names mentioned
- Entity types referenced
- Property values or constraints
- Relationship concepts
- General biomedical concepts

Return a JSON list: ["term1", "term2"] or []"""
        )

        try:
            response_text = self._get_llm_response(prompt, max_tokens=100)

            # Clean up response text before JSON parsing
            cleaned_text = response_text.strip()
            if cleaned_text.startswith("```json"):
                cleaned_text = (
                    cleaned_text.replace("```json", "").replace("```", "").strip()
                )

            state["entities"] = json.loads(cleaned_text)
        except (json.JSONDecodeError, AttributeError):
            # Fallback to empty list if JSON parsing fails
            state["entities"] = []

        if "reasoning_steps" not in state:
            state["reasoning_steps"] = []
        state["reasoning_steps"].append({
            "step" : "extract",
            "thought" : f"Extracted entities: {state['entities']}"
        })

        return state

    def generate_query(self, state: WorkflowState) -> WorkflowState:
        """Generate Cypher query based on question type.

        Creates database queries dynamically using the actual schema and property
        values discovered from the database. This ensures queries are valid and
        use only properties/relationships that actually exist.
        """
        question_type = state.get("question_type", "general")

        # Database exploration questions use a simple schema overview query
        if question_type == "general_db":
            # Use predefined query to show database structure and content overview
            state["cypher_query"] = self.SCHEMA_QUERY
            return state

        # General knowledge questions don't need database queries
        if question_type == "general_knowledge":
            # Skip database query for conceptual questions that don't need data lookup
            state["cypher_query"] = None
            return state

        # Build dynamic relationship guide from actual database schema
        # This replaces hardcoded relationship patterns with discovered relationships
        relationship_guide = f"""
Available relationships:
{' | '.join([f'- {rel}' for rel in self.schema['relationship_types']])}"""

        # Build comprehensive property information from database discovery
        # This gives the LLM concrete examples of what property values exist
        property_details = []
        for prop_name, values in self.property_values.items():
            if values:  # Only include properties with actual values in the database
                # Auto-detect value type to help LLM understand data format
                value_type = (
                    "text values" if isinstance(values[0], str) else "numeric values"
                )
                property_details.append(f"- {prop_name}: {values} ({value_type})")

        property_info = f"""Property names and values:
Node properties: {self.schema['node_properties']}
Available property values:
{chr(10).join(property_details) if property_details else "- No values available"}
Use WHERE property IN [value1, value2] for filtering."""
        prompt = f"""Create a Cypher query for this biomedical question:

Question: {state['user_question']}
Type: {question_type}
Schema:
Nodes: {', '.join(self.schema['node_labels'])}
Relations: {', '.join(self.schema['relationship_types'])}
{property_info}
{relationship_guide}
Entities: {state.get('entities', [])}

Use MATCH, WHERE with CONTAINS for filtering, RETURN, LIMIT 10.
IMPORTANT: Use property names from schema above and IN filtering for property values.
Return only the Cypher query."""

        cypher_query = self._get_llm_response(prompt, max_tokens=150)

        # Clean up LLM response formatting (remove markdown code blocks)
        # LLMs often wrap code in ```cypher blocks, so we need to extract just the query
        if cypher_query.startswith("```"):
            cypher_query = "\n".join(
                line
                for line in cypher_query.split("\n")
                # Remove markdown code block markers and language specifiers
                if not line.startswith("```") and not line.startswith("cypher")
            ).strip()

        state["cypher_query"] = cypher_query
        
        #record the reasoning
        if "reasoning_steps" not in state:
            state["reasoning_steps"] = []
        state["reasoning_steps"].append({
            "step": "generate",
            "thought": f"Generated query:\n{state['cypher_query']}"
        })

        return state

    def execute_query(self, state: WorkflowState) -> WorkflowState:
        """Execute the generated Cypher query against the database.

        Safely executes the LLM-generated query with error handling to prevent
        crashes from malformed queries while capturing useful error information.
        """
        try:
            query = state.get("cypher_query")
            # Execute query only if one was generated (some question types
            # skip this step)
            state["results"] = self.graph_db.execute_query(query) if query else []
        except Exception as e:
            # Capture query execution errors but continue workflow to provide
            # helpful feedback
            state["error"] = f"Query failed: {str(e)}"
            # Set empty results so the format step can handle the error gracefully
            state["results"] = []

        #record the reasoning
        if "reasoning_steps" not in state:
            state["reasoning_steps"] = []
        query = state.get("cypher_query")
        status = "executed successfully" if state.get("results") else "no results or error"
        state["reasoning_steps"].append({
            "step": "execute",
            "thought": f"Executed query ({status}): {query}"
        })

        return state

    def format_answer(self, state: WorkflowState) -> WorkflowState:
        """Format database results into human-readable answer.

        Takes raw database results and converts them into natural language
        responses, handling different question types and error conditions.
        """
        # Handle any errors that occurred during the workflow
        if state.get("error"):
            state["final_answer"] = (
                f"Sorry, I had trouble with that question: {state['error']}"
            )
            return state

        question_type = state.get("question_type")

        # General knowledge questions use LLM knowledge instead of database results
        if question_type == "general_knowledge":
            # Generate answer from LLM's training knowledge rather than database lookup
            state["final_answer"] = self._get_llm_response(
                f"""Answer this general biomedical question using your knowledge:

Question: {state['user_question']}

Provide a clear, informative answer about biomedical concepts.""",
                max_tokens=300,  # Allow more tokens for explanatory content
            )
            return state

        # Handle database-based answers using query results
        results = state.get("results", [])
        if not results:
            # No results found - provide helpful guidance for next steps
            state["final_answer"] = (
                "I didn't find any information for that question. Try asking about "
                "genes, diseases, or drugs in our database."
            )
            return state

        # Convert raw database results into natural language using LLM
        state["final_answer"] = self._get_llm_response(
            f"""Convert these database results into a clear answer:

Question: {state['user_question']}
Results: {json.dumps(results[:5], indent=2)}
Total found: {len(results)}

Make it concise and informative.""",
            max_tokens=250,  # Balanced token limit for informative but concise
            # responses
        )

        #record the reasoning
        if "reasoning_steps" not in state:
            state["reasoning_steps"] = []
        summary = state.get("final_answer", "")[:200]
        state["reasoning_steps"].append({
            "step": "format",
            "thought": f"Formatted final answer preview: {summary}"
        })

        return state
    
    #added new func for llm to reflect its own answer
    def reflect_answer(self, state: WorkflowState) -> WorkflowState:
        """Reflect on the generated answer and verify reasoning consistency."""
        try:
            reflection_prompt = f"""
            You are a critical reviewer.
            Review this biomedical Q&A process and identify any logical gaps, missing details,
            or incorrect assumptions. Suggest how the answer could be improved.

            Question: {state.get('user_question')}
            Reasoning trace: {json.dumps(state.get('reasoning_steps', []), indent=2)}
            Final Answer: {state.get('final_answer')}
            """

            reflection = self._get_llm_response(reflection_prompt, max_tokens=150)

            # Save the reflection result
            if "reasoning_steps" not in state:
                state["reasoning_steps"] = []
            state["reasoning_steps"].append({
                "step": "reflect",
                "thought": reflection
            })

            # Optionally append the reflection to the final answer
            state["final_answer"] += f"\n\n🪞 Reflection: {reflection}"

        except Exception as e:
            state["error"] = f"Reflection failed: {str(e)}"

        return state


    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a biomedical question using the LangGraph workflow."""

        if not hasattr(self, "conversation_state"):
            self.conversation_state = WorkflowState(
                user_question=question,
                question_type=None,
                entities=None,
                cypher_query=None,
                results=None,
                history=[], #initial conversation memory
                reasoning_steps=None,
                reasoning_trace=[],
                final_answer=None,
                error=None,
            )
        else:
            self.conversation_state["user_question"] = question

        # Run workflow - let it complete naturally
        final_state = self.workflow.invoke(self.conversation_state)

        # Get the answer generated by the workflow (includes reflection)
        model_output = final_state.get("final_answer", "No answer generated")

        # Update conversation memory with the workflow result
        self.update_memory(self.conversation_state, question, model_output)

        if "reasoning_trace" not in self.conversation_state:
            self.conversation_state["reasoning_trace"] = []
        self.conversation_state["reasoning_trace"].append(
            "\n".join(
                f"{step['step']}: {step['thought']}"
                for step in final_state.get("reasoning_steps", [])
                if "thought" in step
            )
        )
        
        # print("🧠 Reasoning trace:", final_state.get("reasoning_steps"))
        return {
            "answer": model_output,
            "history": self.conversation_state["history"],
            "question_type": final_state.get("question_type"),
            "entities": final_state.get("entities", []),
            "cypher_query": final_state.get("cypher_query"),
            "results_count": len(final_state.get("results", [])),
            "raw_results": final_state.get("results", [])[:3],
            "error": final_state.get("error"),
            "reasoning_steps": final_state.get("reasoning_steps", []),
        }

    #new func for updata memory when aksing question
    def update_memory(self, state: WorkflowState, user_input: str, model_output: str):
        if "history" not in state or state["history"] is None:
            state["history"] = []
        state["history"].append({
            "user": user_input,
            "assistant": model_output
        })

    def reset_memory(self):
        """Manually clear the conversation history for a fresh start."""
        if hasattr(self, "conversation_state"):
            # Clear conversation memory
            self.conversation_state["history"] = []

            # Clear reasoning trace persistence
            self.conversation_state["reasoning_trace"] = []

            # Clear previous reasoning steps
            self.conversation_state["reasoning_steps"] = None

            # Clear previous question context
            self.conversation_state["user_question"] = ""
            self.conversation_state["question_type"] = None
            self.conversation_state["entities"] = None

            # Clear previous query and results
            self.conversation_state["cypher_query"] = None
            self.conversation_state["results"] = None

            # Clear previous answer and errors
            self.conversation_state["final_answer"] = None
            self.conversation_state["error"] = None

            print("🧹 Conversation memory has been cleared.")
        else:
            print("⚠️ No active conversation found to reset.")

    def build_prompt(self, state: WorkflowState) -> str:
        """Context-aware prompt builder that incorporates relevant history and entities to help the model reason across multiple turns."""
        history = state.get("history", [])
        recent_context = ""
        related_entities = []

        # Include last 3 exchanges as conversational context
        if history:
            for turn in history[-3:]:
                recent_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        # Automatically extract recurring biomedical entities from history
        for turn in history[-5:]:
            if "gene" in turn["user"].lower() or "gene" in turn["assistant"].lower():
                related_entities.append("gene")
            if "protein" in turn["user"].lower() or "protein" in turn["assistant"].lower():
                related_entities.append("protein")
            if "drug" in turn["user"].lower() or "drug" in turn["assistant"].lower():
                related_entities.append("drug")
            if "disease" in turn["user"].lower() or "disease" in turn["assistant"].lower():
                related_entities.append("disease")

        # Deduplicate keywords
        related_entities = list(set(related_entities))
        print(f"🧠 build_prompt — Detected previous entities: {', '.join(related_entities) if related_entities else 'none'}")

        current_question = state.get("user_question","")

        # Simple topic similarity guard — avoid unrelated context
        if history and not any(word in current_question.lower() for word in ["gene", "protein", "drug", "disease"]):
            print("🧩 Skipping old context — new question seems unrelated.")
            recent_context = ""
            related_entities = []

        # Build adaptive context instructions
        context_instruction = (
            f"The user has previously asked questions involving: {', '.join(related_entities)}.\n"
            if related_entities
            else "No explicit entity context detected from previous turns.\n"
        )

        # Include last reasoning trace
        reasoning_trace = state.get("reasoning_trace", [])
        if reasoning_trace:
            recent_reasoning = "\n".join(reasoning_trace[-2:])  # include last 2 reasoning steps
            print("🪄 Continuing reasoning chain from previous run.")
        else:
            recent_reasoning = ""

        # Combine into the final prompt
        current_question = state.get("user_question", "")
        prompt = f"""
    You are a biomedical AI assistant that answers questions using knowledge graphs and LangGraph workflows.

    Here is the recent conversation history (most recent first):
    {recent_context}

    {context_instruction}

    {f"Previous reasoning context:\n{recent_reasoning}\n" if recent_reasoning else ""}
    Now answer the following question, considering all relevant prior context:
    Question: {current_question}
    """
        return prompt.strip()


def create_workflow_graph() -> Any:
    """Factory function for LangGraph Studio."""
    load_dotenv()

    graph_interface = GraphInterface(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        user=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", ""),
    )

    agent = WorkflowAgent(
        graph_interface=graph_interface,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )

    return agent.workflow

