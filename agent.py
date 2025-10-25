import logging
from typing import TypedDict, Annotated, Sequence
from operator import add
import warnings
from langchain_core.messages import BaseMessage
import google.generativeai as genai
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
from retriever_offline import retrieve_offline
from retriever_online import retrieve_online



# Completely suppress all logging and warnings
logging.basicConfig(level=logging.CRITICAL)
logging.getLogger().setLevel(logging.CRITICAL)
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
logger.setLevel(logging.CRITICAL)



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    query: str
    query_type: str  # conceptual, howto, comparison, troubleshooting
    context: str
    response: str
    mode: str


class Agent:
    
    def __init__(self, google_api_key: str = None, mode: str = "offline", tavily_api_key: str = None):
        self.mode = mode
        self.tavily_api_key = tavily_api_key
        if mode == "online" and google_api_key:
            genai.configure(api_key=google_api_key)
            self.llm = genai.GenerativeModel(
                "gemini-2.5-flash",
                generation_config={"temperature": 0.3}
            )
        elif mode == "offline":
            try:
                self.llm = Ollama(
                    model="llama3.2:latest",
                    temperature=0.3
                )
            except Exception as e:
                self.llm = None
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph agent."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("classify", self._classifier_node)
        workflow.add_node("retrieve", self._retriever_node)
        workflow.add_node("synthesize", self._synthesizer_node)
        
        # Define edges
        workflow.set_entry_point("classify")
        workflow.add_edge("classify", "retrieve")
        workflow.add_edge("retrieve", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _classifier_node(self, state: AgentState) -> AgentState:
        """Classify query type for tailored response strategy"""
        query = state['query'].lower()
        
        # Rule-based classification
        if any(word in query for word in ['how do i', 'how to', 'implement', 'add', 'create', 'setup', 'configure', 'build', 'make']):
            query_type = "howto"
        elif any(word in query for word in ['difference', 'vs', 'versus', 'compare', 'between', 'when to use', 'best', 'choose', 'tradeoff', 'which']):
            query_type = "comparison"
        elif any(word in query for word in ['error', 'fix', 'issue', 'problem', 'not working', 'fails', 'cannot', 'undefined', 'importerror', 'broken']):
            query_type = "troubleshooting"
        elif any(word in query for word in ['what is', 'what are', 'explain', 'define', 'design', 'architecture', 'why', 'concept']):
            query_type = "conceptual"
        else:
            query_type = "conceptual"
        
        return {
            **state,
            "query_type": query_type
        }
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documentation"""
        
        try:
            # Adjust top_k based on query type
            query_type = state.get('query_type', 'conceptual')
            top_k = 4 if query_type == 'comparison' else 3
            
            # Choose retrieval method based on mode
            if self.mode == "online":
                # Online: Use Tavily web search
                docs = retrieve_online(
                    state['query'], 
                    top_k=top_k,
                    api_key=self.tavily_api_key
                )
                # Combine docs into context string
                context = "\n\n".join([doc['text'] for doc in docs])
            else:
                # Offline: Use local BM25 index
                context = retrieve_offline(state['query'], top_k=top_k)
            
            # Simple filtering: keep only code and key sections
            filtered_context = self._filter_context(context, query_type)
            
            return {
                **state,
                "context": filtered_context
            }
        except Exception as e:
            # Return empty context - let generate node handle it gracefully
            return {
                **state,
                "context": ""
            }
    
    def _filter_context(self, context: str, query_type: str) -> str:
        """Filter context to keep code and key sections"""
        # Split context by double newlines to get chunks
        chunks = context.split('\n\n')
        
        filtered = []
        for chunk in chunks[:3]:  # Max 3 chunks
            lines = chunk.strip().split('\n')
            essential = []
            in_code = False
            
            for line in lines[:60]:  # Max 60 lines per chunk
                # Keep code blocks
                if '```' in line:
                    in_code = not in_code
                    essential.append(line)
                elif in_code:
                    essential.append(line)
                # Keep important lines
                elif any(kw in line.lower() for kw in ['import', 'from', 'def ', 'class ', '##', 'example:', 'usage:', 'note:', 'key']):
                    essential.append(line)
            
            if essential:
                filtered.append('\n'.join(essential))
        
        if not filtered:
            return ""
        
        result = '\n\n'.join(filtered)
        return result[:1500]  # Hard limit
    
    def _synthesizer_node(self, state: AgentState) -> AgentState:
        """Synthesize final response from context"""
        
        query_type = state.get('query_type', 'conceptual')
        context = state.get('context', '')
        
        # Base system prompt (used for all types)
        base_system_prompt = """You are a senior LangGraph developer. Give clean, natural responses.Start with working code when relevant. Keep explanations under 150 words. Use natural language without bold markers, labels, or source citations like [S1] or [S2]. Only use information from the context below. If the context is empty or irrelevant, say you couldn't find it in the docs.

        Context:
        {context}

        Mode: {mode}"""
        
        # Per-type templates
        type_prompts = {
            "howto": "Show a working code example first, then explain how it works in 2-3 sentences. Add key points if helpful.",
            
            "comparison": "Explain the key difference in one sentence, then show code examples for both options. Explain when to use each.",
            
            "troubleshooting": "Show the corrected code first, then explain why it was broken and how the fix works.",
            
            "conceptual": "Explain what it is in 2-3 sentences, then provide a minimal code example showing the concept."
        }
        
        # Handle empty/irrelevant context - NO HALLUCINATION
        if not context or context.strip() == "" or len(context) < 50:
            return {
                **state,
                "response": f"I couldn't retrieve relevant documentation for '{state['query']}'. This might mean:\n- The topic isn't covered in the current documentation\n- The query is too vague or ambiguous\n- It's outside LangGraph/LangChain scope\n\nTry asking about: StateGraph basics, adding nodes/edges, persistence, checkpointers, or message handling."
            }
        
        # Check if context is actually relevant to query (relaxed check)
        query_lower = state['query'].lower()
        context_lower = context.lower()
        
        # Extract key terms from query (ignore common words)
        query_words = [w for w in query_lower.split() if len(w) > 4 and w not in ['what', 'how', 'who', 'when', 'where', 'which', 'that', 'this', 'with', 'from', 'does', 'should']]
        
        # Check if ANY query terms appear in context (need at least 20% match)
        if query_words and len(query_words) > 2:
            relevance_score = sum(1 for word in query_words if word in context_lower)
            if relevance_score == 0 or (relevance_score / len(query_words)) < 0.2:
                # Context has insufficient relevant terms
                return {
                    **state,
                    "response": f"I couldn't find relevant information about '{state['query']}' in the documentation. This question might be outside my scope. I specialize in LangGraph and LangChain topics like: StateGraph, nodes, edges, persistence, and checkpointers."
                }
        
        # Build final prompt
        system_prompt = base_system_prompt.format(
            mode=self.mode,
            context=context
        )
        type_prompt = type_prompts.get(query_type, type_prompts["conceptual"])
        user_prompt = f"{type_prompt}\n\nQ: {state['query']}\n\nA:"
        
        if hasattr(self, 'llm') and self.llm is not None:
            try:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                if self.mode == "online":
                    # Use native Google SDK
                    response = self.llm.generate_content(full_prompt)
                    answer = response.text
                else:  # offline
                    answer = self.llm.invoke(full_prompt)
                
                # Brevity gate: cap at 40 lines (code-first approach)
                lines = answer.split('\n')
                if len(lines) > 40:
                    answer = '\n'.join(lines[:40])
                    
            except Exception as e:
                # Graceful degradation - extract key info from context
                answer = f"""Based on the documentation:

            {context[:500]}

Note: Online LLM unavailable. Please check your GOOGLE_API_KEY in .env file."""
        else:
            # No LLM available - show relevant context directly
            answer = f"""From the documentation:

            {context[:700]}

Note: For a more detailed explanation with code examples, ensure Ollama is running (`./start_ollama.sh`)."""
        
        return {
            **state,
            "response": answer
        }
    
    def run(self, query: str) -> dict:
        """Run the agent workflow"""
        
        initial_state = {
            "messages": [],
            "query": query,
            "query_type": "",
            "context": "",
            "response": "",
            "mode": self.mode
        }
        
        result = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "query_type": result.get("query_type", ""),
            "context": result.get("context", ""),
            "response": result.get("response", ""),
            "mode": self.mode
        }
