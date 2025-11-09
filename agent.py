import warnings
warnings.filterwarnings('ignore')

import shutil
from pathlib import Path
from typing import Annotated, Sequence, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

try:
    from langchain import hub
    HAS_HUB = True
except:
    HAS_HUB = False


# ============================================================
# STATE
# ============================================================

class AgentState(dict):
    """Agent state with messages"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================
# AGENT
# ============================================================

class Agent:
    """ Agentic RAG """
    
    def __init__(self, mode="offline", google_api_key=None, tavily_api_key=None, verbose=False):
        """Initialize agent (minimal setup)"""
        self.mode = mode
        self.tavily_api_key = tavily_api_key
        self.verbose = verbose
        self.data_dir = Path("data")
        self.chroma_dir = self.data_dir / "processed" / "chroma_db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Embeddings (only needed for offline mode with vector store)
        if mode == "offline":
            from langchain_community.embeddings import HuggingFaceEmbeddings
            if self.verbose:
                print("Loading embeddings...")
            self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            if self.verbose:
                print("✓ Ready")
        else:
            self.embeddings = None  # Not needed for online mode
        

        # Optional: Use Google's Gemini if API key is provided
        # if google_api_key:
        #     from langchain_google_genai import ChatGoogleGenerativeAI
        #     self.llm = ChatGoogleGenerativeAI(
        #         model="gemini-1.5-flash",
        #         google_api_key=google_api_key,
        #         temperature=0.3
        #     )
        #     if self.verbose:
        #         print("Using Google Gemini")
        # else:
        from langchain_community.chat_models import ChatOllama
        self.llm = ChatOllama(model="llama3.2:latest", temperature=0.3)
        
        self.vectorstore = None
        self.graph = self.build_graph()
    
    # ============================================================
    # SETUP 
    # ============================================================
    
    def pull_docs(self, verbose=True):
        """Download documentation (official pattern)"""
        urls = [
            "https://langchain-ai.github.io/langgraph/llms.txt",
            "https://langchain-ai.github.io/langgraph/llms-full.txt",
            "https://python.langchain.com/llms.txt"
        ]
        
        if verbose:
            print("Loading docs...")
        
        # Load docs (official pattern)
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]
        
        if verbose:
            print(f" Loaded {len(docs_list)} documents")
        
        return docs_list
    
    def build_index(self, docs_list=None, verbose=True):
        """Build Chroma vector store (official pattern)"""
        if docs_list is None:
            docs_list = self.pull_docs(verbose=False)
        
        if verbose:
            print("Building vector store...")
        
        # Split docs (official pattern: tiktoken with smaller chunks)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        
        if verbose:
            print(f"  ✓ Split into {len(doc_splits)} chunks")
        
        # Build Chroma (official pattern)
        if self.chroma_dir.exists():
            shutil.rmtree(self.chroma_dir)
        
        self.vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=self.embeddings,
            persist_directory=str(self.chroma_dir)
        )
        
        if verbose:
            print(f" Vector store ready ({len(doc_splits)} docs)")
        return True
    
    def load_vectorstore(self):
        """Load Chroma """
        if self.vectorstore is None:
            if not self.chroma_dir.exists():
                raise FileNotFoundError("Run agent.build_index() first")
            
            self.vectorstore = Chroma(
                collection_name="rag-chroma",
                embedding_function=self.embeddings,
                persist_directory=str(self.chroma_dir)
            )
        return self.vectorstore
    
    def refresh_index(self, force=False, max_age_hours=24):
        """Refresh index if stale or forced"""
        import time
        metadata_file = self.chroma_dir / "metadata.json"
        
        # Check if refresh needed
        if not force and metadata_file.exists():
            import json
            with open(metadata_file) as f:
                age = (time.time() - json.load(f)['timestamp']) / 3600
                if age < max_age_hours:
                    print(f"✓ Index fresh ({age:.1f}h old)")
                    return False
        
        # Rebuild index
        print("Refreshing documentation index...")
        docs = self.pull_docs(verbose=False)
        self.build_index(docs_list=docs, verbose=True)
        
        # Save timestamp
        import json
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump({'timestamp': time.time()}, f)
        
        return True
    
    # ============================================================
    # RETRIEVER TOOL
    # ============================================================
    
    def create_retriever_tool(self):
        """Create retriever tool """
        # Lazy load: don't load vectorstore until tool is actually called
        def get_retriever():
            vectorstore = self.load_vectorstore()
            return vectorstore.as_retriever()
        
        @tool
        def retrieve_langraph_docs(query: str) -> str:
            """Search and return information about LangGraph and LangChain documentation. Use for questions about concepts, APIs, and usage patterns."""
            retriever = get_retriever()
            docs = retriever.invoke(query)
            return "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        return retrieve_langraph_docs
    
    # ============================================================
    # GRAPH NODES 
    # ============================================================
    
    def web_search_node(self, state: AgentState):
        """Web search using Tavily """
        if self.verbose:
            print("---WEB SEARCHING---")
        
        question = state['messages'][0].content
        
        # Tavily search
        web_search_tool = TavilySearchResults(max_results=3, api_key=self.tavily_api_key)
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        
        return {"messages": [ToolMessage(content=web_results, tool_call_id="web_search")]}
    
    def agent_node(self, state: AgentState):
        """Agent node (placeholder for pattern consistency)"""
        if self.verbose:
            print("---AGENT---")
        # Just pass through - actual retrieval happens in retrieve_node
        return state
    
    def retrieve_node(self, state: AgentState):
        """Retrieve documents from vector store"""
        if self.verbose:
            print("---RETRIEVING DOCUMENTS---")
        
        question = state['messages'][0].content
        
        # Load vectorstore and retrieve
        vectorstore = self.load_vectorstore()
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke(question)
        
        # Format documents
        doc_content = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        return {"messages": [ToolMessage(content=doc_content, tool_call_id="retrieve")]}
    
    def evaluate_documents(self, state: AgentState):
        """
        LLM evaluates retrieved documents from vector store (offline mode).
        """
        if self.verbose:
            print("---EVALUATING DOCUMENTS---")
        
        last_message = state['messages'][-1]
        if not isinstance(last_message, ToolMessage):
            return state
        
        docs = last_message.content
        question = next((msg.content for msg in state['messages'] if isinstance(msg, HumanMessage)), None)
        
        if not question:
            return state
        
        # LLM-driven grading with full reasoning
        response = (
            ChatPromptTemplate.from_messages([
                ("system", """You are a grading assistant. Evaluate if the retrieved documentation contains information to answer the question.
                              Provide your assessment in this format:
                              DECISION: [yes/no]
                              REASONING: [brief explanation]"""),
                ("human", "Question: {question}\n\nDocuments:\n{docs}\n\nCan these documents answer the question?")
            ])
            | self.llm
            | StrOutputParser()
        ).invoke({"question": question, "docs": docs[:2000]})
        
        # LLM decides the next action
        response_lower = response.lower()
        if "decision: yes" in response_lower or "decision:yes" in response_lower:
            next_action = "generate"
            relevance = "relevant"
        else:
            next_action = "end"
            relevance = "not_relevant"
        
        # Extract LLM reasoning
        reasoning = response.split("REASONING:")[-1].strip() if "REASONING:" in response else response
        
        if self.verbose:
            print(f"  → Decision: {relevance}")
            print(f"  → LLM Reasoning: {reasoning[:150]}...")
        
        # Store LLM's decision in state
        return {"messages": [AIMessage(content="", additional_kwargs={
            "llm_decision": next_action,
            "relevance": relevance,
            "reasoning": reasoning
        })]}
    
    def evaluate_web_search(self, state: AgentState):
        """
        LLM evaluates web search results (online mode).
        """
        if self.verbose:
            print("---EVALUATING WEB SEARCH RESULTS---")
        
        last_message = state['messages'][-1]
        if not isinstance(last_message, ToolMessage):
            return state
        
        web_results = last_message.content
        question = next((msg.content for msg in state['messages'] if isinstance(msg, HumanMessage)), None)
        
        if not question:
            return state
        
        # LLM-driven grading with full reasoning
        response = (
            ChatPromptTemplate.from_messages([
                ("system", """You are a grading assistant. Evaluate if the web search results contain information to answer the question.
                               Provide your assessment in this format:
                               DECISION: [yes/no]
                               REASONING: [brief explanation]"""),
                ("human", "Question: {question}\n\nWeb search results:\n{results}\n\nCan these results answer the question?")
            ])
            | self.llm
            | StrOutputParser()
        ).invoke({"question": question, "results": web_results[:2000]})
        
        # LLM decides the next action
        response_lower = response.lower()
        if "decision: yes" in response_lower or "decision:yes" in response_lower:
            next_action = "generate"
            relevance = "relevant"
        else:
            next_action = "end"
            relevance = "not_relevant"
        
        # Extract LLM reasoning
        reasoning = response.split("REASONING:")[-1].strip() if "REASONING:" in response else response
        
        if self.verbose:
            print(f"  → Decision: {relevance}")
            print(f"  → LLM Reasoning: {reasoning[:150]}...")
        
        # Store LLM's decision in state
        return {"messages": [AIMessage(content="", additional_kwargs={
            "llm_decision": next_action,
            "relevance": relevance,
            "reasoning": reasoning
        })]}
    
    def generate_node(self, state: AgentState):
        """Generate answer"""
        if self.verbose:
            print("---GENERATING ANSWER---")
        
        
        context = "\n\n".join([msg.content for msg in state['messages'] if isinstance(msg, ToolMessage)])
        question = next((msg.content for msg in reversed(state['messages']) if isinstance(msg, HumanMessage)), None)
        
        if HAS_HUB:
            try:
                if self.verbose:
                    print("  Using hub prompt (rlm/rag-prompt)")
                response = (hub.pull("rlm/rag-prompt") | self.llm | StrOutputParser()).invoke({"context": context, "question": question})
                return {"messages": [AIMessage(content=response, additional_kwargs={"documents": context, "question": question})]}
            except:
                pass
        

        response = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a LangGraph documentation assistant. Use provided docs to answer with code examples when relevant."),
                ("human", "Documentation:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ])
            | self.llm
            | StrOutputParser()
        ).invoke({"context": context[:4000], "question": question})
        
        return {"messages": [AIMessage(content=response, additional_kwargs={"documents": context, "question": question})]}
    
    def grade_generation(self, state: AgentState) -> Literal["useful", "not useful", "not supported"]:
        """
        Check if generation is grounded in documents and answers the question.
        
        Args:
            state: Current agent state
            
        Returns:
            Decision: "useful", "not useful", or "not supported"
        """
        print("---CHECK HALLUCINATIONS---")
        
        last_message = state['messages'][-1]
        if not isinstance(last_message, AIMessage):
            return "useful"
        
        generation = last_message.content
        documents = last_message.additional_kwargs.get("documents", "")
        question = last_message.additional_kwargs.get("question", "")
        
        if not documents or not question:
            return "useful"
        
        # Hallucination grader
        class HallucinationGrade(BaseModel):
            """Binary score for hallucination check"""
            binary_score: str = Field(description="Answer is grounded in facts, 'yes' or 'no'")
        
        llm_with_hallucination_grader = self.llm.with_structured_output(HallucinationGrade)
        
        hallucination_score = (
            ChatPromptTemplate.from_messages([
                ("system", "You are a grader assessing whether an LLM generation is grounded in retrieved documents."),
                ("human", "Documents:\n\n{documents}\n\nLLM generation: {generation}\n\nGive a binary score 'yes' or 'no' to indicate if the generation is grounded in the documents.")
            ])
            | llm_with_hallucination_grader
        ).invoke({"documents": documents[:1000], "generation": generation})
        
        # Check hallucination
        if hallucination_score.binary_score == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            
            # Answer quality grader
            class AnswerGrade(BaseModel):
                """Binary score for answer quality"""
                binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")
            
            llm_with_answer_grader = self.llm.with_structured_output(AnswerGrade)
            
            print("---GRADING GENERATION vs QUESTION---")
            answer_score = (
                ChatPromptTemplate.from_messages([
                    ("system", "You are a grader assessing whether an answer addresses a question."),
                    ("human", "Question: {question}\n\nLLM generation: {generation}\n\nGive a binary score 'yes' or 'no' to indicate if the answer addresses the question.")
                ])
                | llm_with_answer_grader
            ).invoke({"question": question, "generation": generation})
            
            if answer_score.binary_score == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not supported"
    
    # ============================================================
    # GRAPH CONSTRUCTION
    # ============================================================
    
    def build_graph(self):
        """
        Build agentic RAG graph with LLM-controlled flow
        
        Offline mode: agent → retrieve → evaluate_documents → generate
        Online mode:  agent → web_search → evaluate_web_search → generate
        
        Both modes follow the pattern:
        Agent node (pass-through for consistency)
        Retrieve/search node
        LLM evaluation node
        Generate answer (if relevant)
        """
        # Create tool
        retriever_tool = self.create_retriever_tool()
        tools = [retriever_tool]
        
        # Bind tools to LLM (graceful fallback for Ollama)
        try:
            self.llm_with_tools = self.llm.bind_tools(tools)
        except NotImplementedError:
            self.llm_with_tools = self.llm  # Ollama doesn't support tool binding
        
        # Build graph
        workflow = StateGraph(AgentState)
        
        if self.mode == "online":
            # Online mode: agent → web_search → evaluate_web_search → generate
            workflow.add_node("agent", self.agent_node)
            workflow.add_node("web_search", self.web_search_node)
            workflow.add_node("evaluate_web_search", self.evaluate_web_search)
            workflow.add_node("generate", self.generate_node)
            
            workflow.set_entry_point("agent")
            workflow.add_edge("agent", "web_search")
            workflow.add_edge("web_search", "evaluate_web_search")
            
            # LLM decides if web results are good enough
            workflow.add_conditional_edges(
                "evaluate_web_search",
                lambda state: "generate" if state['messages'][-1].additional_kwargs.get('llm_decision', 'end') == "generate" else END
            )
            workflow.add_edge("generate", END)
        else:
            # Offline mode: agent → retrieve → evaluate_documents → generate
            workflow.add_node("agent", self.agent_node)
            workflow.add_node("retrieve", self.retrieve_node)
            workflow.add_node("evaluate_documents", self.evaluate_documents)
            workflow.add_node("generate", self.generate_node)
            
            workflow.set_entry_point("agent")
            
            # Always retrieve (keeps agent node for pattern consistency)
            workflow.add_edge("agent", "retrieve")
            workflow.add_edge("retrieve", "evaluate_documents")
            
            # LLM decides: generate or end
            workflow.add_conditional_edges(
                "evaluate_documents",
                lambda state: "generate" if state['messages'][-1].additional_kwargs.get('llm_decision', 'end') == "generate" else END
            )
            workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    # ============================================================
    # RUN
    # ============================================================
    
    def run(self, query: str) -> dict:
        """Run agentic RAG"""
        result = self.graph.invoke({"messages": [HumanMessage(content=query)]})
        response = result['messages'][-1].content if isinstance(result['messages'][-1], AIMessage) else str(result['messages'][-1])
        return {"query": query, "response": response, "mode": self.mode}
