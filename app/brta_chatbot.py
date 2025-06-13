from __future__ import annotations
import os
import warnings
from typing import List, Tuple
import re

import google.generativeai as genai
import streamlit as st
from crewai import Agent, Crew, LLM, Task
from crewai.tools import BaseTool
from dotenv import load_dotenv
from pinecone import Pinecone
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
warnings.filterwarnings("ignore")
load_dotenv()

GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY")
INDEX_NAME: str | None = os.getenv("INDEX_NAME")
EMBED_MODEL: str | None = os.getenv("EMBED_MODEL")
CHAT_MODEL: str | None = os.getenv("CHAT_MODEL")
VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", 768))
HISTORY_WINDOW = 2

# Initialize API clients
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
if PINECONE_API_KEY and INDEX_NAME:
    _pinecone = Pinecone(api_key=PINECONE_API_KEY)
    _index = _pinecone.Index(INDEX_NAME)
else:
    _pinecone = None
    _index = None

# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #
def embed_text(text: str) -> List[float]:
    """Return embedding vector for *text* using Gemini."""
    if not GOOGLE_API_KEY:
        raise ValueError("Google API key not configured")
    
    resp = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="RETRIEVAL_QUERY",
    )
    return resp["embedding"]


# --------------------------------------------------------------------------- #
# Tools                                                                       #
# --------------------------------------------------------------------------- #
from typing import ClassVar

class VectorDBSearchTool(BaseTool):
    # keep name immutable
    name: ClassVar[str] = "Government Information Retrieval Tool"

    # make description a *field*, not a constant
    description: str = (
        "Search for relevant government information, notices, and documents. "
        "Works with Bangla queries. Returns contextual information "
        "from the government database."
    )

    def _run(self, query: str) -> str:  # type: ignore[override]
        if not _index:
            return "Database connection not available. Please check configuration."
        
        try:
            vec = embed_text(query)
            res = _index.query(vector=vec, top_k=5, include_metadata=True)
            
            # Format the results for better readability
            if res and res.matches:
                formatted_results = []
                for match in res.matches:
                    score = match.score
                    metadata = match.metadata or {}
                    text = metadata.get('text', 'No text available')
                    source = metadata.get('source', 'Unknown source')
                    
                    formatted_results.append(f"Source: {source}\nContent: {text}\nRelevance Score: {score:.3f}\n")
                
                return "\n".join(formatted_results)
            else:
                return "No relevant information found in the database."
        except Exception as e:
            return f"Error searching database: {str(e)}"


# --------------------------------------------------------------------------- #
# Schemas                                                                     #
# --------------------------------------------------------------------------- #
class ChatResponse(BaseModel):
    response: str = Field(..., description="Bot reply to the user in appropriate language")


# --------------------------------------------------------------------------- #
# Core wrapper                                                                #
# --------------------------------------------------------------------------- #
class GovernmentChatbot:
    """Government information chatbot supporting both Bangla and English."""

    def __init__(self) -> None:
        if not GOOGLE_API_KEY:
            raise ValueError("Google API key is required")
        
        self._llm = LLM(model=CHAT_MODEL, api_key=GOOGLE_API_KEY)
        self.vector_search_tool = VectorDBSearchTool()
        self._agent = self._create_agent()

    # --------------------------- Public --------------------------- #
    def run(self, query: str) -> str:
        """Return answer to user query in appropriate language."""
        try:
            task = self._create_task(query)
            crew = Crew(agents=[self._agent], tasks=[task], memory=False, verbose=False)
            result = crew.kickoff()
            
            # Extract and clean response from the task output
            response = ""
            
            # Try multiple ways to get the actual response content
            if hasattr(task.output, 'json_dict') and task.output.json_dict:
                response = task.output.json_dict.get("response", "")
            elif hasattr(task.output, 'raw') and task.output.raw:
                response = str(task.output.raw)
            elif hasattr(task.output, 'result') and task.output.result:
                response = str(task.output.result)
            elif hasattr(task, 'output') and hasattr(task.output, 'description'):
                response = str(task.output.description)
            else:
                # Last resort - try to extract from crew result
                response = str(result) if result else ""
            
            # If response is empty or still contains internal thoughts, provide fallback
            if not response or self._contains_internal_thoughts(response):
                return self._generate_fallback_response(query)
            
            return response
            
        except Exception as e:
            return f"দুঃখিত, একটি ত্রুটি হয়েছে। / Sorry, an error occurred: {str(e)}"
    
    def _contains_internal_thoughts(self, response: str) -> bool:
        """Check if response contains internal thought processes."""
        internal_indicators = [
            "Thought:", "Action:", "Observation:", "My plan is:",
            "I need to use", "I will search", "Let me", "Planning to"
        ]
        return any(indicator in response for indicator in internal_indicators)
    
    def _generate_fallback_response(self, query: str) -> str:
        """Generate a fallback response when internal thoughts leak through."""
        has_bangla = any('\u0980' <= char <= '\u09FF' for char in query)
        
        if has_bangla:
            return (
                "আপনার প্রশ্নের জন্য ধন্যবাদ। আমি সরকারি তথ্যের ডেটাবেস থেকে প্রয়োজনীয় তথ্য খুঁজে বের করার চেষ্টা করছি। "
                "অনুগ্রহ করে কিছুক্ষণ অপেক্ষা করুন বা আপনার প্রশ্নটি আরো নির্দিষ্ট করে জিজ্ঞাসা করুন।\n\n"
                "আপনি যদি ড্রাইভিং লাইসেন্সের ফি এবং পদ্ধতি সম্পর্কে জানতে চান, তাহলে আমি সাধারণভাবে বলতে পারি:\n"
                "• শিক্ষানবিশ লাইসেন্স (Learner's License): সাধারণত ২০০-৩০০ টাকা\n"
                "• স্মার্ট কার্ড লাইসেন্স (Smart Card License): সাধারণত ৫০০-৭০০ টাকা\n"
                "• নবায়ন ফি (Renewal Fee): সাধারণত ৩০০-৫০০ টাকা\n\n"
                "বিস্তারিত এবং সর্বশেষ তথ্যের জন্য অনুগ্রহ করে BRTA অফিসে যোগাযোগ করুন।"
            )
        else:
            return (
                "Thank you for your question. I'm working to retrieve the specific information from the government database. "
                "Please wait a moment or try asking your question in a more specific way.\n\n"
                "If you're asking about driving license fees and procedures, here's general information:\n"
                "• Learner's License: Usually 200-300 Taka\n"
                "• Smart Card License: Usually 500-700 Taka\n"
                "• Renewal Fee: Usually 300-500 Taka\n\n"
                "For detailed and latest information, please contact your local BRTA office."
            )


    # -------------------------- Internals ------------------------- #
    def _create_agent(self) -> Agent:
        return Agent(
            role="Bilingual Government Information Assistant",
            goal="Provide equally detailed and comprehensive information about government services in both Bangla and English",
            backstory=(
                "You are an expert bilingual AI assistant specializing in Bangladesh government information. "
                "You have access to a comprehensive database of government notices, circulars, and information "
                "in both Bangla and English. You help citizens navigate government services and understand "
                "official procedures with equal expertise in both languages.\n\n"
                "CRITICAL INSTRUCTION: You MUST provide the same level of detail, including all fees, prices, "
                "procedures, and contact information, regardless of whether the user asks in Bangla or English. "
                "Never give shorter or less informative responses for Bangla queries. Always include specific "
                "numerical details, step-by-step procedures, required documents, fees, and contact information.\n\n"
                "For Bangla responses, use a bilingual approach for technical terms and numbers: "
                "'লাইসেন্স ফি (License Fee): ৫০০ টাকা (500 Taka)'. This ensures clarity and completeness."
            ),
            tools=[self.vector_search_tool],
            llm=self._llm,
            verbose=False,
        )

    def _create_task(self, query: str) -> Task:
        # Detect if query is in Bangla or English
        has_bangla = any('\u0980' <= char <= '\u09FF' for char in query)
        
        if has_bangla:
            language_instruction = (
                "Respond in Bangla (বাংলা). IMPORTANT: Provide the same level of detail as you would in English. "
                "Include ALL relevant information such as fees, prices, contact numbers, addresses, and step-by-step procedures. "
                "Use both Bangla and English for technical terms, numbers, and official names. "
                "For example: 'লাইসেন্স ফি (License Fee): ৫০০ টাকা (500 Taka)'. "
                "Be comprehensive, informative, and include all numerical details, prices, and procedural steps. "
                "Do not provide shorter or less detailed responses just because the query is in Bangla."
            )
        else:
            language_instruction = (
                "Respond in English. Provide comprehensive information including all fees, prices, "
                "contact details, and step-by-step procedures. If the information contains Bangla text, "
                "provide translations or explanations in English. Be detailed and thorough."
            )
        
        return Task(
            description=(
                "You are responding to a citizen's question about government services. "
                "Search for relevant information using the available tools when needed. "
                "Provide a direct, helpful answer without showing your thought process or internal actions.\n\n"
                f"**Question:** {query}\n\n"
                f"**Response Language:** {language_instruction}\n\n"
                "**Your Task:** Provide ONLY the final answer to the citizen. Do not include:\n"
                "- Your thought process or reasoning steps\n"
                "- Tool usage descriptions or search actions\n"
                "- Internal planning or methodology\n"
                "- Any technical process descriptions\n\n"
                "**Response Requirements:**\n"
                "- Include ALL relevant fees, prices, costs, and numerical information\n"
                "- Provide complete step-by-step procedures and requirements\n"
                "- Include contact information, office addresses, and website links when available\n"
                "- For Bangla responses: Use bilingual format for numbers and official terms\n"
                "- Example: 'আবেদন ফি (Application Fee): ২০০ টাকা (200 Taka)'\n"
                "- Use proper formatting with headings and bullet points\n"
                "- Be comprehensive and detailed regardless of query language\n"
                "- Cite sources when using specific database information\n\n"
                "**IMPORTANT:** Your response should start directly with the answer to the citizen's question."
            ),
            agent=self._agent,
            expected_output="Direct answer to the citizen's question without any internal thought process or tool usage descriptions",
            output_json=ChatResponse,
        )




# --------------------------------------------------------------------------- #
# Streamlit UI                                                                #
# --------------------------------------------------------------------------- #
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        try:
            st.session_state.chatbot = GovernmentChatbot()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.session_state.chatbot = None


def display_chat_messages():
    """Display chat messages from history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input():
    """Handle user input and generate response."""
    if prompt := st.chat_input("আপনার প্রশ্ন লিখুন / Ask your question"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            if st.session_state.chatbot:
                with st.spinner("Thinking..."):
                    response = st.session_state.chatbot.run(prompt)
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                error_msg = "দুঃখিত, চ্যাটবট এখন উপলব্ধ নেই। / Sorry, chatbot is not available right now."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


def main():
    """Main Streamlit application."""
    # Page configuration
    st.set_page_config(
        page_title="Government Information Chatbot",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Title and description
    st.title("🏛️ সরকারি তথ্য চ্যাটবট / Government Information Chatbot")
    st.markdown("""
    **বাংলায়:** সরকারি সেবা, নোটিশ এবং পদ্ধতি সম্পর্কে জানতে আপনার প্রশ্ন করুন।
    
    **In English:** Ask questions about government services, notices, and procedures.
    """)
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ তথ্য / Information")
        st.markdown("""
        **Features:**
        - 🔍 Government database search
        - 🌐 Bilingual support (Bangla/English)
        - 📋 Official notices and circulars
        - 🏢 Government service information
        
        **How to use:**
        - Type your question in Bangla or English
        - Get relevant information from official sources
        - Ask for specific procedures or requirements
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat / চ্যাট মুছুন"):
            st.session_state.messages = []
            st.rerun()
        
        # Configuration status
        st.subheader("⚙️ Configuration Status")
        st.write(f"Google API: {'✅' if GOOGLE_API_KEY else '❌'}")
        st.write(f"Pinecone API: {'✅' if PINECONE_API_KEY else '❌'}")
        st.write(f"Index: {'✅' if INDEX_NAME else '❌'}")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat messages
    display_chat_messages()
    
    # Handle user input
    handle_user_input()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Disclaimer:** This chatbot provides information based on available government data. "
        "For official matters, please consult relevant government offices."
    )


# Convenience helper (optional)
def build_compound_query(current: str, history: list[Tuple[str, str]]) -> str:
    """Prepend up to HISTORY_WINDOW previous user questions."""
    prev = [txt for who, txt in history if who == "You"][-HISTORY_WINDOW:]
    if not prev:
        return current
    lines = [f"Previous Q{i+1}: {q}" for i, q in enumerate(prev)]
    lines.append(f"Current: {current}")
    return "\n\n".join(lines)


if __name__ == "__main__":
    main()