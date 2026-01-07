import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
import traceback

# RAG imports
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ===============================
# CONSOLE LOGGER
# ===============================
def log_console(message, level="INFO"):
    """Enhanced console logging with timestamps"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️",
        "DEBUG": "🔍",
        "API": "🌐",
        "RAG": "📚",
        "USER": "👤",
        "BOT": "🤖"
    }.get(level, "📝")
    
    print(f"\n{prefix} [{timestamp}] {level}: {message}")
    print("=" * 80)

# ===============================
# ENV SETUP
# ===============================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

log_console("Starting Waste Management & Recycling Process Explainer Bot", "INFO")
log_console(f"API Key Status: {'Found ✓' if API_KEY else 'Not Found ✗'}", "API")

if not API_KEY:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    log_console("CRITICAL: API Key not found in environment variables", "ERROR")
    st.stop()

log_console(f"API Key Loaded: {API_KEY[:20]}...", "SUCCESS")

try:
    client = genai.Client(api_key=API_KEY)
    log_console("Gemini Client initialized successfully", "SUCCESS")
except Exception as e:
    log_console(f"Failed to initialize Gemini client: {e}", "ERROR")
    st.error(f"Client initialization failed: {e}")
    st.stop()

# Available models - prioritize your custom model
AVAILABLE_MODELS = [
    "gemini-3-flash-preview",  # Your custom model from AI Studio
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash"
]
log_console(f"Available Models: {', '.join(AVAILABLE_MODELS)}", "INFO")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="♻️ Waste Management & Recycling Explainer",
    page_icon="♻️",
    layout="wide"
)

# ===============================
# RAG HELPER FUNCTIONS
# ===============================
def extract_text_from_pdfs(pdf_docs):
    """Extract text from uploaded PDF files"""
    log_console(f"Starting PDF text extraction for {len(pdf_docs)} files", "RAG")
    text = ""
    
    for idx, pdf in enumerate(pdf_docs, 1):
        try:
            log_console(f"Processing PDF {idx}/{len(pdf_docs)}: {pdf.name}", "RAG")
            pdf_reader = PdfReader(pdf)
            page_count = len(pdf_reader.pages)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                text += page_text
                log_console(f"  └─ Page {page_num}/{page_count}: Extracted {len(page_text)} characters", "DEBUG")
            
            log_console(f"✓ Successfully extracted text from {pdf.name}", "SUCCESS")
            
        except Exception as e:
            log_console(f"✗ Error reading {pdf.name}: {e}", "ERROR")
            st.error(f"Error reading {pdf.name}: {e}")
    
    log_console(f"Total text extracted: {len(text)} characters from {len(pdf_docs)} files", "SUCCESS")
    return text

def create_text_chunks(text):
    """Split text into chunks for embedding"""
    log_console("Creating text chunks for embedding...", "RAG")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    
    log_console(f"Created {len(chunks)} text chunks (size=1000, overlap=200)", "SUCCESS")
    
    if chunks:
        sample = chunks[0][:100] + "..." if len(chunks[0]) > 100 else chunks[0]
        log_console(f"Sample chunk preview: {sample}", "DEBUG")
    
    return chunks

def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks with batch processing"""
    log_console("Initializing vector store creation...", "RAG")
    
    try:
        log_console("Creating embeddings using Google Generative AI...", "API")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        
        batch_size = 10
        total_chunks = len(text_chunks)
        
        log_console(f"Processing {total_chunks} chunks in batches of {batch_size}...", "RAG")
        
        all_embeddings = []
        for i in range(0, total_chunks, batch_size):
            batch = text_chunks[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size
            
            log_console(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...", "RAG")
            
            try:
                if i > 0:
                    time.sleep(2)
                
                batch_embeddings = embeddings.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                log_console(f"✓ Batch {batch_num}/{total_batches} completed", "SUCCESS")
                
            except Exception as batch_error:
                error_str = str(batch_error)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    log_console(f"⚠ Rate limit hit at batch {batch_num}, waiting 60 seconds...", "WARNING")
                    st.warning(f"Rate limit reached. Waiting 60 seconds... (Batch {batch_num}/{total_batches})")
                    time.sleep(60)
                    
                    batch_embeddings = embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    log_console(f"✓ Batch {batch_num} completed after retry", "SUCCESS")
                else:
                    raise batch_error
        
        log_console(f"All embeddings generated successfully", "SUCCESS")
        
        log_console("Building FAISS index...", "RAG")
        documents = [Document(page_content=text) for text in text_chunks]
        vector_store = FAISS.from_documents(documents, embeddings)
        
        log_console("Saving vector store to disk (faiss_index)...", "RAG")
        vector_store.save_local("faiss_index")
        
        log_console("✓ Vector store created and saved successfully", "SUCCESS")
        return vector_store
        
    except Exception as e:
        error_str = str(e)
        log_console(f"✗ Error creating vector store: {e}", "ERROR")
        traceback.print_exc()
        
        if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
            st.error("⏰ **Embedding Rate Limit Exceeded!**")
            st.info("""
            **Solutions:**
            1. ⏳ Wait 1 hour and try again
            2. 📊 Check usage at: https://ai.dev/usage?tab=rate-limit
            3. 💡 Try uploading smaller PDFs
            4. 🔄 Use the bot without RAG (it will still work!)
            """)
        else:
            st.error(f"Error creating vector store: {e}")
        
        return None

def load_vector_store():
    """Load existing FAISS vector store"""
    log_console("Attempting to load existing vector store...", "RAG")
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=API_KEY
        )
        vector_store = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
        log_console("✓ Vector store loaded successfully", "SUCCESS")
        return vector_store
        
    except Exception as e:
        log_console(f"✗ Could not load vector store: {e}", "WARNING")
        return None

def search_relevant_docs(query, vector_store, top_k=3):
    """Search for relevant document chunks"""
    log_console(f"Searching for relevant documents (top_k={top_k})", "RAG")
    log_console(f"Query: '{query}'", "DEBUG")
    
    try:
        docs = vector_store.similarity_search(query, k=top_k)
        log_console(f"Found {len(docs)} relevant document chunks", "SUCCESS")
        
        for idx, doc in enumerate(docs, 1):
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            log_console(f"  Chunk {idx}: {preview}", "DEBUG")
        
        return docs
        
    except Exception as e:
        log_console(f"✗ Search error: {e}", "ERROR")
        st.error(f"Search error: {e}")
        return []

# ===============================
# SYSTEM PROMPTS
# ===============================
BASE_SYSTEM_PROMPT = """
You are a Waste Management & Recycling Process Explainer Bot for environmental services.

Your role:
1. Explain waste segregation rules in simple, clear language
2. Clarify waste collection and recycling processes
3. Answer questions about composting, hazardous waste, and disposal
4. Help citizens and organizations understand proper waste handling
5. Provide information about environmental compliance

IMPORTANT RESTRICTIONS:
- You ONLY explain and educate - NEVER schedule pickups or impose penalties
- You NEVER make operational decisions about waste collection
- You NEVER replace municipal authorities or waste management services
- Always encourage users to contact local waste management services for scheduling
- If asked to schedule or enforce actions, politely decline and redirect to proper authorities
- Focus on public awareness and environmental education
"""

RAG_SYSTEM_PROMPT = """
You are a Waste Management & Recycling Process Explainer with access to uploaded waste management documents.

Based on the provided context from the documents, answer the user's question clearly and accurately.

RULES:
- Use ONLY information from the provided context
- If the answer is not in the context, say "This information is not available in the uploaded documents"
- Keep explanations simple and practical for citizens and organizations
- NEVER schedule pickups or impose penalties
- Always remind users to contact local authorities for service requests
- Focus on education and awareness

Context from documents:
{context}

Question: {question}
"""

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    log_console("Initialized new chat history", "INFO")

if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = 0

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ===============================
# UI
# ===============================
st.title("♻️ Waste Management & Recycling Process Explainer Bot")
st.caption("AI Assistant for Understanding Waste Segregation, Collection & Recycling Workflows")

# Status indicators
col1, col2, col3 = st.columns(3)
with col1:
    if API_KEY:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")

with col2:
    if st.session_state.pdf_processed:
        st.info(f"📚 {len(st.session_state.uploaded_files)} Documents Loaded")
    else:
        st.warning("📭 No Documents Uploaded")

with col3:
    if st.session_state.rag_enabled:
        st.success("🔍 RAG Enabled")
    else:
        st.info("💬 Chat Mode")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.header("📁 Upload Waste Management Documents")
    
    pdf_docs = st.file_uploader(
        "Upload waste management guidelines, recycling rules (optional)",
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if st.button("🔄 Process Documents"):
        if pdf_docs:
            log_console("=" * 80, "INFO")
            log_console("USER ACTION: Process Documents Button Clicked", "USER")
            log_console(f"Number of files: {len(pdf_docs)}", "INFO")
            
            for idx, pdf in enumerate(pdf_docs, 1):
                file_size = pdf.size / 1024
                log_console(f"File {idx}: {pdf.name} ({file_size:.2f} KB)", "INFO")
            
            with st.spinner("Processing PDFs..."):
                raw_text = extract_text_from_pdfs(pdf_docs)
                
                if raw_text:
                    text_chunks = create_text_chunks(raw_text)
                    st.info(f"Created {len(text_chunks)} text chunks")
                    
                    vector_store = create_vector_store(text_chunks)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.rag_enabled = True
                        st.session_state.pdf_processed = True
                        st.session_state.uploaded_files = [pdf.name for pdf in pdf_docs]
                        
                        log_console(f"✓ Successfully processed {len(pdf_docs)} documents", "SUCCESS")
                        log_console(f"Files: {', '.join(st.session_state.uploaded_files)}", "INFO")
                        st.success(f"✅ Processed {len(pdf_docs)} documents successfully!")
                    else:
                        log_console("✗ Failed to create vector store", "ERROR")
                        st.error("Failed to create vector store")
                else:
                    log_console("✗ No text extracted from PDFs", "ERROR")
                    st.error("No text extracted from PDFs")
        else:
            log_console("⚠ No PDF files selected for processing", "WARNING")
            st.warning("Please upload at least one PDF document")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.subheader("📄 Uploaded Files:")
        for idx, filename in enumerate(st.session_state.uploaded_files, 1):
            st.caption(f"{idx}. {filename}")
    
    # RAG toggle
    if st.session_state.pdf_processed:
        rag_toggle = st.checkbox(
            "🔍 Use RAG (Search uploaded docs)",
            value=st.session_state.rag_enabled
        )
        
        if rag_toggle != st.session_state.rag_enabled:
            st.session_state.rag_enabled = rag_toggle
            log_console(f"RAG Mode: {'ENABLED' if rag_toggle else 'DISABLED'}", "INFO")
    
    st.markdown("---")
    st.header("📝 Example Questions")
    
    examples = [
        "Explain waste segregation rules",
        "What happens to recyclable waste?",
        "Explain composting process",
        "What are hazardous wastes?",
        "How is waste collected and sorted?"
    ]
    
    for q in examples:
        if st.sidebar.button(q, key=f"ex_{q}"):
            st.session_state.user_question = q
            log_console(f"Example question selected: '{q}'", "USER")
    
    if st.button("🗑️ Clear Chat"):
        log_console("Chat history cleared by user", "INFO")
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.caption(f"🔑 API: {API_KEY[:15]}...")
    st.caption(f"🤖 RAG: {'ON' if st.session_state.rag_enabled else 'OFF'}")
    st.caption(f"💬 Messages: {len(st.session_state.chat_history)}")
    
    # Project Info
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.caption("**Domain:** Waste Management / Environmental Services")
    st.caption("**Team Size:** 4-5 Members")
    st.caption("**Tech Stack:** Google AI Studio API, Gemini Flash, Python, Streamlit")

# ===============================
# GEMINI FUNCTION WITH STREAMING
# ===============================
def ask_gemini(question: str, use_rag: bool = False, placeholder=None) -> str:
    """Query Gemini with optional RAG support and streaming"""
    log_console("=" * 80, "API")
    log_console("GEMINI API CALL INITIATED", "API")
    log_console(f"Mode: {'RAG (Document Search)' if use_rag else 'Standard Chat'}", "INFO")
    log_console(f"Question: '{question}'", "USER")
    
    # Prepare prompt
    if use_rag and st.session_state.vector_store:
        log_console("Using RAG mode - searching documents...", "RAG")
        
        relevant_docs = search_relevant_docs(question, st.session_state.vector_store)
        
        if relevant_docs:
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            prompt = RAG_SYSTEM_PROMPT.format(context=context, question=question)
            log_console(f"Context length: {len(context)} characters", "DEBUG")
        else:
            log_console("No relevant documents found, falling back to base prompt", "WARNING")
            prompt = f"{BASE_SYSTEM_PROMPT}\n\nQuestion: {question}"
    else:
        log_console("Using standard chat mode (no RAG)", "INFO")
        prompt = f"{BASE_SYSTEM_PROMPT}\n\nQuestion: {question}"
    
    # Create content
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)]
        )
    ]
    
    config = types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=1024,
        thinking_config=types.ThinkingConfig(
            thinking_level="MINIMAL",
        ),
        media_resolution="MEDIA_RESOLUTION_LOW"
    )
    
    log_console("API Config: temperature=0.4, max_tokens=1024, thinking=MINIMAL", "DEBUG")
    
    response_text = ""
    successful_model = None
    
    # Try each model
    for model_idx, model_name in enumerate(AVAILABLE_MODELS, 1):
        log_console(f"Trying model {model_idx}/{len(AVAILABLE_MODELS)}: {model_name}", "API")
        
        try:
            chunk_count = 0
            
            # Try with and without 'models/' prefix
            for prefix in ["", "models/"]:
                full_model_name = f"{prefix}{model_name}"
                
                try:
                    log_console(f"  Attempting: {full_model_name}", "DEBUG")
                    
                    for chunk in client.models.generate_content_stream(
                        model=full_model_name,
                        contents=contents,
                        config=config,
                    ):
                        chunk_count += 1
                        if chunk.candidates:
                            for part in chunk.candidates[0].content.parts:
                                if part.text:
                                    response_text += part.text
                                    log_console(f"  Chunk {chunk_count}: {len(part.text)} chars", "DEBUG")
                                    
                                    # UPDATE PLACEHOLDER WITH STREAMING TEXT (CURSOR EFFECT)
                                    if placeholder:
                                        placeholder.markdown(response_text + "▌")
                    
                    if response_text:
                        successful_model = full_model_name
                        log_console(f"✓ SUCCESS with {full_model_name}", "SUCCESS")
                        log_console(f"Total chunks received: {chunk_count}", "DEBUG")
                        log_console(f"Response length: {len(response_text)} characters", "DEBUG")
                        
                        # FINAL UPDATE WITHOUT CURSOR
                        if placeholder:
                            placeholder.markdown(response_text)
                        break
                
                except Exception as prefix_error:
                    error_str = str(prefix_error)
                    if "404" not in error_str and "NOT_FOUND" not in error_str:
                        raise prefix_error
                    continue
            
            if response_text:
                break
                
        except Exception as e:
            error_str = str(e)
            
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                log_console(f"✗ Model {model_name} rate limited, trying next...", "WARNING")
                continue
            elif "404" in error_str or "NOT_FOUND" in error_str:
                log_console(f"✗ Model {model_name} not found, trying next...", "WARNING")
                continue
            else:
                log_console(f"✗ Error with {model_name}: {e}", "ERROR")
                continue
    
    if not response_text:
        log_console("✗ All models failed to generate response", "ERROR")
    else:
        preview = response_text[:200] + "..." if len(response_text) > 200 else response_text
        log_console(f"Response preview: {preview}", "BOT")
        log_console(f"Model used: {successful_model}", "SUCCESS")
    
    log_console("=" * 80, "API")
    
    return response_text.strip()

# ===============================
# CHAT DISPLAY
# ===============================
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "source" in msg:
            st.caption(f"📚 Source: {msg['source']}")
        if "timestamp" in msg:
            st.caption(f"🕐 {msg['timestamp']}")

# ===============================
# USER INPUT
# ===============================
user_input = st.chat_input("Ask about waste segregation, recycling, or disposal processes...")

if "user_question" in st.session_state:
    user_input = st.session_state.user_question
    del st.session_state.user_question

# ===============================
# HANDLE MESSAGE WITH STREAMING
# ===============================
if user_input:
    now = time.time()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    log_console("=" * 80, "USER")
    log_console("NEW USER MESSAGE RECEIVED", "USER")
    log_console(f"Message: '{user_input}'", "USER")
    log_console(f"Timestamp: {timestamp}", "INFO")
    
    # Rate limiting
    time_since_last = now - st.session_state.last_call_time
    if time_since_last < 4:
        wait_time = 4 - time_since_last
        log_console(f"⚠ Rate limit active - {wait_time:.1f}s remaining", "WARNING")
        st.warning(f"⏳ Please wait {wait_time:.1f} more seconds before asking again")
        st.stop()
    
    st.session_state.last_call_time = now
    
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })
    
    # Display user message immediately
    with st.chat_message("user"):
        st.write(user_input)
        st.caption(f"🕐 {timestamp}")
    
    # Create assistant message container with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            answer = ask_gemini(user_input, use_rag=st.session_state.rag_enabled, placeholder=message_placeholder)
            
            if not answer:
                answer = "⚠️ No response received. Please try again."
                log_console("⚠ Empty response from Gemini", "WARNING")
                message_placeholder.markdown(answer)
            
            # Add bot response to history
            response_data = {
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            
            if st.session_state.rag_enabled:
                response_data["source"] = "RAG (Document Search)"
                response_data["files"] = ", ".join(st.session_state.uploaded_files)
                st.caption(f"📚 Source: RAG (Document Search)")
            
            st.caption(f"🕐 {response_data['timestamp']}")
            
            st.session_state.chat_history.append(response_data)
            
            log_console("✓ Response added to chat history", "SUCCESS")
            log_console(f"Chat history length: {len(st.session_state.chat_history)} messages", "INFO")
            
        except Exception as e:
            error_msg = str(e)
            log_console(f"✗ EXCEPTION DURING MESSAGE HANDLING", "ERROR")
            log_console(f"Error: {error_msg}", "ERROR")
            traceback.print_exc()
            
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                message_placeholder.error("⏰ Rate limit exceeded. Please wait 30 seconds and try again.")
            else:
                message_placeholder.error(f"❌ Error: {error_msg}")
    
    st.rerun()

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
<strong>⚠️ Disclaimer:</strong> This bot provides informational guidance only about waste management and recycling processes.
<br>For waste collection scheduling or enforcement actions, please contact your local municipal waste management services.
<br>Always follow your region's official waste management guidelines and environmental compliance rules.
</div>
""", unsafe_allow_html=True)

log_console("Application ready for user interaction", "INFO")