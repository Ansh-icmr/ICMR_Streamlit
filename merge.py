
import os
import json
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from langsmith import traceable

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings


import streamlit as st
import uuid


load_dotenv()

directory = "icmr_url_pdf"

def get_pdf_files_info():
    """Get information about all PDF files in the directory"""
    pdf_info = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                path = os.path.join(root, file)
                stat = os.stat(path)
                pdf_info[path] = {
                    'mtime': stat.st_mtime,
                    'size': stat.st_size
                }
    return pdf_info

def has_pdfs_changed():
    """Check if any PDFs have been added, modified, or removed"""
    if 'pdf_files_info' not in st.session_state:
        return True
    
    current_info = get_pdf_files_info()
    previous_info = st.session_state['pdf_files_info']
    
    # Check if files were added or removed
    if set(current_info.keys()) != set(previous_info.keys()):
        return True
    
    # Check if any file was modified
    for path, info in current_info.items():
        if path in previous_info:
            if info['mtime'] != previous_info[path]['mtime'] or info['size'] != previous_info[path]['size']:
                return True
    
    return False

def load_pdf_documents():
    """Load PDF documents and cache them in session state"""
    if 'cached_documents' not in st.session_state or has_pdfs_changed():
        # Create a DirectoryLoader to load all PDFs in the directory
        loader = DirectoryLoader(
            directory,
            glob="**/*.pdf",    
            show_progress=True,
            use_multithreading=True
        )
        
        # Load all PDF documents from the directory
        documents = loader.load()
        
        print(f"Loaded {len(documents)} PDF documents from {directory}")
        # List all loaded PDFs
        for doc in documents:
            print(f"Loaded: {doc.metadata['source']}")
        
        # Update cache
        st.session_state['cached_documents'] = documents
        st.session_state['pdf_files_info'] = get_pdf_files_info()
        return documents
    else:
        print("Using cached PDF documents")
        return st.session_state['cached_documents']

PDF_PATHS = load_pdf_documents()
INDEX_ROOT = Path(".indices")
INDEX_ROOT.mkdir(exist_ok=True)

# ----------------- helpers (traced) -----------------
@traceable(name="load_pdf")
def load_pdf(path: str):
    return PyPDFLoader(path).load()  

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vectorstore")
def build_vectorstore(splits, embed_model_name: str):
    emb = OllamaEmbeddings(model=embed_model_name)
    return FAISS.from_documents(splits, emb)

# ----------------- cache key / fingerprint -----------------
def _file_fingerprint(documents: list) -> dict:
    combined_hash = hashlib.sha256()
    total_size = 0
    latest_mtime = 0
    
    # Process each document's source file
    for doc in documents:
        path = doc.metadata['source']
        p = Path(path)
        # Update the hash with file contents
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                combined_hash.update(chunk)
        # Add to total size
        total_size += p.stat().st_size
        # Keep track of the latest modification time
        latest_mtime = max(latest_mtime, int(p.stat().st_mtime))
    
    return {
        "sha256": combined_hash.hexdigest(),
        "total_size": total_size,
        "latest_mtime": latest_mtime
    }

def _index_key(documents: list, chunk_size: int, chunk_overlap: int, embed_model_name: str) -> str:
    meta = {
        "pdf_fingerprint": _file_fingerprint(documents),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
        "format": "v1",
    }
    return hashlib.sha256(json.dumps(meta, sort_keys=True).encode("utf-8")).hexdigest()

# ----------------- explicitly traced load/build runs -----------------
@traceable(name="load_index", tags=["index"])
def load_index_run(index_dir: Path, embed_model_name: str):
    emb = OllamaEmbeddings(model=embed_model_name)
    return FAISS.load_local(
        str(index_dir),
        emb,
        allow_dangerous_deserialization=True
    )

@traceable(name="build_index", tags=["index"])
def build_index_run(pdf_path: list, index_dir: Path, chunk_size: int, chunk_overlap: int, embed_model_name: str):
    docs = pdf_path  # we already have the loaded documents
    splits = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # child
    vs = build_vectorstore(splits, embed_model_name)  # child
    index_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(index_dir))
    (index_dir / "meta.json").write_text(json.dumps({
        "pdf_path": os.path.abspath(pdf_path),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embed_model_name,
    }, indent=2))
    return vs

# ----------------- dispatcher (not traced) -----------------
def load_or_build_index(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "bge-m3:latest",
    force_rebuild: bool = False,
):
    key = _index_key(pdf_path, chunk_size, chunk_overlap, embed_model_name)
    index_dir = INDEX_ROOT / key
    cache_hit = index_dir.exists() and not force_rebuild
    if cache_hit:
        return load_index_run(index_dir, embed_model_name)
    else:
        return build_index_run(pdf_path, index_dir, chunk_size, chunk_overlap, embed_model_name)

# ----------------- model, prompt, and pipeline -----------------
llm = ChatOpenAI(base_url="http://127.0.0.1:11434/v1", api_key="Ollama", model="llama3.2:latest", temperature=0.8)

prompt = ChatPromptTemplate.from_messages([
    ("system", """Answer ONLY if the question is related to ICMR (Indian Council of Medical Research) and the information is explicitly found in the provided context.
      If the question is not related to ICMR, or the answer is not in the context, or if it involves writing code, making calculations, or any unrelated tasks, respond ONLY with 'I don't know.
     'Do not provide any other information, explanations, or responses outside of ICMR-related queries from the context."""),

    ("human", "Question: {question}\n\nContext:\n{context}")
])
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

@traceable(name="setup_pipeline", tags=["setup"])
def setup_pipeline(pdf_path: str, chunk_size=1000, chunk_overlap=150, embed_model_name="bge-m3:latest", force_rebuild=False):
    return load_or_build_index(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_name=embed_model_name,
        force_rebuild=force_rebuild,
    )

@traceable(name="RAG_run")
def setup_pipeline_and_query(
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "bge-m3:latest",
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(PDF_PATHS, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    parallel = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()

    return chain.invoke(
        question,
        config={"run_name": "pdf_rag_query", "tags": ["qa"], "metadata": {"k": 4}}
    )

@traceable(name="RAG_stream")
def setup_pipeline_and_stream(
    question: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    embed_model_name: str = "bge-m3:latest",
    force_rebuild: bool = False,
):
    vectorstore = setup_pipeline(PDF_PATHS, chunk_size, chunk_overlap, embed_model_name, force_rebuild)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = format_docs(docs)

    parallel = RunnableParallel({
        "context": RunnableLambda(lambda x: context),
        "question": RunnablePassthrough(),
    })
    chain = parallel | prompt | llm | StrOutputParser()

    stream = chain.stream(
        question,
        config={"run_name": "pdf_rag_stream", "tags": ["qa"], "metadata": {"k": 4}}
    )
    return stream, docs

# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['thread_histories'][st.session_state['thread_id']] = []
    st.session_state['message_history'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    if thread_id not in st.session_state['thread_histories']:
        st.session_state['thread_histories'][thread_id] = []

def load_conversation(thread_id):
    return st.session_state['thread_histories'].get(thread_id, [])

# **************************************** Session Setup ******************************
if 'thread_histories' not in st.session_state:
    st.session_state['thread_histories'] = {}

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = list(st.session_state['thread_histories'].keys())

add_thread(st.session_state['thread_id'])

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = load_conversation(st.session_state['thread_id'])

# **************************************** Sidebar UI *********************************

st.sidebar.title('START NEW CHAT')

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)[:8] + '...'):  # Shorten UUID for better UI
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        st.session_state['message_history'] = messages[:]  # Copy list to avoid reference issues

# **************************************** Main UI ************************************

st.title("🖐 HELLO ICMR")
# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])  # Use markdown for better formatting if needed

user_input = st.chat_input('Ask a question about the ICMR')

if user_input:
    # Add user message to current history and thread history
    user_msg = {'role': 'user', 'content': user_input}
    st.session_state['message_history'].append(user_msg)
    st.session_state['thread_histories'][st.session_state['thread_id']].append(user_msg)
    with st.chat_message('user'):
        st.markdown(user_input)

    # Generate response using RAG stream
    with st.chat_message('assistant'):
        try:
            stream, retrieved_docs = setup_pipeline_and_stream(user_input)
            ai_message = st.write_stream(stream)
            
            # Display references
            if retrieved_docs:
                st.markdown("**References:**")
                for doc in retrieved_docs:
                    # Get the absolute path and convert to URL format
                    file_path = Path(doc.metadata['source'])
                    try:
                        relative_path = file_path.relative_to(directory)
                        url = "https://" + "/".join(relative_path.parts)
                        st.markdown(f"- {url}")
                        # st.markdown(f"  Page {doc.metadata.get('page', 0) + 1}")
                    except ValueError:
                        # If the file is not under the directory path
                        st.markdown(f"- {file_path}")
                        # st.markdown(f"  Page {doc.metadata.get('page', 0) + 1}")
                
                references_text = "\n\n**References:**\n" + "\n".join(
                    f"- https://www.icmr.gov.in/{'/'.join(Path(doc.metadata['source']).relative_to(directory).parts)}\n  Page {doc.metadata.get('page', 0) + 1}" 
                    for doc in retrieved_docs
                    if Path(doc.metadata['source']).is_relative_to(directory)
                )
                ai_message += references_text
            else:
                st.markdown("**No references found.**")
                ai_message += "\n\n**No references found.**"
        except Exception as e:
            ai_message = f"Error generating response: {str(e)}"
            st.error(ai_message)

    # Add AI message to histories
    ai_msg = {'role': 'assistant', 'content': ai_message}
    st.session_state['message_history'].append(ai_msg)
    st.session_state['thread_histories'][st.session_state['thread_id']].append(ai_msg)

# ----------------- CLI (Optional, for testing RAG standalone) -----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        ans = setup_pipeline_and_query(q)
        print("\nA:", ans)
    else:
        print("PDF RAG ready. Ask a question via command line args or run the Streamlit app.")