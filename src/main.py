#!/usr/bin/env python3
"""
MANTIS: Field Manual RAG Chat Interface
==================================================
Libraries: requests, json only.
"""

import json
import os
import re
import sys
import subprocess
import atexit
import time
import glob
from audit_logger import log_interaction
from security import load_encrypted_json

try:
    import requests
except ImportError:
    print("[FATAL] 'requests' library not found. Install with: pip install requests")
    sys.exit(1)


# --- Configuration ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_FILE = os.path.join(_SCRIPT_DIR, "..", "data", "knowledge_base.json")
KOBOLD_API_URL = "http://localhost:5001/api/v1/generate"
KOBOLD_MODEL_URL = "http://localhost:5001/api/v1/model"

# Generation parameters
GEN_PARAMS = {
    "temperature": 0.05,
    "max_length": 1000,
    "top_p": 0.9,
    "top_k": 40,
    "rep_pen": 1.1,
}

# Number of context chunks to retrieve
TOP_K_CHUNKS = 3

# System instruction for the LLM
SYSTEM_PROMPT = """You are a military maintenance assistant.
Answer ONLY using the Context below. Cite the source document.
If the answer is NOT in the Context, say: "Not found in loaded manuals."
NEVER invent or guess procedures."""


# --- Stopwords for keyword filtering ---
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "shall", "can", "to", "of", "in",
    "for", "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "about", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "i", "me", "my", "we",
    "our", "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "if", "up", "out", "off", "over", "any",
}

# --- Synonyms ---
SYNONYMS = {
    "chopper": "helicopter",
    "heli": "helicopter",
    "bird": "helicopter",
    "specs": "specifications",
    "rpm": "revolutions",
    "psi": "pressure",
    "lbs": "pounds",
    "temp": "temperature",
    "max": "maximum",
    "min": "minimum",
    "batt": "battery",
    "trans": "transmission",
    "nav": "navigation"
}


def load_knowledge_base(filepath: str) -> list:
    """Load the JSON knowledge base into memory."""
    try:
        kb = load_encrypted_json(filepath)
        if not kb:
            print(f"[WARN] Knowledge base '{filepath}' is empty or not found!")
            print("       Run 'python ingest.py' first to build it.")
        else:
            print(f"[INFO] Loaded {len(kb)} chunks from '{filepath}'")
        return kb
    except Exception as e:
        print(f"[ERROR] Failed to load '{filepath}': {e}")
        return []


def tokenize_query(query: str) -> list:
    """
    Tokenize and clean query for keyword matching.
    - Lowercase
    - Preserve platform identifiers (e.g., AH-1, RC-12)
    - Filter stopwords
    - Remove component parts of platform identifiers
    """
    query_lower = query.lower()
    
    platform_tokens = re.findall(r"[a-z]{1,2}[\-_]?\d{1,2}", query_lower)
    
    platform_parts = set()
    for pt in platform_tokens:
        parts = re.findall(r"[a-z]+|\d+", pt)
        platform_parts.update(parts)
    
    regular_tokens = re.findall(r"[a-z0-9]+", query_lower)
    
    regular_tokens = [
        t for t in regular_tokens 
        if t not in STOPWORDS and len(t) > 1 and t not in platform_parts
    ]
    
    regular_tokens = [SYNONYMS.get(t, t) for t in regular_tokens]
    
    seen = set()
    tokens = []
    for t in platform_tokens + regular_tokens:
        if t not in seen:
            seen.add(t)
            tokens.append(t)
    
    return tokens


def weighted_keyword_search(query: str, knowledge_base: list, top_k: int = 3) -> list:
    """
    Score chunks based on keyword frequency matching.
    Returns top_k most relevant chunks.
    
    Scoring:
    - Each keyword match = +1 point
    - Platform match in query = +3 bonus points
    - Exact phrase match = +5 bonus points
    """
    query_lower = query.lower()
    query_tokens = tokenize_query(query)

    if not query_tokens:
        return []

    scored_chunks = []

    for chunk in knowledge_base:
        text_lower = chunk["text"].lower()
        score = 0

        for token in query_tokens:
            pattern = r"\b" + re.escape(token) + r"\b"
            matches = re.findall(pattern, text_lower)
            if matches:
                score += 10.0 + (len(matches) * 0.1)

        platform = chunk.get("platform", "UNKNOWN")
        if platform != "UNKNOWN" and platform.lower() in query_lower:
            score += 10  # Strong boost for platform match
        elif platform != "UNKNOWN":
            for pt in ["ah-1", "rc-12", "uh-1", "oh-58", "c-12", "ch-47", "uh-60"]:
                if pt in query_lower and pt not in platform.lower():
                    score -= 5  # Penalty for wrong platform
                    break

        # Exact phrase bonus (for multi-word queries)
        if len(query_tokens) >= 2:
            phrase = " ".join(query_tokens[:3])
            if phrase in text_lower:
                score += 5

        if score >= 3:
            scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    return [chunk for _, chunk in scored_chunks[:top_k]]


def format_context(chunks: list) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{text}")

    return "\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """Build the full prompt for KoboldCPP using ChatML format (Qwen)."""
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Context:
{context}

Question: {query}

Answer in English:<|im_end|>
<|im_start|>assistant
"""
    return prompt


def query_kobold(prompt: str) -> str:
    """
    Send generation request to KoboldCPP server.
    Handles connection errors gracefully.
    """
    payload = {
        "prompt": prompt,
        "max_length": GEN_PARAMS["max_length"],
        "temperature": GEN_PARAMS["temperature"],
        "top_p": GEN_PARAMS["top_p"],
        "top_k": GEN_PARAMS["top_k"],
        "rep_pen": GEN_PARAMS["rep_pen"],
        "stop_sequence": [
            "<|im_end|>",
            "<|endoftext|>",
        ],
    }

    try:
        response = requests.post(
            KOBOLD_API_URL,
            json=payload,
            timeout=300,  # 5 minute
        )
        response.raise_for_status()
        result = response.json()

        # KoboldCPP response format
        if "results" in result and len(result["results"]) > 0:
            text = result["results"][0].get("text", "").strip()

            if "Not found in loaded manuals" in text:
                return "Not found in loaded manuals."
            return text
        else:
            return "[ERROR] Unexpected response format from server."

    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to KoboldCPP server!\n        Make sure server.exe is running on port 5001."
    except requests.exceptions.Timeout:
        return "[ERROR] Request timed out. The model may be overloaded."
    except requests.exceptions.HTTPError as e:
        # Try to get error details from response body
        try:
            err_detail = response.json().get("detail", str(e))
        except Exception:
            err_detail = str(e)
        return f"[ERROR] Server returned error: {err_detail}"
    except requests.exceptions.RequestException as e:
        return f"[ERROR] API request failed: {e}"


def check_server_status() -> bool:
    """Check if KoboldCPP server is running."""
    try:
        response = requests.get(KOBOLD_MODEL_URL, timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            model_name = model_info.get("result", "Unknown Model")
            print(f"[INFO] KoboldCPP server is running. Model: {model_name}")
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def print_banner():
    """Print the application banner."""
    print()
    print("=" * 60)
    print("  MANTIS: Field Manual RAG System")
    print("=" * 60)
    print()


def print_help():
    """Print help message."""
    print("\n[COMMANDS]")
    print("  /help     - Show this help message")
    print("  /quit     - Exit the program")
    print("  /status   - Check server connection")
    print("  /sources  - Show sources for last query")
    print()


server_process = None

def cleanup_server():
    """Ensure the server subprocess is terminated when the script exits."""
    global server_process
    if server_process and server_process.poll() is None:
        print("\n[INFO] Shutting down KoboldCPP server...")
        server_process.terminate()
        server_process.wait()

atexit.register(cleanup_server)

def start_kobold_server():
    """Start the KoboldCPP server in the background."""
    global server_process
    
    project_root = os.path.join(_SCRIPT_DIR, "..")
    gguf_files = glob.glob(os.path.join(project_root, "*.gguf"))
    if not gguf_files:
        print("[FATAL] No .gguf model found in project directory.")
        sys.exit(1)
        
    model_path = gguf_files[0]
    server_exe = os.path.join(project_root, "server.exe")
    
    if not os.path.exists(server_exe):
        print("[FATAL] 'server.exe' not found in project directory. Please run setup.bat.")
        sys.exit(1)
        
    threads = max(1, os.cpu_count() or 4)
    
    cmd = [
        server_exe,
        "--model", model_path,
        "--port", "5001",
        "--threads", str(threads),
        "--quiet"
    ]
    
    print(f"[INFO] Starting KoboldCPP server in the background (Threads: {threads})...")
    server_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for server to come online
    print("[...] Waiting for server to initialize (this may take a minute)...")
    start_time = time.time()
    while time.time() - start_time < 120:
        if check_server_status():
            print("[OK] Server is ready!\n")
            return
        time.sleep(2)
        
    print("[FATAL] Server failed to start within timeout.")
    sys.exit(1)


def main():
    print_banner()

    # Load knowledge base
    knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_FILE)
    if not knowledge_base:
        print("[FATAL] Cannot proceed without knowledge base. Exiting.")
        return

    if not check_server_status():
        start_kobold_server()
    else:
        print("[INFO] Existing KoboldCPP server detected on port 5001.\n")

    print("[INFO] Type your question, or '/help' for commands.\n")

    last_chunks = []  # Store last retrieved chunks for /sources command

    # Main chat loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n[INFO] Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("[INFO] Goodbye!")
            break
        elif user_input.lower() == "/help":
            print_help()
            continue
        elif user_input.lower() == "/status":
            if check_server_status():
                print("[OK] Server is online.\n")
            else:
                print("[WARN] Server is offline or unreachable.\n")
            continue
        elif user_input.lower() == "/sources":
            if last_chunks:
                print("\n[SOURCES FROM LAST QUERY]")
                for i, chunk in enumerate(last_chunks, 1):
                    print(f"  {i}. {chunk['source']} - Page {chunk['page']} [{chunk['platform']}]")
                print()
            else:
                print("[INFO] No previous query sources available.\n")
            continue

        print("[...] Searching knowledge base...")
        chunks = weighted_keyword_search(user_input, knowledge_base, TOP_K_CHUNKS)
        last_chunks = chunks

        query_lower = user_input.lower()
        unsupported_platforms = [
            "f-16", "f-15", "f-22", "f-35", "f-18", "a-10", "b-52", "b-1", "b-2",
            "747", "737", "777", "787", "a320", "a380", "c-130", "c-17", "c-5",
            "mig", "su-", "su 57", "felon", "tu-", "nuclear", "submarine", "ship", 
            "tank", "m1 abrams", "bradley", "stryker", "humvee",
        ]
        skip_query = False
        for platform in unsupported_platforms:
            if platform in query_lower:
                print(f"\nAssistant: I don't have information about {platform.upper()} in the loaded manuals.")
                print("           The available manuals cover: AH-1, RC-12, C-12, OH-58, UH-1, RD-12.\n")
                skip_query = True
                break
        if skip_query:
            continue

        if not chunks:
            print("\nAssistant: I couldn't find any relevant information for that query.")
            print("           Try rephrasing or using different keywords.\n")
            continue

        context = format_context(chunks)
        prompt = build_prompt(user_input, context)

        print("[...] Generating response...")
        response = query_kobold(prompt)

        print(f"\nAssistant: {response}")
        print(f"           [Sources: {len(chunks)} chunks from knowledge base]\n")
        
        log_interaction(user_input, len(chunks), response)


if __name__ == "__main__":
    main()
