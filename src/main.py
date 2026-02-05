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


def load_knowledge_base(filepath: str) -> list:
    """Load the JSON knowledge base into memory."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            kb = json.load(f)
        print(f"[INFO] Loaded {len(kb)} chunks from '{filepath}'")
        return kb
    except FileNotFoundError:
        print(f"[ERROR] Knowledge base '{filepath}' not found!")
        print("        Run 'python ingest.py' first to build it.")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in '{filepath}': {e}")
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
    
    # First, extract platform identifiers (e.g., ah-1, rc-12, uh-60)
    platform_tokens = re.findall(r"[a-z]{1,2}[\-_]?\d{1,2}", query_lower)
    
    # Build a set of component parts to exclude (e.g., "oh", "58" from "oh-58")
    platform_parts = set()
    for pt in platform_tokens:
        # Split on dash/underscore and add parts
        parts = re.findall(r"[a-z]+|\d+", pt)
        platform_parts.update(parts)
    
    # Then extract regular alphanumeric tokens
    regular_tokens = re.findall(r"[a-z0-9]+", query_lower)
    
    # Filter stopwords, short tokens, and platform component parts
    regular_tokens = [
        t for t in regular_tokens 
        if t not in STOPWORDS and len(t) > 1 and t not in platform_parts
    ]
    
    # Combine, preserving order and removing duplicates
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

        # Keyword frequency scoring with word boundary matching
        for token in query_tokens:
            # Use word boundary regex to avoid substring matches
            # e.g., "oil" should not match "boil" or "coil"
            pattern = r"\b" + re.escape(token) + r"\b"
            matches = re.findall(pattern, text_lower)
            score += len(matches)

        # Platform bonus: if query mentions a platform, strongly boost matching chunks
        platform = chunk.get("platform", "UNKNOWN")
        if platform != "UNKNOWN" and platform.lower() in query_lower:
            score += 10  # Strong boost for platform match
        elif platform != "UNKNOWN":
            # Penalize chunks from wrong platforms when query specifies a platform
            for pt in ["ah-1", "rc-12", "uh-1", "oh-58", "c-12", "ch-47", "uh-60"]:
                if pt in query_lower and pt not in platform.lower():
                    score -= 5  # Penalty for wrong platform
                    break

        # Exact phrase bonus (for multi-word queries)
        if len(query_tokens) >= 2:
            phrase = " ".join(query_tokens[:3])  # First 3 tokens as phrase
            if phrase in text_lower:
                score += 5

        # Only include chunks with meaningful relevance
        if score >= 3:  # Minimum threshold to filter noise
            scored_chunks.append((score, chunk))

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)

    # Return top_k chunks (or empty if none meet threshold)
    return [chunk for _, chunk in scored_chunks[:top_k]]


def format_context(chunks: list) -> str:
    """Format retrieved chunks into a context string for the LLM."""
    if not chunks:
        return "No relevant context found."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "Unknown")
        page = chunk.get("page", "?")
        text = chunk.get("text", "")[:800]  # Truncate to save tokens
        context_parts.append(f"[Source {i}: {source}, Page {page}]\n{text}")

    return "\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """Build the full prompt for KoboldCPP using Qwen's ChatML format."""
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
            "\n\nHowever",
            "\n\nThis section",
            "\n\nNote:",
            "<|im_end|>",
            "Not found in loaded manuals.",  # Stop immediately on refusal
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
            return result["results"][0].get("text", "").strip()
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


def main():
    print_banner()

    # Load knowledge base
    knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_FILE)
    if not knowledge_base:
        print("[FATAL] Cannot proceed without knowledge base. Exiting.")
        return

    # Check server status
    if not check_server_status():
        print("[WARN] KoboldCPP server not detected on port 5001!")
        print("       Start 'server.exe' with your model, then try again.")
        print("       Continuing anyway - server may come online later.\n")

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

        # Handle commands
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

        # Retrieve relevant chunks
        print("[...] Searching knowledge base...")
        chunks = weighted_keyword_search(user_input, knowledge_base, TOP_K_CHUNKS)
        last_chunks = chunks

        # Pre-filter: Check if query asks about unsupported platforms
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

        # Build context and prompt
        context = format_context(chunks)
        prompt = build_prompt(user_input, context)

        # Query the LLM
        print("[...] Generating response...")
        response = query_kobold(prompt)

        print(f"\nAssistant: {response}")
        print(f"           [Sources: {len(chunks)} chunks from knowledge base]\n")


if __name__ == "__main__":
    main()
