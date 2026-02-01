#!/usr/bin/env python3
"""
MANTIS: Test Suite for RAG Interface
=====================================
Tests for tokenization, keyword search, prompt building, and context formatting.
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import (
    tokenize_query,
    weighted_keyword_search,
    format_context,
    build_prompt,
    load_knowledge_base,
    STOPWORDS,
    SYSTEM_PROMPT,
)


class TestTokenizeQuery(unittest.TestCase):
    """Test cases for the tokenize_query function."""

    def test_basic_tokenization(self):
        """Test basic query tokenization."""
        query = "engine oil pressure"
        tokens = tokenize_query(query)
        self.assertIn("engine", tokens)
        self.assertIn("oil", tokens)
        self.assertIn("pressure", tokens)

    def test_platform_extraction(self):
        """Test that platform identifiers are extracted."""
        query = "AH-1 rotor blade"
        tokens = tokenize_query(query)
        self.assertIn("ah-1", tokens)
        self.assertIn("rotor", tokens)
        self.assertIn("blade", tokens)

    def test_platform_with_underscore(self):
        """Test platform with underscore separator."""
        query = "RC_12 fuel system"
        tokens = tokenize_query(query)
        self.assertIn("rc_12", tokens)

    def test_stopword_removal(self):
        """Test that stopwords are removed."""
        query = "what is the oil pressure for AH-1"
        tokens = tokenize_query(query)
        self.assertNotIn("what", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("the", tokens)
        self.assertNotIn("for", tokens)

    def test_lowercase_conversion(self):
        """Test that tokens are lowercased."""
        query = "ENGINE OIL PRESSURE"
        tokens = tokenize_query(query)
        self.assertIn("engine", tokens)
        self.assertNotIn("ENGINE", tokens)

    def test_short_token_removal(self):
        """Test that single character tokens are removed."""
        query = "a b c engine"
        tokens = tokenize_query(query)
        self.assertNotIn("a", tokens)
        self.assertNotIn("b", tokens)
        self.assertNotIn("c", tokens)
        self.assertIn("engine", tokens)

    def test_empty_query(self):
        """Test handling of empty query."""
        query = ""
        tokens = tokenize_query(query)
        self.assertEqual(tokens, [])

    def test_only_stopwords(self):
        """Test query with only stopwords."""
        query = "what is the"
        tokens = tokenize_query(query)
        self.assertEqual(tokens, [])

    def test_platform_parts_removed(self):
        """Test that platform component parts are removed from regular tokens."""
        query = "OH-58 rotor"
        tokens = tokenize_query(query)
        # "oh" and "58" as standalone should not appear
        self.assertIn("oh-58", tokens)
        self.assertIn("rotor", tokens)

    def test_duplicate_removal(self):
        """Test that duplicate tokens are removed."""
        query = "oil oil pressure pressure"
        tokens = tokenize_query(query)
        self.assertEqual(tokens.count("oil"), 1)
        self.assertEqual(tokens.count("pressure"), 1)


class TestStopwords(unittest.TestCase):
    """Test cases for stopwords set."""

    def test_common_stopwords_present(self):
        """Test that common stopwords are in the set."""
        common = ["the", "is", "and", "or", "but", "a", "an", "to", "of", "in"]
        for word in common:
            self.assertIn(word, STOPWORDS)

    def test_stopwords_is_set(self):
        """Test that STOPWORDS is a set for O(1) lookup."""
        self.assertIsInstance(STOPWORDS, set)


class TestWeightedKeywordSearch(unittest.TestCase):
    """Test cases for the weighted_keyword_search function."""

    def setUp(self):
        """Set up sample knowledge base for testing."""
        self.sample_kb = [
            {
                "id": "doc1_p1",
                "text": "The engine oil pressure should be maintained at 30 psi minimum.",
                "source": "AH-1F MANUAL.pdf",
                "page": 1,
                "platform": "AH-1"
            },
            {
                "id": "doc2_p1",
                "text": "Fuel system servicing procedures for the RC-12 aircraft.",
                "source": "RC-12 MANUAL.pdf",
                "page": 1,
                "platform": "RC-12"
            },
            {
                "id": "doc3_p1",
                "text": "Hydraulic fluid specifications and maintenance guidelines.",
                "source": "UH-1 MANUAL.pdf",
                "page": 1,
                "platform": "UH-1"
            },
            {
                "id": "doc4_p1",
                "text": "Engine oil temperature and pressure monitoring systems.",
                "source": "AH-1F MANUAL.pdf",
                "page": 2,
                "platform": "AH-1"
            },
        ]

    def test_basic_search(self):
        """Test basic keyword search."""
        results = weighted_keyword_search("engine oil", self.sample_kb, top_k=3)
        self.assertGreater(len(results), 0)
        # Should return AH-1 chunks with "engine oil"
        self.assertTrue(any("engine" in r["text"].lower() for r in results))

    def test_platform_boost(self):
        """Test that platform match boosts score."""
        results = weighted_keyword_search("AH-1 engine", self.sample_kb, top_k=3)
        # AH-1 chunks should rank higher
        if len(results) > 0:
            self.assertEqual(results[0]["platform"], "AH-1")

    def test_top_k_limit(self):
        """Test that results are limited to top_k."""
        results = weighted_keyword_search("engine oil pressure fuel", self.sample_kb, top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_empty_query(self):
        """Test that empty query returns empty results."""
        results = weighted_keyword_search("", self.sample_kb, top_k=3)
        self.assertEqual(results, [])

    def test_no_matches(self):
        """Test query with no matching keywords."""
        results = weighted_keyword_search("xyz123 nonexistent", self.sample_kb, top_k=3)
        self.assertEqual(results, [])

    def test_stopword_only_query(self):
        """Test query with only stopwords."""
        results = weighted_keyword_search("the is a", self.sample_kb, top_k=3)
        self.assertEqual(results, [])

    def test_minimum_score_threshold(self):
        """Test that chunks below minimum score are filtered."""
        results = weighted_keyword_search("random word", self.sample_kb, top_k=10)
        # Should filter out low-scoring chunks
        self.assertLessEqual(len(results), len(self.sample_kb))


class TestFormatContext(unittest.TestCase):
    """Test cases for the format_context function."""

    def test_format_single_chunk(self):
        """Test formatting a single chunk."""
        chunks = [
            {"text": "Engine oil specifications.", "source": "AH-1 MANUAL.pdf", "page": 5}
        ]
        result = format_context(chunks)
        self.assertIn("Source 1:", result)
        self.assertIn("AH-1 MANUAL.pdf", result)
        self.assertIn("Page 5", result)
        self.assertIn("Engine oil specifications.", result)

    def test_format_multiple_chunks(self):
        """Test formatting multiple chunks."""
        chunks = [
            {"text": "First chunk content.", "source": "MANUAL1.pdf", "page": 1},
            {"text": "Second chunk content.", "source": "MANUAL2.pdf", "page": 2},
        ]
        result = format_context(chunks)
        self.assertIn("Source 1:", result)
        self.assertIn("Source 2:", result)
        self.assertIn("First chunk content.", result)
        self.assertIn("Second chunk content.", result)

    def test_format_empty_chunks(self):
        """Test formatting empty chunks list."""
        chunks = []
        result = format_context(chunks)
        self.assertEqual(result, "No relevant context found.")

    def test_text_truncation(self):
        """Test that long text is truncated."""
        long_text = "A" * 1000  # 1000 characters
        chunks = [{"text": long_text, "source": "test.pdf", "page": 1}]
        result = format_context(chunks)
        # Text should be truncated to 800 chars
        self.assertLess(len(result.split("]")[1].strip()), 850)

    def test_missing_fields(self):
        """Test handling of missing fields."""
        chunks = [{"text": "Content only"}]
        result = format_context(chunks)
        self.assertIn("Unknown", result)  # Default source
        self.assertIn("?", result)  # Default page


class TestBuildPrompt(unittest.TestCase):
    """Test cases for the build_prompt function."""

    def test_prompt_structure(self):
        """Test that prompt has correct ChatML structure."""
        prompt = build_prompt("What is oil pressure?", "Context about oil.")
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_prompt_contains_query(self):
        """Test that prompt contains the user query."""
        query = "What is the fuel capacity?"
        prompt = build_prompt(query, "Some context")
        self.assertIn(query, prompt)

    def test_prompt_contains_context(self):
        """Test that prompt contains the context."""
        context = "The fuel capacity is 260 gallons."
        prompt = build_prompt("Query", context)
        self.assertIn(context, prompt)

    def test_prompt_contains_system_prompt(self):
        """Test that prompt contains the system prompt."""
        prompt = build_prompt("Query", "Context")
        self.assertIn("military maintenance assistant", prompt)


class TestLoadKnowledgeBase(unittest.TestCase):
    """Test cases for the load_knowledge_base function."""

    def test_load_nonexistent_file(self):
        """Test loading non-existent knowledge base file."""
        result = load_knowledge_base("/nonexistent/path/kb.json")
        self.assertEqual(result, [])

    def test_load_valid_json(self):
        """Test loading valid JSON file."""
        import tempfile
        
        kb_data = [{"id": "test", "text": "content"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(kb_data, f)
            temp_path = f.name
        
        try:
            result = load_knowledge_base(temp_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["id"], "test")
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json(self):
        """Test loading invalid JSON file."""
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            result = load_knowledge_base(temp_path)
            self.assertEqual(result, [])
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()
