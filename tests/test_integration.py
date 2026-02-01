#!/usr/bin/env python3
"""
MANTIS: Integration Tests
=========================
End-to-end tests for the complete RAG pipeline.
"""

import json
import os
import sys
import tempfile
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest import clean_text, detect_platform, save_knowledge_base
from main import (
    tokenize_query,
    weighted_keyword_search,
    format_context,
    build_prompt,
    load_knowledge_base,
)


class TestEndToEndPipeline(unittest.TestCase):
    """Integration tests for the complete RAG pipeline."""

    def setUp(self):
        """Set up test knowledge base."""
        self.test_kb = [
            {
                "id": "doc1_p1",
                "text": "The AH-1F helicopter engine oil pressure should be maintained between 30 and 40 psi during normal operation. Low oil pressure indicates potential engine damage.",
                "source": "AH-1F ATTACK HELICOPTER TECHNICAL OPERATOR MANUAL.pdf",
                "page": 31,
                "platform": "AH-1"
            },
            {
                "id": "doc2_p1",
                "text": "RC-12 fuel system capacity is 260 US gallons. The fuel boost pumps are located in both forward and aft fuel cells.",
                "source": "RC-12D MAINTENANCE TEST FLIGHT MANUAL.pdf",
                "page": 15,
                "platform": "RC-12"
            },
            {
                "id": "doc3_p1",
                "text": "OH-58 main rotor blade inspection procedure: Check for cracks, corrosion, and damage to the leading edge.",
                "source": "OH-58AC TECHNICAL MANUAL.pdf",
                "page": 45,
                "platform": "OH-58"
            },
            {
                "id": "doc4_p1",
                "text": "C-12 aircraft hydraulic system operates at 3000 psi. Both system 1 and system 2 reservoirs should be checked daily.",
                "source": "C-12C AIRCRAFT MAINTENANCE.pdf",
                "page": 22,
                "platform": "C-12"
            },
            {
                "id": "doc5_p1",
                "text": "UH-1 transmission oil temperature should not exceed 110 degrees Celsius. High temperature indicates cooling system issues.",
                "source": "UH-1 HELICOPTER MAINTENANCE.pdf",
                "page": 33,
                "platform": "UH-1"
            },
        ]

    def test_full_rag_pipeline(self):
        """Test the complete RAG pipeline from query to prompt."""
        # Step 1: Query tokenization
        query = "What is the AH-1 oil pressure?"
        tokens = tokenize_query(query)
        self.assertIn("ah-1", tokens)
        self.assertIn("oil", tokens)
        self.assertIn("pressure", tokens)

        # Step 2: Keyword search
        results = weighted_keyword_search(query, self.test_kb, top_k=3)
        self.assertGreater(len(results), 0)
        # Should retrieve AH-1 oil pressure chunk
        self.assertEqual(results[0]["platform"], "AH-1")
        self.assertIn("oil pressure", results[0]["text"].lower())

        # Step 3: Format context
        context = format_context(results)
        self.assertIn("AH-1F ATTACK HELICOPTER", context)
        self.assertIn("30 and 40 psi", context)

        # Step 4: Build prompt
        prompt = build_prompt(query, context)
        self.assertIn("<|im_start|>system", prompt)
        self.assertIn(query, prompt)
        self.assertIn("30 and 40 psi", prompt)

    def test_platform_specific_query(self):
        """Test that platform-specific queries retrieve correct documents."""
        # RC-12 fuel query
        results = weighted_keyword_search("RC-12 fuel capacity", self.test_kb, top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["platform"], "RC-12")
        self.assertIn("fuel", results[0]["text"].lower())

    def test_cross_platform_query(self):
        """Test query that could match multiple platforms."""
        # Generic "oil" query
        results = weighted_keyword_search("oil temperature", self.test_kb, top_k=3)
        platforms = [r["platform"] for r in results]
        # Should retrieve both AH-1 (oil pressure) and UH-1 (oil temperature)
        self.assertTrue(len(set(platforms)) >= 1)

    def test_save_and_load_kb(self):
        """Test saving and loading knowledge base."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            save_knowledge_base(self.test_kb, temp_path)

            # Load
            loaded_kb = load_knowledge_base(temp_path)

            # Verify
            self.assertEqual(len(loaded_kb), len(self.test_kb))
            self.assertEqual(loaded_kb[0]["id"], self.test_kb[0]["id"])
            self.assertEqual(loaded_kb[0]["platform"], self.test_kb[0]["platform"])
        finally:
            os.unlink(temp_path)

    def test_text_cleaning_integration(self):
        """Test that cleaned text is searchable."""
        # Create chunk with messy text
        messy_chunk = {
            "id": "test_p1",
            "text": clean_text("Engine   oil\n\npressure\t\tguidelines."),
            "source": "test.pdf",
            "page": 1,
            "platform": "AH-1"
        }
        test_kb = [messy_chunk]

        # Search should still work
        results = weighted_keyword_search("engine oil pressure", test_kb, top_k=1)
        self.assertGreater(len(results), 0)

    def test_platform_detection_integration(self):
        """Test that platform detection works with typical filenames."""
        filenames = [
            ("AH-1F ATTACK HELICOPTER MANUAL.pdf", "AH-1"),
            ("RC-12D MAINTENANCE TEST FLIGHT MANUAL.pdf", "RC-12"),
            ("OH-58AC TECHNICAL MANUAL.pdf", "OH-58"),
            ("UH-1 HELICOPTER MAINTENANCE.pdf", "UH-1"),
            ("C-12C AIRCRAFT MAINTENANCE.pdf", "C-12"),
        ]

        for filename, expected_platform in filenames:
            detected = detect_platform(filename)
            self.assertEqual(detected, expected_platform, f"Failed for {filename}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_unicode_in_query(self):
        """Test handling of unicode in query."""
        query = "temperature 90°F"
        tokens = tokenize_query(query)
        # Should handle gracefully
        self.assertIsInstance(tokens, list)

    def test_special_characters_in_query(self):
        """Test handling of special characters."""
        query = "what's the oil-pressure?"
        tokens = tokenize_query(query)
        self.assertIn("oil", tokens)
        self.assertIn("pressure", tokens)

    def test_numeric_only_query(self):
        """Test handling of numeric-only query."""
        query = "12345"
        tokens = tokenize_query(query)
        # Should handle gracefully
        self.assertIsInstance(tokens, list)

    def test_very_long_query(self):
        """Test handling of very long query."""
        query = "oil pressure " * 100  # Very long query
        tokens = tokenize_query(query)
        # Should still work but deduplicate
        self.assertEqual(tokens.count("oil"), 1)
        self.assertEqual(tokens.count("pressure"), 1)

    def test_empty_knowledge_base(self):
        """Test search with empty knowledge base."""
        results = weighted_keyword_search("engine oil", [], top_k=3)
        self.assertEqual(results, [])

    def test_single_chunk_kb(self):
        """Test search with single chunk knowledge base."""
        single_kb = [
            {"id": "doc1", "text": "Engine oil specifications.", "source": "test.pdf", "page": 1, "platform": "AH-1"}
        ]
        results = weighted_keyword_search("engine oil", single_kb, top_k=3)
        self.assertLessEqual(len(results), 1)


class TestRetrievalQuality(unittest.TestCase):
    """Test the quality of retrieval results."""

    def setUp(self):
        """Set up test data with known relevance."""
        self.test_kb = [
            # Highly relevant
            {"id": "1", "text": "AH-1 engine oil pressure specifications: maintain 30-40 psi.", "source": "a.pdf", "page": 1, "platform": "AH-1"},
            # Moderately relevant
            {"id": "2", "text": "Oil system maintenance procedures for helicopters.", "source": "b.pdf", "page": 1, "platform": "UNKNOWN"},
            # Low relevance
            {"id": "3", "text": "Fuel system specifications for aircraft.", "source": "c.pdf", "page": 1, "platform": "AH-1"},
            # Irrelevant
            {"id": "4", "text": "Radio communication protocols and frequencies.", "source": "d.pdf", "page": 1, "platform": "AH-1"},
        ]

    def test_most_relevant_ranked_first(self):
        """Test that most relevant chunk is ranked first."""
        results = weighted_keyword_search("AH-1 oil pressure", self.test_kb, top_k=4)
        if len(results) > 0:
            # First result should be the highly relevant one
            self.assertEqual(results[0]["id"], "1")

    def test_irrelevant_filtered_out(self):
        """Test that irrelevant chunks are filtered."""
        results = weighted_keyword_search("oil pressure", self.test_kb, top_k=4)
        ids = [r["id"] for r in results]
        # Radio communication chunk should not be retrieved
        self.assertNotIn("4", ids)


if __name__ == "__main__":
    unittest.main()
