#!/usr/bin/env python3
"""
MANTIS: Test Suite for Ingestion Pipeline
==========================================
Tests for PDF ingestion, text cleaning, and platform detection.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ingest import (
    clean_text,
    detect_platform,
    extract_pdf_pages,
    ingest_manuals,
    save_knowledge_base,
    PLATFORM_PATTERNS,
)


class TestCleanText(unittest.TestCase):
    """Test cases for the clean_text function."""

    def test_collapse_whitespace(self):
        """Test that multiple spaces are collapsed to single space."""
        text = "Hello    World"
        result = clean_text(text)
        self.assertEqual(result, "Hello World")

    def test_collapse_newlines(self):
        """Test that newlines are collapsed to single space."""
        text = "Hello\n\n\nWorld"
        result = clean_text(text)
        self.assertEqual(result, "Hello World")

    def test_collapse_tabs(self):
        """Test that tabs are collapsed to single space."""
        text = "Hello\t\t\tWorld"
        result = clean_text(text)
        self.assertEqual(result, "Hello World")

    def test_strip_leading_trailing(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "   Hello World   "
        result = clean_text(text)
        self.assertEqual(result, "Hello World")

    def test_mixed_whitespace(self):
        """Test handling of mixed whitespace characters."""
        text = "  Hello  \n\t  World  \n"
        result = clean_text(text)
        self.assertEqual(result, "Hello World")

    def test_empty_string(self):
        """Test that empty string returns empty string."""
        text = ""
        result = clean_text(text)
        self.assertEqual(result, "")

    def test_only_whitespace(self):
        """Test that whitespace-only string returns empty string."""
        text = "   \n\t  "
        result = clean_text(text)
        self.assertEqual(result, "")

    def test_preserve_single_spaces(self):
        """Test that single spaces between words are preserved."""
        text = "Hello World Test"
        result = clean_text(text)
        self.assertEqual(result, "Hello World Test")


class TestDetectPlatform(unittest.TestCase):
    """Test cases for the detect_platform function."""

    def test_detect_ah1(self):
        """Test detection of AH-1 platform."""
        filename = "AH-1F ATTACK HELICOPTER TECHNICAL MANUAL.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "AH-1")

    def test_detect_ah1_underscore(self):
        """Test detection of AH_1 with underscore."""
        filename = "AH_1F MANUAL.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "AH-1")

    def test_detect_ah1_no_separator(self):
        """Test detection of AH1 without separator."""
        filename = "AH1F MANUAL.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "AH-1")

    def test_detect_rc12(self):
        """Test detection of RC-12 platform."""
        filename = "RC-12D MAINTENANCE TEST FLIGHT MANUAL.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "RC-12")

    def test_detect_uh1(self):
        """Test detection of UH-1 platform."""
        filename = "UH-1 HELICOPTER MAINTENANCE TEST.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "UH-1")

    def test_detect_oh58(self):
        """Test detection of OH-58 platform."""
        filename = "OH-58AC TECHNICAL MANUAL.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "OH-58")

    def test_detect_c12(self):
        """Test detection of C-12 platform (with word boundary)."""
        filename = "C-12C AIRCRAFT MAINTENANCE.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "C-12")

    def test_detect_unknown(self):
        """Test that unknown platforms return UNKNOWN."""
        filename = "RANDOM DOCUMENT.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "UNKNOWN")

    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        filename = "ah-1f manual.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "AH-1")

    def test_rd12(self):
        """Test detection of RD-12 platform."""
        filename = "RD-12G MAINTENANCE MANUAL.pdf"
        result = detect_platform(filename)
        self.assertEqual(result, "RD-12")


class TestPlatformPatterns(unittest.TestCase):
    """Test that platform patterns are correctly defined."""

    def test_all_patterns_are_compiled_regex(self):
        """Test that all patterns are compiled regex objects."""
        import re
        for platform, pattern in PLATFORM_PATTERNS.items():
            self.assertIsInstance(pattern, type(re.compile("")))

    def test_expected_platforms_exist(self):
        """Test that expected platforms are defined."""
        expected = ["AH-1", "RC-12", "UH-1", "OH-58", "C-12"]
        for platform in expected:
            self.assertIn(platform, PLATFORM_PATTERNS)


class TestSaveKnowledgeBase(unittest.TestCase):
    """Test cases for the save_knowledge_base function."""

    def test_save_to_json(self):
        """Test that knowledge base is saved correctly to JSON."""
        kb = [
            {"id": "doc1_p1", "text": "Test content", "source": "test.pdf", "page": 1, "platform": "AH-1"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_knowledge_base(kb, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["id"], "doc1_p1")
            self.assertEqual(loaded[0]["text"], "Test content")
        finally:
            os.unlink(temp_path)

    def test_save_empty_kb(self):
        """Test saving an empty knowledge base."""
        kb = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_knowledge_base(kb, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertEqual(loaded, [])
        finally:
            os.unlink(temp_path)

    def test_save_unicode_content(self):
        """Test saving content with unicode characters."""
        kb = [
            {"id": "doc1_p1", "text": "Temperature: 90°F (32°C)", "source": "test.pdf", "page": 1, "platform": "AH-1"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            save_knowledge_base(kb, temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
            
            self.assertIn("°", loaded[0]["text"])
        finally:
            os.unlink(temp_path)


class TestIngestManuals(unittest.TestCase):
    """Test cases for the ingest_manuals function."""

    def test_nonexistent_folder(self):
        """Test handling of non-existent folder."""
        result = ingest_manuals("/nonexistent/folder/path")
        self.assertEqual(result, [])

    def test_empty_folder(self):
        """Test handling of empty folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ingest_manuals(temp_dir)
            self.assertEqual(result, [])


class TestExtractPdfPages(unittest.TestCase):
    """Test cases for the extract_pdf_pages function."""

    def test_nonexistent_pdf(self):
        """Test handling of non-existent PDF file."""
        result = extract_pdf_pages("/nonexistent/file.pdf")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
