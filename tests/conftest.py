#!/usr/bin/env python3
"""
MANTIS: Test Configuration
===========================
Pytest fixtures and configuration for the test suite.
"""

import json
import os
import sys
import tempfile

import pytest

# Add src to path for all tests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_knowledge_base():
    """Provide a sample knowledge base for testing."""
    return [
        {
            "id": "doc1_p1",
            "text": "The AH-1F helicopter engine oil pressure should be maintained between 30 and 40 psi during normal operation.",
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


@pytest.fixture
def temp_json_file():
    """Provide a temporary JSON file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def real_knowledge_base_path():
    """Return path to the real knowledge base if it exists."""
    kb_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base.json')
    if os.path.exists(kb_path):
        return kb_path
    return None
