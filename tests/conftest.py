"""Shared pytest fixtures."""
import sys
import os

# Make sure the src package is importable when running tests from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
