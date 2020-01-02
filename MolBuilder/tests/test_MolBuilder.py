"""
Unit and regression test for the MolBuilder package.
"""

# Import package, test suite, and other packages as needed
import MolBuilder
import pytest
import sys

def test_MolBuilder_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "MolBuilder" in sys.modules
