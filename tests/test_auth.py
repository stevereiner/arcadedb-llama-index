#!/usr/bin/env python3
"""
Test different authentication combinations for ArcadeDB.
"""

import requests
import json
import pytest

def test_auth_combinations():
    """Test different username/password combinations."""
    
    print("🔧 Testing ArcadeDB authentication...")
    
    # First check if ArcadeDB is available
    try:
        response = requests.get("http://localhost:2480", timeout=2)
    except Exception:
        pytest.skip("ArcadeDB server not available at localhost:2480")
    
    # Try the correct credentials first, then fallbacks
    auth_combinations = [
        ("root", "playwithdata"),  # Correct credentials first
        ("root", "playingwithdata"),  # Common typo
        ("admin", "playwithdata"), 
        ("arcadedb", "playwithdata"),
        ("root", "admin"),
        ("admin", "admin"),
        ("root", ""),
        ("admin", ""),
    ]
    
    for username, password in auth_combinations:
        try:
            print(f"\n🔑 Trying {username}/{password}...")
            
            payload = {
                "command": "SELECT 1 as test",
                "language": "sql"
            }
            
            response = requests.post(
                "http://localhost:2480/api/v1/query/flexible_graphrag",
                json=payload,
                auth=(username, password),
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"✅ SUCCESS with {username}/{password}")
                result = response.json()
                print(f"   Result: {result}")
                assert True  # Test passes if we find valid auth
                return
            else:
                print(f"❌ Failed: {response.status_code} - {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ Error with {username}/{password}: {e}")
    
    print("\n❌ No valid authentication found!")
    assert False, "No valid authentication found"

if __name__ == "__main__":
    test_auth_combinations()
