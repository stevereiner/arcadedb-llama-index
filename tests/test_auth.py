#!/usr/bin/env python3
"""
Test different authentication combinations for ArcadeDB.
"""

import requests
import json

def test_auth_combinations():
    """Test different username/password combinations."""
    
    print("🔧 Testing ArcadeDB authentication...")
    
    # Common username/password combinations to try
    auth_combinations = [
        ("root", "playingwithdata"),
        ("admin", "playingwithdata"), 
        ("arcadedb", "playingwithdata"),
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
                return username, password
            else:
                print(f"❌ Failed: {response.status_code} - {response.text[:100]}")
                
        except Exception as e:
            print(f"❌ Error with {username}/{password}: {e}")
    
    print("\n❌ No valid authentication found!")
    return None, None

if __name__ == "__main__":
    test_auth_combinations()
