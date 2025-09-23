#!/usr/bin/env python3
"""
Test with the correct password from docker-compose.yml
"""

import requests
import json

def test_correct_auth():
    """Test with the correct credentials."""
    
    print("🔧 Testing ArcadeDB with correct credentials...")
    
    try:
        # Use the password from docker-compose.yml: playwithdata (not playingwithdata)
        auth = ("root", "playwithdata")
        
        print("\n1️⃣ Testing connection with root/playwithdata...")
        
        payload = {
            "command": "SELECT count(*) as total FROM schema:types",
            "language": "sql"
        }
        
        response = requests.post(
            "http://localhost:2480/api/v1/query/flexible_graphrag",
            json=payload,
            auth=auth,
            timeout=10
        )
        
        print(f"✅ Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Schema types: {result}")
            
            # Test INSERT
            print("\n2️⃣ Testing INSERT...")
            payload = {
                "command": "INSERT INTO PERSON SET name = 'TestUser', role = 'Tester'",
                "language": "sql"
            }
            
            response = requests.post(
                "http://localhost:2480/api/v1/command/flexible_graphrag",
                json=payload,
                auth=auth,
                timeout=10
            )
            
            print(f"✅ INSERT Status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ INSERT result: {result}")
                
                # Check if it was inserted
                print("\n3️⃣ Checking inserted record...")
                payload = {
                    "command": "SELECT name, role FROM PERSON WHERE name = 'TestUser'",
                    "language": "sql"
                }
                
                response = requests.post(
                    "http://localhost:2480/api/v1/query/flexible_graphrag",
                    json=payload,
                    auth=auth,
                    timeout=10
                )
                
                print(f"✅ SELECT Status: {response.status_code}")
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Found record: {result}")
                else:
                    print(f"❌ SELECT failed: {response.text}")
            else:
                print(f"❌ INSERT failed: {response.text}")
        else:
            print(f"❌ Connection failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_correct_auth()
