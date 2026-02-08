import httpx
import time
import json
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    print("Testing /health...")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}, Response: {response.json()}")
    assert response.status_code == 200

def test_stats():
    print("\nTesting /stats...")
    response = httpx.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}, Response: {response.json()}")
    assert response.status_code == 200

def test_query():
    print("\nTesting /query...")
    payload = {
        "query": "temperature sensor",
        "top_k": 2,
        "namespace": "mixed_100", # Using an existing namespace
        "with_audit": True
    }
    response = httpx.post(f"{BASE_URL}/query", json=payload, timeout=30.0)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Results found: {len(data['results'])}")
        if data['results'] and data['results'][0].get('audit'):
            print(f"Audit Result: {data['results'][0]['audit']['label']}")
    else:
        print(f"Error: {response.text}")

def test_multi_query():
    print("\nTesting /multi-query...")
    payload = {
        "queries": ["pressure sensor", "mixer unit"],
        "top_k": 1,
        "namespace": "mixed_100"
    }
    response = httpx.post(f"{BASE_URL}/multi-query", json=payload, timeout=30.0)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Queries processed: {len(data['results'])}")

def test_config():
    print("\nTesting /config...")
    payload = {"alpha": 0.8, "top_k": 5}
    response = httpx.post(f"{BASE_URL}/config", json=payload)
    print(f"Status: {response.status_code}, Response: {response.json()}")
    assert response.status_code == 200

def test_logs():
    print("\nTesting /logs...")
    response = httpx.get(f"{BASE_URL}/logs?lines=5")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Log lines: {len(response.json()['logs'])}")

if __name__ == "__main__":
    print("Starting API tests...")
    try:
        test_health()
        test_stats()
        test_query()
        test_multi_query()
        test_config()
        test_logs()
        print("\nAll basic tests completed!")
    except Exception as e:
        print(f"\nTests failed: {e}")
