#!/usr/bin/env python3
"""
Test script for MCP server.
Verifies that the server responds correctly to MCP protocol messages.
"""

import subprocess
import json
import sys

def test_mcp_server():
    print("🧪 Testing RAG MCP Server...\n")
    
    # Start MCP server process
    proc = subprocess.Popen(
        [sys.executable, "mcp_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Test 1: Initialize
        print("1️⃣ Testing initialize...")
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0"}
            }
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        
        response = json.loads(proc.stdout.readline())
        assert response["result"]["protocolVersion"] == "2024-11-05"
        print("   ✅ Initialize successful\n")
        
        # Test 2: List tools
        print("2️⃣ Testing tools/list...")
        request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        
        response = json.loads(proc.stdout.readline())
        tools = response["result"]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "search_documents" in tool_names
        assert "index_folder" in tool_names
        assert "get_index_status" in tool_names
        print(f"   ✅ Found {len(tools)} tools: {tool_names}\n")
        
        # Test 3: Get index status
        print("3️⃣ Testing get_index_status...")
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_index_status",
                "arguments": {}
            }
        }
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()
        
        response = json.loads(proc.stdout.readline())
        status = json.loads(response["result"]["content"][0]["text"])
        print(f"   ✅ Index status: {status}\n")
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        stderr = proc.stderr.read()
        print(f"Server stderr: {stderr}")
    finally:
        proc.terminate()


if __name__ == "__main__":
    test_mcp_server()
