"""
Example: Inspecting MLflow trace data with nested request/response structures.
"""

import sys
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from dataframe_inspector import Inspector

# Sample MLflow trace data
trace_data = pd.DataFrame(
    {
        "trace_id": ["trace_001", "trace_002", "trace_003"],
        "request": [
            {
                "query": "What is my deductible?",
                "context": {"user_id": "123", "session": "abc"},
            },
            {
                "query": "Find doctors near me",
                "context": {"user_id": "456", "session": "def"},
            },
            {
                "query": "Check my claims",
                "context": {"user_id": "789", "session": "ghi"},
            },
        ],
        "response": [
            {
                "answer": "Your deductible is $500",
                "metadata": {"confidence": 0.95, "sources": ["policy_doc_1"]},
            },
            {
                "answer": "Here are nearby doctors...",
                "metadata": {"confidence": 0.87, "sources": [{"db": "provider_db"}]},
            },
            {
                "answer": "Your recent claims...",
                "metadata": {"confidence": 0.92, "sources": [{"db": "claims_db"}]},
            },
        ],
    }
)

# Create inspector
inspector = Inspector(trace_data)

print("=" * 80)
print("MLflow Trace Inspection Example")
print("=" * 80)

# Step 1: Overview - see what's in the DataFrame
print("\n📋 Step 1: Get overview of the DataFrame")
inspector.overview()

# Step 2: Inspect specific nested columns
print("\n🔍 Step 2: Deep dive into 'request' column:")
inspector.inspect_column("request", sample_size=2, max_depth=3)

print("\n🔍 Step 3: Deep dive into 'response' column:")
inspector.inspect_column("response", sample_size=2, max_depth=4)

print("\n" + "=" * 80)
print("Example complete!")
print("=" * 80)
