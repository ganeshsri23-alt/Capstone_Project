import os
import re
import json
import time
import uuid
import sqlite3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from pathlib import Path

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm

# TASK 1: CHUNK POLICY DOCUMENTS
def chunk_documents(chunk_size=40, overlap=10):
    chunks = []
    policy_folder = "data/policies"
    for filename in sorted(os.listdir(policy_folder)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(policy_folder, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        words = text.split()
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "source": filename,
                "text": chunk_text
            })
            if end >= len(words):
                break
            start += chunk_size - overlap
    return chunks

# Execute chunking
chunks = chunk_documents()
print("\n========== POLICY CHUNKING ==========")
print("Total chunks:", len(chunks))
print(pd.Series([c["source"] for c in chunks]).value_counts())

# TASK 2: TF-IDF VECTORISATION AND RETRIEVAL
# Extract text from all chunks
chunk_texts = [chunk["text"] for chunk in chunks]

# Fit TF-IDF on policy chunks
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)
def retrieve(query, k=3):
    # Transform query using the same fitted vectorizer
    query_vector = vectorizer.transform([query])
    # Calculate cosine similarity
    scores = cosine_similarity(query_vector, chunk_vectors)[0]
    # Get top k chunk indexes
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for index in top_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(scores[index]),
            "text": chunks[index]["text"]
        })
    return results

# Test retrieval
test_results = retrieve("refund policy late delivery")
print("\n========== RETRIEVAL TEST ==========")
for result in test_results:
    print("Source:", result["source"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"][:100], "...")
    print()

# TASK 3: POLICY ASSISTANT
POLICY_SYSTEM_PROMPT = """
You are a POLICY ASSISTANT.

Answer questions ONLY using the supplied policy context.
Do not use outside knowledge or invent information.
If the answer is not available in the context,
respond exactly:
I could not find that information in the policy documents.
"""
def answer(question):
    # Retrieve top 3 relevant policy chunks
    retrieved = retrieve(question, k=3)
    # Build context
    context_parts = []
    for item in retrieved:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )
    context = "\n".join(context_parts)

    # User message format
    user_message = (
        f"CONTEXT:\n{context}\n"
        f"QUESTION:\n{question}"
    )
    messages = [
        {
            "role": "system",
            "content": POLICY_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    # Call mock LLM
    response = mock_llm.chat(messages)
    final_answer = response.choices[0].message.content

    # Return answer and sources
    sources = [item["source"] for item in retrieved]
    return {
        "answer": final_answer,
        "sources": sources
    }

# Test policy assistant
print("\n========== POLICY ASSISTANT TEST ==========")
result = answer("What is the refund policy for late delivery?")
print("Answer:", result["answer"])
print("Sources:", result["sources"])

# TASK 4: RETRIEVAL AND GROUNDING EVALUATION
def evaluate_policy():
    evaluation = pd.read_csv(
        "data/raw/policy_eval.csv",
        keep_default_na=False
    )
    results = []
    for _, row in evaluation.iterrows():
        question = row["question"]
        expected_doc = row["expected_doc"]
        expected_fact = row["expected_fact"]
        retrieved = retrieve(question, k=3)
        result = answer(question)
        retrieved_docs = [
            item["source"] for item in retrieved
        ]

        # Empty expected_doc means unanswerable
        answerable = expected_doc.strip().upper() != "NONE"
        doc_found = (
            expected_doc in retrieved_docs
            if answerable else False
        )
        fact_found = (
            expected_fact in result["answer"]
            if answerable else False
        )
        fallback_found = result["answer"].startswith(
            "I could not find"
        )
        results.append({
            "question": question,
            "expected_doc": expected_doc,
            "answerable": answerable,
            "doc_found": doc_found,
            "fact_found": fact_found,
            "fallback_found": fallback_found
        })
    results_df = pd.DataFrame(results)
    answerable_results = results_df[
        results_df["answerable"]
    ]
    unanswerable_results = results_df[
        ~results_df["answerable"]
    ]
    print("\n========== POLICY EVALUATION ==========")
    print(
        "Expected document retrieved:",
        answerable_results["doc_found"].sum(),
        "/",
        len(answerable_results)
    )
    print(
        "Expected fact found:",
        answerable_results["fact_found"].sum(),
        "/",
        len(answerable_results)
    )
    print(
        "Fallback answers:",
        unanswerable_results["fallback_found"].sum(),
        "/",
        len(unanswerable_results)
    )
    return results_df
evaluation_results = evaluate_policy()

# TASK 4B: ZERO-OVERLAP EVALUATION
# Create chunks without overlap
chunks_zero_overlap = chunk_documents(
    chunk_size=40,
    overlap=0
)

# Temporarily replace original chunks
original_chunks = chunks
chunks = chunks_zero_overlap

# Fit a new TF-IDF vectorizer
chunk_texts = [chunk["text"] for chunk in chunks]
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)

# Evaluate again
zero_overlap_results = evaluate_policy()
failed_zero_overlap = zero_overlap_results[
    zero_overlap_results["answerable"] & ~zero_overlap_results["fact_found"]
]
print("\nAnswerable questions that failed with zero overlap:")
for question in failed_zero_overlap["question"]:
    print("-", question)
print("\n========== ZERO OVERLAP CHUNKING ==========")
print("Total chunks:", len(chunks_zero_overlap))

# Restore original chunks and vectorizer
chunks = original_chunks
chunk_texts = [chunk["text"] for chunk in chunks]
vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)
print("Original overlapping chunks restored.")

# TASK 5: REFUND TOOLS
from datetime import datetime
# Tool 1: Get order
def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    query = """
        SELECT order_id, status, order_amount, delivery_minutes
        FROM orders
        WHERE order_id = ?
    """
    result = pd.read_sql_query(
        query,
        conn,
        params=(order_id,)
    )
    conn.close()
    if result.empty:
        return {"error": "order not found"}
    return result.iloc[0].to_dict()

# Tool 2: Search policy
def search_policy(query):
    results = retrieve(query, k=3)
    return {"results": results}

# Tool 3: Calculate refund
def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        refund = float(order_amount)
    else:
        refund = 0.0
    return {"refund_inr": refund}

# Helper: Append ledger entry
def write_ledger(tool_name, order_id, refund_inr):
    os.makedirs("outputs", exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool_name": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr
    }
    with open(
        "outputs/refund_ledger.jsonl",
        "a",
        encoding="utf-8"
    ) as file:

        file.write(json.dumps(entry) + "\n")

# Tool 4: Issue refund
def issue_refund(order_id, refund_inr):
    write_ledger(
        "issue_refund",
        order_id,
        refund_inr
    )
    return {"refund_inr": refund_inr}

# Tool 5: Escalate to human
def escalate_to_human(order_id, refund_inr):
    write_ledger(
        "escalate_to_human",
        order_id,
        refund_inr
    )
    return {"refund_inr": refund_inr}

# Tool specifications in JSON-Schema style
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Retrieve order details using its order ID.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Retrieve relevant refund policy documents.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund amount using delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically issued refund.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]

# Tool map
TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}
print("\n========== REFUND TOOL VERIFICATION ==========")
print("Tools:", list(TOOL_MAP.keys()))
print("Refund for 60 minutes:",
      calculate_refund(128.25, 60))
print("Refund for 30 minutes:",
      calculate_refund(128.25, 30))

# TASK 8: RESUMABLE AGENT
def save_checkpoint(run_id, scenario_id, messages,
                   tokens_used, steps, tools_executed,
                   step_number):
    os.makedirs("state", exist_ok=True)
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "steps": steps,
        "tools_executed": tools_executed,
        "step_number": step_number
    }
    checkpoint_path = f"state/{run_id}.json"
    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(checkpoint, file, indent=4)

def run_agent(
    user_message,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    scenario_id=None,
    run_id=None,
    crash_after_tool=None
):
    if run_id is None:
        run_id = str(uuid.uuid4())
    messages = [
        {
            "role": "system",
            "content": "AGENT: You are an Instamart refund agent. Use tools to verify orders, check policies, calculate refunds, and process or escalate refunds."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]
    steps = []
    tools_executed = []
    llm_calls = 0
    tokens_used = 0
    start_time = time.monotonic()
    status = None
    final_answer = None
    for iteration in range(max_iterations):
        # Timeout check
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = (
                "FALLBACK_TIMEOUT: Time limit reached. "
                "Please review manually."
            )
            break
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        assistant_message = response.choices[0].message

        # Token limit check
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = (
                "FALLBACK_TOKEN_LIMIT: Token limit exceeded. "
                "Please review manually."
            )
            break
        tool_calls = (
            getattr(assistant_message, "tool_calls", None)
            or getattr(assistant_message, "requested_tool_calls", None)
        )

        # Final answer
        if not tool_calls:
            final_answer = assistant_message.content
            status = final_answer.split(":")[0].strip()
            break
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })
        for call in tool_calls:
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_name = call.name
            observation = TOOL_MAP[tool_name](**arguments)
            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })
            tools_executed.append(tool_name)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })
            # Save checkpoint AFTER tool observation
            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                steps,
                tools_executed,
                len(steps)
            )
            # Simulate crash after requested tool execution
            if crash_after_tool is not None:
                if len(tools_executed) == crash_after_tool:
                    raise RuntimeError("SIMULATED CRASH")
    if final_answer is None and status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = (
            "FALLBACK_MAX_ITERATIONS: Agent reached the maximum "
            "number of iterations. Please review manually."
        )
    return {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

# RESUME AGENT
def resume_agent(
    run_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10
):
    checkpoint_path = f"state/{run_id}.json"
    with open(checkpoint_path, "r", encoding="utf-8") as file:
        checkpoint = json.load(file)
    messages = checkpoint["messages"]
    scenario_id = checkpoint["scenario_id"]
    tokens_used = checkpoint["tokens_used"]
    steps = checkpoint["steps"]
    tools_executed = checkpoint["tools_executed"]
    start_time = time.monotonic()
    status = None
    final_answer = None
    llm_calls = 0
    for iteration in range(max_iterations):
        if time.monotonic() - start_time >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = (
                "FALLBACK_TIMEOUT: Time limit reached. "
                "Please review manually."
            )
            break
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens
        assistant_message = response.choices[0].message
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = (
                "FALLBACK_TOKEN_LIMIT: Token limit exceeded. "
                "Please review manually."
            )
            break
        tool_calls = (
            getattr(assistant_message, "tool_calls", None)
            or getattr(assistant_message, "requested_tool_calls", None)
        )
        if not tool_calls:
            final_answer = assistant_message.content
            status = final_answer.split(":")[0].strip()
            break
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call.id,
                    "name": call.name,
                    "arguments": call.arguments
                }
                for call in tool_calls
            ]
        })
        for call in tool_calls:
            arguments = call.arguments
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            tool_name = call.name
            observation = TOOL_MAP[tool_name](**arguments)
            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })
            tools_executed.append(tool_name)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": tool_name,
                "content": json.dumps(observation)
            })
            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                steps,
                tools_executed,
                len(steps)
            )
    if final_answer is None and status is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = (
            "FALLBACK_MAX_ITERATIONS: Agent reached the maximum "
            "number of iterations. Please review manually."
        )
    return {
        "run_id": run_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "tools_executed": tools_executed
    }

# RUN ALL 7 AGENT SCENARIOS
scenarios = pd.read_csv("data/raw/agent_scenarios.csv")
os.makedirs("outputs", exist_ok=True)

# Start each full run with a clean ledger and clean checkpoints
open("outputs/refund_ledger.jsonl", "w", encoding="utf-8").close()
os.makedirs("state", exist_ok=True)
for old_checkpoint in os.listdir("state"):
    if old_checkpoint.endswith(".json"):
        os.remove(os.path.join("state", old_checkpoint))
agent_results = []

# Reset mock before the independent scenario batch
mock_llm.reset()
with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as file:
    for _, scenario in scenarios.iterrows():
        scenario_id = scenario["scenario_id"]

        # Identify the user-message column
        user_message = scenario.get(
            "user_message",
            scenario.get("request", scenario.get("message", None))
        )
        if pd.isna(user_message):
            raise ValueError(
                f"User message missing for scenario {scenario_id}"
            )
        result = run_agent(
            user_message=str(user_message),
            scenario_id=scenario_id
        )
        # Save scenario result
        record = {
            "scenario_id": scenario_id,
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        # DOCX Task 6 verification: expected status and refund amount
        expected_status = str(scenario["expected_outcome"])
        if scenario_id == "S7":
            assert str(result["status"]).startswith("FALLBACK"), (
                f"S7 must fall back, got {result['status']}"
            )
        else:
            assert result["status"] == expected_status, (
                f"{scenario_id}: expected status {expected_status}, "
                f"got {result['status']}"
            )
        if scenario_id in {"S1", "S2", "S3", "S4", "S5", "S6"}:
            refund_observations = [
                step["observation"]["refund_inr"]
                for step in result["steps"]
                if step["tool"] == "calculate_refund"
                and "refund_inr" in step.get("observation", {})
            ]
            expected_refund = float(scenario["expected_refund_inr"])
            if refund_observations:
                actual_refund = float(refund_observations[-1])
            else:
                # NOT_FOUND scenarios never reach calculate_refund; no refund is issued.
                actual_refund = 0.0
            assert abs(actual_refund - expected_refund) <= 0.01, (
                f"{scenario_id}: expected refund INR {expected_refund:.2f}, "
                f"got INR {actual_refund:.2f}"
            )
        file.write(json.dumps(record) + "\n")
        agent_results.append(record)
        print(f"\nScenario: {scenario_id}")
        print("Status:", result["status"])
        print("Final answer:", result["final_answer"])
        print("LLM calls:", result["llm_calls"])
        print("Tokens:", result["tokens_used"])
        print("Tools:", [step["tool"] for step in result["steps"]])

# AGENT SUMMARY
agent_results_df = pd.DataFrame(agent_results)
print("\n========== AGENT SCENARIO RESULTS ==========")
print(
    agent_results_df[
        ["scenario_id", "status", "llm_calls", "tokens_used"]
    ].to_string(index=False)
)
print("\nSaved: outputs/agent_runs.jsonl")

# TASK 7: TOKEN LIMIT DEMO
print("\n========== TOKEN LIMIT DEMO ==========")
s1_row = scenarios.loc[
    scenarios["scenario_id"] == "S1"
].iloc[0]

# Automatically identify the user-message column
message_column = next(
    col for col in ["user_message", "request", "message"]
    if col in scenarios.columns
)
s1_message = str(s1_row[message_column])
mock_llm.reset()
token_demo = run_agent(
    user_message=s1_message,
    scenario_id="S1_TOKEN_DEMO",
    token_limit=200
)
print("Status:", token_demo["status"])
print("Final answer:", token_demo["final_answer"])
print("LLM calls:", token_demo["llm_calls"])
print("Tokens:", token_demo["tokens_used"])

# TASK 7: TIMEOUT DEMO
print("\n========== TIMEOUT DEMO ==========")
mock_llm.reset()
timeout_demo = run_agent(
    user_message=s1_message,
    scenario_id="S1_TIMEOUT_DEMO",
    time_limit_seconds=0
)
print("Status:", timeout_demo["status"])
print("Final answer:", timeout_demo["final_answer"])
print("LLM calls:", timeout_demo["llm_calls"])
print("Tokens:", timeout_demo["tokens_used"])

# TASK 8: CRASH AND RESUME DEMONSTRATION
print("\n========== CRASH AND RESUME DEMO ==========")

# Reuse s1_message identified in the token-limit demo
# Create a unique run ID
resume_run_id = "S1_RESUME_DEMO"

# Reset mock before starting this independent run
mock_llm.reset()

# First execution: simulate crash after second tool
try:
    first_run = run_agent(
        user_message=s1_message,
        scenario_id="S1",
        run_id=resume_run_id,
        crash_after_tool=2
    )
except RuntimeError as error:
    print("Simulated crash:", error)

# Resume from saved checkpoint
resumed_result = resume_agent(resume_run_id)
print("\n========== RESUMED RESULT ==========")
print("Status:", resumed_result["status"])
print("Final answer:", resumed_result["final_answer"])
print("Tools executed:", resumed_result["tools_executed"])
print("LLM calls after resume:", resumed_result["llm_calls"])
print("Tokens after resume:", resumed_result["tokens_used"])
print("\nTotal tools across crash and resume:")
print(resumed_result["tools_executed"])