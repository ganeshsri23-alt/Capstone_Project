# PART C: SUPPORT TICKET TRIAGE
import pandas as pd
import numpy as np
import re
import json
import random
import time
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import mock_llm

# Configuration
MAX_INPUT_TOKENS = 400
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 100000

# Retry counters (reset for each run_pipeline call)
RETRY_FAILURE_COUNT = 0
PERMANENT_FAILURE_COUNT = 0
ALLOWED_CATEGORIES = {
    "billing",
    "delivery",
    "technical",
    "fraud"
}

# Load tickets
ticket_data = pd.read_csv(
    "data/raw/tickets.csv",
    keep_default_na=False
)
print("\n========== TICKET DATA ==========")
print("Shape:", ticket_data.shape)
print("Columns:", ticket_data.columns.tolist())
print(ticket_data.head())

# TASK 1: INPUT GUARDRAILS
# Empty ticket check
ticket_data["text"] = ticket_data["text"].astype(str)
ticket_data["is_empty"] = ticket_data["text"].str.strip().eq("")

# Token estimation using mock_llm
ticket_data["estimated_tokens"] = ticket_data["text"].apply(
    mock_llm.est_tokens
)

# Identify long tickets
ticket_data["is_long"] = (
    ticket_data["estimated_tokens"] > MAX_INPUT_TOKENS
)
print("\n========== INPUT GUARDRAILS ==========")
print("Empty tickets:", ticket_data["is_empty"].sum())
print("Long tickets:", ticket_data["is_long"].sum())
print("\nLong ticket token estimates:")
print(
    ticket_data.loc[
        ticket_data["is_long"],
        ["ticket_id", "estimated_tokens"]
    ]
)

# TASK 2: PII REDACTION
# Exact patterns and placeholders
EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"
def redact_pii(text):
    detected_email_count = len(re.findall(EMAIL_PATTERN, text))
    detected_phone_count = len(re.findall(PHONE_PATTERN, text))
    detected_order_id_count = len(re.findall(ORDER_PATTERN, text))
    text = re.sub(EMAIL_PATTERN, "[EMAIL]", text)
    text = re.sub(PHONE_PATTERN, "[PHONE]", text)
    text = re.sub(ORDER_PATTERN, "[ORDER_ID]", text)
    return text, detected_email_count, detected_phone_count, detected_order_id_count

# Apply redaction to all tickets
pii_redaction_results = ticket_data["text"].apply(redact_pii)
ticket_data["redacted_text"] = pii_redaction_results.apply(lambda x: x[0])
ticket_data["email_count"] = pii_redaction_results.apply(lambda x: x[1])
ticket_data["phone_count"] = pii_redaction_results.apply(lambda x: x[2])
ticket_data["order_count"] = pii_redaction_results.apply(lambda x: x[3])
print("\n========== REDACTION RESULTS ==========")
print("Emails:", ticket_data["email_count"].sum())
print("Phones:", ticket_data["phone_count"].sum())
print("Order IDs:", ticket_data["order_count"].sum())
print(
    "Total substitutions:",
    ticket_data[["email_count", "phone_count", "order_count"]].sum().sum()
)

# Verify no PII remains after redaction
remaining_email_count = ticket_data["redacted_text"].apply(
    lambda x: len(re.findall(EMAIL_PATTERN, x))
).sum()
remaining_phone_count = ticket_data["redacted_text"].apply(
    lambda x: len(re.findall(PHONE_PATTERN, x))
).sum()
remaining_order_id_count = ticket_data["redacted_text"].apply(
    lambda x: len(re.findall(ORDER_PATTERN, x))
).sum()
print("\nRemaining emails:", remaining_email_count)
print("Remaining phones:", remaining_phone_count)
print("Remaining order IDs:", remaining_order_id_count)
assert remaining_email_count == 0
assert remaining_phone_count == 0
assert remaining_order_id_count == 0

# TASK 3: SYSTEM PROMPTS
ZERO_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.
Allowed categories:
- billing
- delivery
- technical
- fraud
Return JSON only in this format:
{"category": "...", "reason": "..."}
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer support ticket into exactly one category.

Allowed categories:
- billing
- delivery
- technical
- fraud

EXAMPLES:
Ticket: I was charged twice for my order.
{"category": "billing", "reason": "Duplicate charge"}

Ticket: My package has not arrived.
{"category": "delivery", "reason": "Package not received"}

Ticket: Someone used my card without permission.
{"category": "fraud", "reason": "Unauthorized transaction"}

Return JSON only in this format:
{"category": "...", "reason": "..."}
"""
print("\n========== PROMPT VERIFICATION ==========")
print("Zero-shot contains CLASSIFY:", "CLASSIFY" in ZERO_SHOT_PROMPT)
print("Zero-shot contains EXAMPLES:", "EXAMPLES:" in ZERO_SHOT_PROMPT)
print("Few-shot contains CLASSIFY:", "CLASSIFY" in FEW_SHOT_PROMPT)
print("Few-shot contains EXAMPLES:", "EXAMPLES:" in FEW_SHOT_PROMPT)

# TASK 4: RETRY LOGIC
def call_llm_with_retry(llm_messages, current_ticket_id):
    global RETRY_FAILURE_COUNT, PERMANENT_FAILURE_COUNT
    for retry_attempt in range(MAX_RETRIES + 1):
        try:
            llm_response = mock_llm.chat(llm_messages)
            return llm_response
        except mock_llm.TransientLLMError:
            RETRY_FAILURE_COUNT += 1
            if retry_attempt == MAX_RETRIES:
                PERMANENT_FAILURE_COUNT += 1
                raise
            retry_wait_seconds = (
                BASE_DELAY
                * (2 ** retry_attempt)
                * random.uniform(0.9, 1.15)
            )
            print(
                f"Ticket {current_ticket_id} | "
                f"Retry {retry_attempt} | "
                f"Waiting {retry_wait_seconds:.4f} seconds"
            )
            time.sleep(retry_wait_seconds)

# TASK 5: OUTPUT VALIDATION
def validate_response(llm_response_text):
    # Try JSON first
    try:
        validated_result = json.loads(llm_response_text)
        predicted_category = validated_result.get("category")
        if predicted_category in ALLOWED_CATEGORIES:
            return predicted_category, "classified_json"
        return None, "flagged_invalid_output"
    except (json.JSONDecodeError, AttributeError):
        pass
    # Keyword extraction for non-JSON replies
    response_words = re.findall(
        r"[a-z]+",
        llm_response_text.lower()
    )
    matched_category_names = [
        predicted_category
        for predicted_category in ALLOWED_CATEGORIES
        if predicted_category in response_words
    ]
    if len(matched_category_names) == 1:
        return matched_category_names[0], "classified_keyword"
    return None, "flagged_invalid_output"

print("\n========== VALIDATION TEST ==========")
print(
    validate_response(
        '{"category": "billing", "reason": "wrong invoice"}'
    )
)
print(
    validate_response(
        "The customer appears to have a billing-related problem."
    )
)
print(
    validate_response(
        '{"category": "shipping", "reason": "parcel"}'
    )
)
print(
    validate_response(
        "billing and fraud issue"
    )
)

# TASK 4 + 5: FULL ZERO-SHOT PIPELINE
def run_pipeline(classification_system_prompt, pipeline_token_budget=None):
    global RETRY_FAILURE_COUNT, PERMANENT_FAILURE_COUNT
    # Reset mock and retry counters once before each complete run
    mock_llm.reset()
    RETRY_FAILURE_COUNT = 0
    PERMANENT_FAILURE_COUNT = 0
    pipeline_results = []
    total_tokens_used = 0
    for _, ticket_row in ticket_data.iterrows():
        current_ticket_id = ticket_row["ticket_id"]
        current_customer_id = ticket_row["customer_id"]
        original_ticket_text = ticket_row["text"]
        pii_redacted_text = ticket_row["redacted_text"]
        predicted_category = None
        classification_status = None
        ticket_tokens_used = 0
        # INPUT GUARDRAILS
        if original_ticket_text.strip() == "":
            classification_status = "skipped_empty"
        elif mock_llm.est_tokens(original_ticket_text) > MAX_INPUT_TOKENS:
            classification_status = "flagged_long"
        # TOKEN BUDGET
        elif pipeline_token_budget is not None and total_tokens_used >= pipeline_token_budget:
            classification_status = "budget_exhausted"
        # MODEL CALL
        else:
            # Verify that redacted text contains no PII
            assert not re.search(EMAIL_PATTERN, pii_redacted_text)
            assert not re.search(PHONE_PATTERN, pii_redacted_text)
            assert not re.search(ORDER_PATTERN, pii_redacted_text)
            llm_messages = [
                {
                    "role": "system",
                    "content": classification_system_prompt
                },
                {
                    "role": "user",
                    "content": pii_redacted_text
                }
            ]
            llm_response = call_llm_with_retry(
                llm_messages,
                current_ticket_id
            )
            ticket_tokens_used = llm_response.usage.total_tokens
            total_tokens_used += ticket_tokens_used
            llm_response_text = llm_response.choices[0].message.content
            predicted_category, classification_status = validate_response(
                llm_response_text
            )
        # SAVE RESULT
        pipeline_results.append({
            "ticket_id": current_ticket_id,
            "customer_id": current_customer_id,
            "status": classification_status,
            "category": predicted_category,
            "tokens_used": ticket_tokens_used,
            "true_category": ticket_row["true_category"]
        })
    pipeline_results_df = pd.DataFrame(pipeline_results)
    print("\n========== PIPELINE RESULTS ==========")
    print(pipeline_results_df["status"].value_counts())
    print("\nTotal tickets:", len(pipeline_results_df))
    print("Total tokens:", total_tokens_used)
    classified_ticket_data = pipeline_results_df[
        pipeline_results_df["status"].isin([
            "classified_json",
            "classified_keyword"
        ])
    ]
    print("Classified tickets:", len(classified_ticket_data))
    if len(classified_ticket_data) > 0:
        classification_accuracy = (
            classified_ticket_data["category"]
            == classified_ticket_data["true_category"]
        ).mean()
        tokens_per_classified_ticket = total_tokens_used / len(classified_ticket_data)
        print("Accuracy:", classification_accuracy)
        print("Tokens per classified ticket:", round(tokens_per_classified_ticket, 1))
    else:
        print("Accuracy: N/A")
        print("Tokens per classified ticket: N/A")
    print(f"Total transient failures: {RETRY_FAILURE_COUNT}")
    print(f"Total permanent failures: {PERMANENT_FAILURE_COUNT}")
    return pipeline_results_df

# ZERO-SHOT FULL RUN
zero_shot_results = run_pipeline(
    ZERO_SHOT_PROMPT
)
print("\n========== ZERO-SHOT STATUS COUNTS ==========")
print(zero_shot_results["status"].value_counts())

# TASK 6: ZERO-SHOT CONFUSION MATRIX
classified_zero_shot_data = zero_shot_results[
    zero_shot_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]
zero_shot_confusion_matrix = pd.crosstab(
    classified_zero_shot_data["true_category"],  # type: ignore
    classified_zero_shot_data["category"]  # type: ignore
)
zero_shot_confusion_matrix = zero_shot_confusion_matrix.reindex(
    index=["billing", "delivery", "technical", "fraud"],
    columns=["billing", "delivery", "technical", "fraud"],
    fill_value=0
)
print("\n========== ZERO-SHOT CONFUSION MATRIX ==========")
print(zero_shot_confusion_matrix)
few_shot_pipeline_results = run_pipeline(FEW_SHOT_PROMPT)

# FEW-SHOT METRICS
classified_few_shot_data = few_shot_pipeline_results[
    few_shot_pipeline_results["status"].isin([
        "classified_json",
        "classified_keyword"
    ])
]
few_shot_accuracy = (
    classified_few_shot_data["category"]
    == classified_few_shot_data["true_category"]
).mean()
few_shot_total_tokens = few_shot_pipeline_results["tokens_used"].sum()
print("\n========== FEW-SHOT METRICS ==========")
print("Classified tickets:", len(classified_few_shot_data))
print("Accuracy:", few_shot_accuracy)
print("Total tokens:", few_shot_total_tokens)
print("Tokens per classified ticket:", few_shot_total_tokens / len(classified_few_shot_data))

# TASK 7A: TOKEN BUDGET DEMO
os.makedirs("outputs", exist_ok=True)
TOKEN_BUDGET = 5000
token_budget_results = run_pipeline(
    ZERO_SHOT_PROMPT,
    pipeline_token_budget=TOKEN_BUDGET
)
token_budget_results.to_csv(
    "outputs/triage_budget_demo.csv",
    index=False
)
budget_exhausted_ticket_count = (
    token_budget_results["status"] == "budget_exhausted"
).sum()
print("\n========== BUDGET DEMO ==========")
print("Token budget:", TOKEN_BUDGET)
print("Budget exhausted tickets:", budget_exhausted_ticket_count)
print("Saved: outputs/triage_budget_demo.csv")

# TASK 7B: FINAL FEW-SHOT RUN
TOKEN_BUDGET = 100000
final_triage_results = run_pipeline(
    FEW_SHOT_PROMPT,
    pipeline_token_budget=TOKEN_BUDGET
)

# CUSTOMER SEGMENT INFORMATION
customer_segment_data = pd.read_csv(
    "outputs/customer_segments.csv"
)

# Keep only the required customer information
customer_segment_data = customer_segment_data[
    ["customer_id", "segment_name", "is_premium"]
]

# TASK 7C: ROUTING
final_triage_results = final_triage_results.merge(
    customer_segment_data,
    on="customer_id",
    how="left"
)

# Missing segment customers are treated as not premium
final_triage_results["is_premium"] = (
    final_triage_results["is_premium"].fillna(False)
)

# Default routing
final_triage_results["route"] = "support_queue"
final_triage_results["priority"] = "NORMAL"

# Human review statuses
human_review_status_list = [
    "skipped_empty",
    "flagged_long",
    "flagged_invalid_output",
    "budget_exhausted"
]
human_review_ticket_mask = final_triage_results["status"].isin(
    human_review_status_list
)
final_triage_results.loc[
    human_review_ticket_mask, "route"
] = "human_review"

# Fraud tickets go to security team
fraud_ticket_mask = (
    (final_triage_results["category"] == "fraud")
    & (~human_review_ticket_mask)
)
final_triage_results.loc[
    fraud_ticket_mask, "route"
] = "security_team"
final_triage_results.loc[
    fraud_ticket_mask, "priority"
] = "HIGH"

# Premium customers go to premium desk
premium_customer_mask = (
    final_triage_results["is_premium"]
    & (~human_review_ticket_mask)
    & (~fraud_ticket_mask)
)
final_triage_results.loc[
    premium_customer_mask, "route"
] = "premium_desk"
final_triage_results.loc[
    premium_customer_mask, "priority"
] = "HIGH"

# TASK 7D: SAVE FINAL OUTPUT
final_triage_results["segment_name"] = (
    final_triage_results["segment_name"].fillna("")
)
final_triage_results["category"] = (
    final_triage_results["category"].fillna("")
)
final_triage_output = final_triage_results[
    [
        "ticket_id",
        "customer_id",
        "status",
        "category",
        "segment_name",
        "priority",
        "route",
        "tokens_used"
    ]
]
final_triage_output.to_csv(
    "outputs/triage_results.csv",
    index=False
)
print("\n========== FINAL TRIAGE OUTPUT ==========")
print("Total tickets:", len(final_triage_output))
print("\nRoute counts:")
print(final_triage_output["route"].value_counts())
print("\nPriority counts:")
print(final_triage_output["priority"].value_counts())
print("\nStatus counts:")
print(final_triage_output["status"].value_counts())
print("\nSaved: outputs/triage_results.csv")

# TASK 7E: FINAL VALIDATION
assert len(final_triage_output) == 164
assert final_triage_output["ticket_id"].is_unique
assert final_triage_output.loc[
    final_triage_output["status"].isin(human_review_status_list),
    "category"
].eq("").all()
print("\nFINAL OUTPUT VALIDATION PASSED")