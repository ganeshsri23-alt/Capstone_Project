# Instamart Customer Intelligence and Support Copilot
## Overview
An offline educational project using synthetic data and a mock LLM (no real payments, no hosted API). All amounts are in INR.

- **Part A:** audit, clean, store (CSV + SQLite), SQL queries, pandas analysis.
- **Part B:** customer segmentation and lapse prediction.
- **Part C:** support-ticket triage with PII redaction, retries, validation and routing.
- **Part D:** policy RAG assistant and a bounded refund agent.

```text
Capstone_Project/
├── data/            (raw/, clean/, policies/, instamart.db)
├── outputs/
├── state/
├── sql/queries.sql
├── src/             (data_pipeline, store_validate, run_sql, eda_analysis,
│                     customer_segmentation, ticket_triage, instamart)
├── generate_data.py
├── mock_llm.py
├── .gitignore
├── README.md
└── requirements.txt
```

Requirements: Python **3.11+**, and the packages in `requirements.txt` (pandas, NumPy, scikit-learn, joblib).

## Reproduce
Run from the repository root, each step after the previous one succeeds.

1. Create the virtual environment: `python -m venv .venv`
2. Activate it: `.\.venv\Scripts\Activate.ps1` (macOS/Linux: `source .venv/bin/activate`)
3. Upgrade pip: `python -m pip install --upgrade pip`
4. Install dependencies: `pip install -r requirements.txt`
5. Generate raw data and policies: `python generate_data.py`
6. Data pipeline: `python src/data_pipeline.py`
7. Store and validate: `python src/store_validate.py`
8. Run the seven SQL queries: `python src/run_sql.py`
9. Pandas EDA and cross-checks: `python src/eda_analysis.py`
10. Segmentation and lapse prediction: `python src/customer_segmentation.py`
11. Ticket triage: `python src/ticket_triage.py`
12. RAG assistant and refund agent: `python src/instamart.py`

Expected generator output: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`

## Part A: Data Foundation
### Raw data audit
| Dataset | Shape | Columns and data types |
|---|---:|---|
| Customers | 1,200 × 5 | `customer_id`, `city`, `signup_date`, `gender` (string); `age` (float64) |
| Products | 40 × 4 | `product_id`, `product_name`, `category` (string); `price_inr` (int64) |
| Orders | 13,084 × 9 | `order_id`, `customer_id`, `product_id`, `order_ts`, `status` (string); `quantity`, `discount_pct` (int64); `order_amount` (text, some non-numeric); `delivery_minutes` (float64) |

**Missing values:** customers `age` 61 (5.08%) and `gender` 30 (2.50%), all other customer columns 0 (0%). Products: 0 (0%) in every column. Orders `order_amount` 53 blank (0.41%) and `delivery_minutes` 2,541 (19.42%), all other order columns 0 (0%).

| Audit check | Result |
|---|---:|
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct `order_id` values | 12,309 |
| Non-numeric or blank order amounts | 120 |
| Negative delivery times | 106 |
| Customer ages outside 10–100 | 6 |
| Raw city spellings | 24 |
| Raw gender spellings (non-blank) | 10 |

### Cleaning rules
**Customers:** trimmed and title-cased cities (476 changed); standardised genders (920 changed); 30 blank genders set to `unknown`; 6 out-of-range and 61 blank ages set to missing (67 total) and filled with the city median; `signup_date` converted to datetime.

**Orders:** parsed all 13,084 `order_ts`; removed 119 exact duplicates; kept the latest record per `order_id` (656 older records superseded, so 0 pending rows remain); repaired 112 bad amounts as price × quantity × (1 − discount/100); set 105 negative delivery times to missing; filled 774 missing delivery times for completed orders with the category median; cleared delivery time on cancelled orders (0 remain).

### Validation checkpoints
| Checkpoint | Result |
|---|---:|
| Cities | 6 |
| Gender counts | 610 / 560 / 30 |
| Age range | 18–60 |
| Clean orders | 12,309 |
| Completed / cancelled | 11,110 / 1,199 |
| Completed order amount total | ₹2,998,668.80 |

The clean tables (`customers`, `products`, `orders`) are stored in `data/instamart.db`; `order_ts` is saved as text in `YYYY-MM-DD HH:MM:SS` format. `store_validate.py` runs 15 assertions and prints `ALL CHECKPOINTS PASSED`.

### SQL results
| Query | Result |
|---|---|
| Q1 | 30 customers |
| Q2 | 30 rows; grand total ₹2,998,668.80; largest cell Bengaluru / Electronics ₹233,909.45 |
| Q3 | 11 products |
| Q4 | 25 customers |
| Q5 | 550 customers above the average spend of ₹2,552.06 |
| Q6 | Top per city: Bengaluru C0082 ₹12,856.20; Chennai C0930 ₹10,151.40; Delhi C0291 ₹13,742.60; Hyderabad C0743 ₹12,730.05; Mumbai C0693 ₹14,766.55; Pune C0426 ₹10,377.10 |
| Q7 | 183 daily rows; final cumulative total ₹2,998,668.80 |

For Q6, 8 rows differ between `RANK()` and `ROW_NUMBER()` because tied values share a rank under `RANK()` but get unique numbers under `ROW_NUMBER()`.

### Pandas cross-check and EDA
The pivot-table grand total matches the SQL total (₹2,998,668.80).

| City | Sum (₹) | Mean (₹) | Orders |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

- Order amount: mean ₹269.91, median ₹135.00, skew about 4.61.
- Quartiles: Q1 ₹64.80, Q3 ₹297.00, IQR ₹232.20; fences −₹283.50 and ₹645.30. There are 897 outliers above the upper fence and none below the lower one.
- Correlations: amount–discount −0.0951; amount–quantity +0.0873; amount–delivery −0.0031; discount–quantity −0.0004; discount–delivery −0.0098; quantity–delivery −0.0126.

**Interpretation:** the data describes synthetic customers, products and orders. Order amounts are strongly right-skewed and the middle 50% lies between ₹64.80 and ₹297.00. There are 897 high-value outliers and no low ones. Relationships are weak; the strongest pair is amount and discount (−0.0951). Small correlations do not show causation.

## Part B: Customer Segments and Lapse Prediction
### RFM and K-means
RFM uses completed orders with snapshot date **2025-10-01**: recency is days since the last order, frequency is distinct orders, monetary is total spend. Features were standardised before K-means.

The RFM table has 1,175 customers. Recency: median 19, min 0, max 180 days. Frequency: mean 9.46, max 33. Monetary: mean ₹2,552.06, max ₹14,766.55.

K-means ran for K = 2 to 8 (`random_state=42`, `n_init=10`):

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

Inertia can only fall as K grows (1,820.8 to 366.5), so it cannot pick K alone. **K = 4** was chosen: it has the highest silhouette among K=3, 4 and 5 (0.526), the curve bends there (the drop from K=4 to 5 is about 126 versus about 525 from K=3 to 4), and it gives four interpretable groups.

### Segments
Names come from the cluster profiles (highest mean monetary = Premium; recency above average = Inactive; frequency above average = Frequent; the rest = Occasional). Cluster numbers are only labels.

| Segment | Size | Recency | Frequency | Monetary (₹) | Action |
|---|---:|---:|---:|---:|---|
| Premium | 138 | 24.12 | 6.77 | 7,129.70 | Protect loyalty with priority service and rewards |
| Inactive | 182 | 109.04 | 2.20 | 685.99 | Targeted win-back campaign |
| Frequent | 419 | 8.77 | 19.40 | 3,410.02 | Convenient reorder offers |
| Occasional | 436 | 28.86 | 3.77 | 1,057.62 | Low-friction reminders and bundles |

PCA explained variance: 0.634 and 0.218, total **0.852** for two components.

### Lapse cohort and split
The cohort has 1,156 customers; 443 lapsed (38.32%). The lapse window is 2025-09-01 to 2025-10-01.

| Split | Customers | Not lapsed | Lapsed |
|---|---:|---:|---:|
| Train | 924 | 570 | 354 |
| Test | 232 | 143 | 89 |

No orders from 2025-09-01 onward were used as features, and segment labels were not features. The scaler learned a frequency mean of 8.115 (versus 8.029 for the full cohort) because it was fitted on the 924 training rows only. The encoder, scaler and resampler never saw test rows.

### Models
Logistic Regression used `GridSearchCV` over `C ∈ {0.01, 0.1, 1, 10, 100}`. Random Forest (`n_estimators=100`, `random_state=42`) searched `max_depth ∈ {3, 5, 8}` and `min_samples_leaf ∈ {5, 10, 20}` (45 fits). Both used `cv=5` and `scoring="f1"`.

| Model | Best parameters | CV F1 | Accuracy | Precision | Recall | F1 | Confusion matrix |
|---|---|---:|---:|---:|---:|---:|---|
| Baseline | — | — | 0.616 | — | — | 0.000 | [[143, 0], [89, 0]] |
| Logistic Regression | C = 100 | 0.651 | 0.728 | 0.64 | 0.65 | 0.648 | [[111, 32], [31, 58]] |
| Random Forest | depth 8, leaf 5 | 0.690 | 0.685 | 0.59 | 0.61 | 0.597 | [[105, 38], [35, 54]] |

The baseline predicts nobody as lapsed, so it misses all 89 lapsed customers despite 0.616 accuracy. **Logistic Regression** is selected because it has the higher lapsed F1 (0.648 vs 0.597) and higher accuracy. Missing a lapsing customer costs future revenue, while targeting a loyal customer wastes campaign spend, so the threshold should be tuned to campaign capacity and these costs.

### Oversampling, features and retention list
Oversampling changed precision 0.644 → 0.589, recall 0.652 → 0.854 and F1 0.648 → 0.697. Oversampling raises recall but lowers precision: more false positives in exchange for catching more lapsed customers. For the retention team this means more customers get contacted, some unnecessarily, but far fewer lapsing customers are missed.

| Feature | RF importance | LR sign | Reading |
|---|---:|:---:|---|
| `frequency` | 0.268 | − | More orders go with lower lapse risk (habit) |
| `monetary` | 0.242 | + | Association only; may reflect related features |
| `recency` | 0.113 | + | Longer since last order, higher risk |
| Avg delivery time | 0.065 | + | Association, not proof of cause |
| Cancellation rate | 0.057 | − | Small and unexpected; not a causal claim |

RF importance and LR signs measure different things, and neither shows causation.

The tuned Logistic Regression was saved to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and reproduced the original test predictions exactly.

The retention list has 10 customers, of whom 3 had actually lapsed (example: C0525, probability 0.60469, `actually_lapsed = 1`). **Caution:** `is_premium` is a post-hoc flag from whole-window RFM, which includes September (the lapse period). It is not leakage-safe, was not used as a model feature, and the retention list is illustrative only.

## Part C: Support Ticket Triage
The dataset has 164 tickets. The pipeline handles empty and long tickets, redacts PII, retries transient failures, validates output and routes tickets.

Redactions: 55 emails, 91 phone numbers and 132 order IDs (**278 total**). Re-applying the three patterns to the redacted text found 0 emails, 0 phones and 0 order IDs (checked by assertions).

Each long ticket estimates 734 tokens against a 400-token limit. There were 4 empty and 4 long tickets; of the other 156, 147 were classified and 9 were flagged for review.

Statuses: `classified_json` 135, `classified_keyword` (fallback) 12, `flagged_invalid_output` 9, `skipped_empty` 4, `flagged_long` 4.

Routes: `support_queue` 102, `security_team` 34, `human_review` 17, `premium_desk` 11 (total 164).

Rule-validated outputs are the schema-checked `classified_json` replies and the accepted keyword fallbacks. Invalid outputs, plus empty and long tickets, go to human review or are skipped instead of being accepted silently.

### Zero-shot vs few-shot
| Prompt | Accuracy | Tokens per ticket |
|---|---:|---:|
| Zero-shot | 85.7% | 70.3 |
| Few-shot | 100.0% | 123.4 |

Zero-shot confusion matrix (rows true, columns predicted: billing, delivery, technical, fraud):

| True \ Pred | Billing | Delivery | Technical | Fraud |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Technical | 10 | 0 | 29 | 0 |
| Fraud | 11 | 0 | 0 | 23 |

Zero-shot got 126 of 147 right; all 21 errors were technical or fraud tickets labelled billing, including 11 fraud tickets that could have missed the security team. Few-shot reached 100% for about 53 extra tokens per ticket. The extra cost is worth it when a wrong label is expensive, though this synthetic result does not guarantee real-world performance.

### Retries and budget
- 19 transient failures were all recovered by retry; 0 permanent failures.
- All retry waits were within 0.045–0.0575 seconds.
- With a budget of 5,000 tokens, 80 tickets ended as `budget_exhausted`.
- Total tokens: zero-shot 10,336; few-shot 18,136.

## Part D: Policy RAG Assistant and Refund Agent
### Retrieval
The policy corpus has 15 chunks (refund 4, delivery 3, escalation 3, payment 3, privacy 2). For the query "refund policy late delivery", the top three chunks all came from `refund_policy.txt`.

TF measures how often a term appears in a chunk; IDF weights terms found in fewer chunks more. TF-IDF is lexical, so different wording can hide a relevant passage.

Normal chunks: 12/12 documents, 12/12 fact checks, 2/2 abstention. 40-word zero-overlap chunks: 12/12 documents, 10/12 fact checks, 2/2 abstention.

With 40-word chunks and no overlap (13 chunks), retrieval stayed 12/12 but fact checks fell to 10/12 because facts split across a chunk boundary were lost. The delivery answer returned only "receives a full refund of the order amount." because "more than 45 minutes" was cut off, and the payment answer lost "48 hours".

### Seven refund-agent scenarios
| Scenario | Result | Calls | Tokens |
|---|---|---:|---:|
| S1 | `REFUND` ₹128.25 | 5 | 972 |
| S2 | `REFUND` ₹135.00 | 5 | 972 |
| S3 | `ESCALATE` ₹1,999.00 | 5 | 959 |
| S4 | `DENIED` | 4 | 665 |
| S5 | `DENIED` | 3 | 382 |
| S6 | `NOT_FOUND` | 2 | 102 |
| S7 | `FALLBACK_MAX_ITERATIONS` | 6 | 3,090 |

S1 with `token_limit=200` ended in `FALLBACK_TOKEN_LIMIT` after 3 calls. With `time_limit_seconds=0` it ended in `FALLBACK_TIMEOUT` with 0 calls.

### Crash and resume
The run did `get_order`, `search_policy`, then a simulated crash, and resumed with `calculate_refund` and `issue_refund`. No tool was repeated, and it finished with `REFUND` ₹128.25.

### ReAct vs plan-and-execute
Only ReAct was implemented and measured (S1: 5 calls / 972 tokens, S5: 3 / 382, S6: 2 / 102). It picks each action from the latest observation, so a `NOT_FOUND` or cancelled order stops the refund path safely.

Plan-and-execute writes a sequence first, which is easier to inspect and audit. As an estimate, not a measurement, S1 might take about 2 calls (one plan, one answer) and S6 about 3 because the plan would need revising after `NOT_FOUND`. I chose ReAct for this small bounded agent because it reacts directly to observations while keeping iteration, time and token limits. Plan-and-execute may suit longer workflows with stable steps and a reviewable plan.

## Limitations and Next Steps
- All data is synthetic and the LLM is mocked, so results are educational, not production evidence.
- TF-IDF is lexical and chunk boundaries can split facts.
- The refund agent is bounded, with fallbacks for token, time and iteration limits.
- The premium flag is post-hoc whole-window RFM, so the retention list is illustrative, not a leakage-free evaluation.
- Next steps: test on real data, monitor drift, and improve semantic retrieval and chunk handling.
