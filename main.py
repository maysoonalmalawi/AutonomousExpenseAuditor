import base64
import os
import asyncio
import httpx
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Dict

# ==========================================
# STEP 1: INITIALIZATION & ENVIRONMENT SETUP
# ==========================================

# Build an absolute path to the .env file located in the same directory
# as this script to avoid relative-path bugs in production.

current_dir = Path(__file__).parent.resolve()
env_path = current_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Initialize the async OpenAI-compatible client, but pointed at Google's Gemini endpoint.
client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY")
)


def encode_image(image_path_obj):
    # This converts image files to base64.
    with open(image_path_obj, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ==========================================
# STEP 2: ENTERPRISE DATA MODELING (SCHEMAS)
# ==========================================

# Pydantic models to make sure that the model will always return valid, schema-conformant JSON

class LineItem(BaseModel):
    description: str
    amount: float

class ReceiptData(BaseModel):
    merchant: str
    date: str
    total_amount: float
    currency: str
    line_items: List[LineItem]
    category: str = Field(description="Categorize into: Travel, Meals, Software SaaS, or Office Supplies")
    confidence_score: float = Field(description="Confidence from 0.0 to 1.0 on extraction accuracy.")

class PolicyAudit(BaseModel):
    is_compliant: bool
    reasoning: str
    routing_destination: str = Field(description="Route to: 'AUTO_APPROVE', 'MANAGER_REVIEW', or 'HUMAN_AUDIT_QUEUE'")

# ==========================================
# STEP 3: RESILIENCE ENGINE (RETRY WRAPPER)
# ==========================================

async def call_llm_with_retry(api_call_func, task_name: str, max_retries=5, initial_delay=5):
    # This implements 'Exponential Backoff' for Rate Limit (429) errors.
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return await api_call_func()
        except Exception as e:
            err_msg = str(e)
            if ("503" in err_msg or "429" in err_msg) and attempt < max_retries - 1:
                print(f"   [Rate Limit/Busy] {task_name} throttled. Retrying in {delay}s...")
                await asyncio.sleep(delay)
                delay *= 2  # Exponential increase
            else:
                print(f"   [Error] {task_name} critically failed.")
                raise e # Non-recoverable error

# ==========================================
# STEP 4: DETERMINISTIC COMPUTATION ENGINE
# ==========================================

async def get_sar_exchange_rate(from_currency: str) -> float:
    # This prevents the LLM from performing hallucinated rates and keeps costs down.
    from_currency = from_currency.upper()
    if from_currency == "SAR":
        return 1.0
    try:
        async with httpx.AsyncClient() as http_client:
            response = await http_client.get(f"https://open.er-api.com/v6/latest/{from_currency}")
            if response.status_code == 200:
                return response.json()["rates"].get("SAR", 1.0)
    except Exception:
        pass
    return 1.0

# ==========================================
# STEP 5: CORE PIPELINE LOGIC (PER EXPENSE)
# ==========================================

# The function below runs the full 3-stage audit pipeline for one receipt image:
#       Stage 1 (Extraction):    LLM reads the image and returns structured ReceiptData.
#       Stage 2 (Normalization): Convert the amount to SAR using a live exchange rate.
#       Stage 3 (Compliance):    LLM checks the extracted data against corporate policy
#                                 and assigns a routing destination.

async def audit_single_expense(file_path: Path, corporate_policy: str, results_pool: Dict):
    if not file_path.exists():
        return

    image_name = file_path.name
    base64_image = encode_image(file_path)
    print(f"[Processing] Ingesting file: {image_name}")

    # Stage 1: Extraction
    async def run_extraction():
        return await client.beta.chat.completions.parse(
            model="gemini-2.5-flash",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract all receipt details strictly matching the schema."},
                    # The image is passed inline as a base64 data URI which is the standard
                    # multimodal format for vision-capable LLMs.
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]}
            ],
            response_format=ReceiptData,
        )

    extraction_completion = await call_llm_with_retry(run_extraction, f"Extraction ({image_name})")
    extracted_data = extraction_completion.choices[0].message.parsed
    print(f"   ✅ {image_name} successfully extracted.")

    # Stage 2: Normalization
    sar_rate = await get_sar_exchange_rate(extracted_data.currency)
    amount_in_sar = round(extracted_data.total_amount * sar_rate, 2)

    # Stage 3: Routing
    # Bundle the extracted data + SAR amount into a single payload for the compliance LLM.
    audit_payload = {
        "extracted_receipt": extracted_data.model_dump(), # Convert Pydantic model → plain dict
        "calculated_amount_in_sar": amount_in_sar,
        "is_low_confidence": extracted_data.confidence_score < 0.85 # Flag uncertain extractions
    }

    async def run_audit():
        return await client.beta.chat.completions.parse(
            model="gemini-2.5-flash",
            messages=[
                # The corporate policy is injected as a system prompt
                # so the model treats it as authoritative rules.
                {"role": "system", "content": f"You are a compliance router. Policy:\n{corporate_policy}"},
                {"role": "user", "content": f"Review this audit payload: {audit_payload}"}
            ],
            response_format=PolicyAudit,
        )

    audit_completion = await call_llm_with_retry(run_audit, f"Compliance Audit ({image_name})")
    audit_result = audit_completion.choices[0].message.parsed

    # Write the final result for this receipt into the shared pool
    results_pool[image_name] = {
        "merchant": extracted_data.merchant[:18],
        "category": extracted_data.category[:14],
        "amount_sar": f"{amount_in_sar:,.2f} SAR",
        "compliant": "YES" if audit_result.is_compliant else "NO",
        "routing": audit_result.routing_destination,
        "reasoning": audit_result.reasoning
    }

# ==========================================
# STEP 6: ENTRYPOINT & REPORTING ENGINE
# ==========================================

async def main():
    # Load policy from file
    policy_path = current_dir / "policy.txt"
    try:
        with open(policy_path, "r", encoding="utf-8") as f:
            corporate_policy = f.read()
    except FileNotFoundError:
        print("[Error] policy.txt not found. Using default fallback.")
        corporate_policy = "Default Policy: Expense limit 150 SAR."

    input_folder = current_dir / "receipts"

    if not input_folder.exists():
        input_folder.mkdir(parents=True, exist_ok=True)
        return

    receipt_files = list(input_folder.glob("*.[jJ][pP]*[gG]"))

    print(f"[Batch Engine] Found {len(receipt_files)} files. Starting execution...")
    start_time = time.time()

    # A shared dict that all concurrent tasks write their results into.
    shared_results_pool = {}
    receipt_tasks = []

    for file_path in receipt_files:
        receipt_tasks.append(audit_single_expense(file_path, corporate_policy, shared_results_pool))
        await asyncio.sleep(1.5) # Throttling to stay under API request limits

    # Run all receipt tasks concurrently.
    # return_exceptions=True means one failing task won't cancel the rest.
    await asyncio.gather(*receipt_tasks, return_exceptions=True)

    # --- TABLE RENDERING ENGINE ---
    print("\n" + "=" * 110)
    print("                         ENTERPRISE COMPLIANCE AUDIT OVERVIEW SUMMARY                         ")
    print("=" * 110)
    header_fmt = "  {:<15} | {:<18} | {:<14} | {:<14} | {:<9} | {:<20}"
    print(header_fmt.format("FILE NAME", "MERCHANT", "CATEGORY", "AMOUNT (SAR)", "COMPLIANT", "ROUTING TRACK"))
    print("-" * 110)

    # Sort by filename for deterministic output order regardless of async completion order.
    for file_path in sorted(receipt_files):
        filename = file_path.name
        stats = shared_results_pool.get(filename)
        if stats:
            print(header_fmt.format(filename, stats["merchant"], stats["category"], stats["amount_sar"],
                                    stats["compliant"], stats["routing"]))
        else:
            # If the receipt failed all retries, output a clear error row instead of silently skipping.
            print(f"  {filename:<15} | ERROR: Quota Exhausted")

    print("=" * 110 + f"\nProcessing finished in {round(time.time() - start_time, 2)}s.")

if __name__ == "__main__":
    # Standard async entrypoint that creates a new event loop then cleanly shuts it down after completion.
    asyncio.run(main())