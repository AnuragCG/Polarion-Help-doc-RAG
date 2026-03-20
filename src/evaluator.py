
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------
# OpenAI client
# -----------------------------
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Load dataset
# -----------------------------
def load_eval_dataset(path="data/eval_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------
# Strict Recall@K (page-based)
# -----------------------------
def recall_at_k(retrieved_pages, expected_pages, k=3):
    top_k = retrieved_pages[:k]
    return int(any(p in top_k for p in expected_pages))


# -----------------------------
# Relaxed Recall (content-based)
# -----------------------------
def relaxed_recall(retrieved_items, keywords):
    for item in retrieved_items:
        text = item["document"].lower()
        if all(k.lower() in text for k in keywords):
            return 1
    return 0


# -----------------------------
# Keyword score (baseline)
# -----------------------------
def keyword_match_score(answer, keywords):
    answer = answer.lower()
    return sum(1 for k in keywords if k.lower() in answer) / len(keywords)


# -----------------------------
# LLM-based Faithfulness
# -----------------------------
def llm_faithfulness(context: str, answer: str) -> int:
    """
    1 → supported by context
    0 → not supported (hallucination)
    """

    prompt = f"""
You are a strict evaluator.

Context:
{context}

Answer:
{answer}

Task:
Is the answer fully supported by the context?

Rules:
- Do NOT use outside knowledge
- Be strict
- If even partially unsupported → UNSUPPORTED

Return ONLY one word:
SUPPORTED
or
UNSUPPORTED
"""

    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    verdict = response.choices[0].message.content.strip().upper()

    return 1 if "SUPPORTED" in verdict else 0


# -----------------------------
# Main evaluation
# -----------------------------
def evaluate_pipeline(pipeline_fn):
    dataset = load_eval_dataset()

    total_recall = 0
    total_relaxed = 0
    total_keyword_score = 0
    total_faithfulness = 0

    for item in dataset:
        query = item["question"]
        expected_pages = item["expected_pages"]
        keywords = item["keywords"]

        print("\n" + "=" * 80)
        print(f"Evaluating: {query}")
        print("=" * 80)

        # -----------------------------
        # Retrieval
        # -----------------------------
        context, retrieved_items = pipeline_fn(query)

        retrieved_pages = [r["page"] for r in retrieved_items]

        # -----------------------------
        # Metrics
        # -----------------------------
        recall = recall_at_k(retrieved_pages, expected_pages)
        relaxed = relaxed_recall(retrieved_items, keywords)

        total_recall += recall
        total_relaxed += relaxed

        # -----------------------------
        # Generate answer
        # -----------------------------
        from rag import generate_answer
        answer = generate_answer(query, context)

        # -----------------------------
        # Keyword score (baseline)
        # -----------------------------
        keyword_score = keyword_match_score(answer, keywords)
        total_keyword_score += keyword_score

        # -----------------------------
        # 🔥 LLM Faithfulness
        # -----------------------------
        faithfulness = llm_faithfulness(context, answer)
        total_faithfulness += faithfulness

        # -----------------------------
        # Print results
        # -----------------------------
        print(f"Retrieved Pages: {retrieved_pages}")
        print(f"Expected Pages: {expected_pages}")
        print(f"Recall@3: {recall}")
        print(f"Relaxed Recall: {relaxed}")
        print(f"Keyword Score: {keyword_score:.2f}")
        print(f"Faithfulness (LLM): {faithfulness}")

        print("\nAnswer:")
        print(answer)

    # -----------------------------
    # Final summary
    # -----------------------------
    n = len(dataset)

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Avg Recall@3: {total_recall / n:.2f}")
    print(f"Avg Relaxed Recall: {total_relaxed / n:.2f}")
    print(f"Avg Keyword Score: {total_keyword_score / n:.2f}")
    print(f"Avg Faithfulness: {total_faithfulness / n:.2f}")

