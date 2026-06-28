import argparse
import json
import gzip
import csv
import sys
import os
from rank_core import evaluate_candidates, generate_custom_reasoning

# Static Job Description text for TF-IDF matching
JD_TEXT = """
Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid)
Skills required:
- Production experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5)
- Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Strong Python, hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B testing)
Nice to have:
- LLM fine-tuning experience (LoRA, QLoRA, PEFT)
- Experience with learning-to-rank models (XGBoost-based or neural)
- Prior exposure to HR-tech, recruiting tech, or marketplace products
- Background in distributed systems or large-scale inference optimization
- Open-source contributions in the AI/ML space
"""

def main():
    parser = argparse.ArgumentParser(description="Rank candidates for Founding Senior AI Engineer role.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to output submission.csv")
    
    args = parser.parse_args()
    
    candidates_path = args.candidates
    out_path = args.out
    
    if not os.path.exists(candidates_path):
        print(f"Error: Candidate file not found at {candidates_path}")
        sys.exit(1)
        
    print(f"Loading candidates from {candidates_path}...")
    candidates = []
    
    open_func = gzip.open if candidates_path.endswith(".gz") else open
    
    # Read candidate profiles
    try:
        with open_func(candidates_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                candidates.append(json.loads(line))
    except Exception as e:
        print(f"Error loading candidates: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(candidates)} candidates. Evaluating fit...")
    
    # Run the core evaluation logic
    scored_list = evaluate_candidates(candidates, JD_TEXT)
    
    # Filter candidates with score > 0 (valid candidates only)
    valid_scored = [item for item in scored_list if item[1] > 0]
    
    print(f"Found {len(valid_scored)} valid candidates. Sorting and selecting top 100...")
    
    # Sort:
    # 1. By score descending
    # 2. By candidate_id ascending (deterministic tie-breaker per challenge rules)
    valid_scored.sort(key=lambda x: (-x[1], x[0]))
    
    top_100 = valid_scored[:100]
    
    # Generate final shortlist rows
    shortlist = []
    for rank_idx, (cid, score, _, c) in enumerate(top_100):
        rank = rank_idx + 1
        # Generate custom reasoning tailored to the rank and candidate's details
        reasoning = generate_custom_reasoning(c, rank, score)
        shortlist.append({
            "candidate_id": cid,
            "rank": rank,
            "score": round(score, 3),
            "reasoning": reasoning
        })
        
    # Write to CSV
    print(f"Writing shortlist to {out_path}...")
    try:
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(["candidate_id", "rank", "score", "reasoning"])
            for row in shortlist:
                writer.writerow([row["candidate_id"], row["rank"], row["score"], row["reasoning"]])
    except Exception as e:
        print(f"Error writing CSV: {e}")
        sys.exit(1)
        
    print("Done! Shortlist generated successfully.")

if __name__ == "__main__":
    main()
