from evaluator import evaluate_pipeline
from rag import retrieve_context, initialize_retrieval

if __name__ == "__main__":
    initialize_retrieval()  
    evaluate_pipeline(retrieve_context)