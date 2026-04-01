"""
Main entry point for modularized RAG system.
Replaces original app.py with production architecture.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_settings
from src.data import DocumentLoader
from src.rag import VectorstoreManager
from src.agents import RAGChain
from src.eval import RAGEvaluator, TESLA_10Q_DATASET


def setup_vectorstore():
    """Initialize vectorstore with documents."""
    print("📚 Setting up vectorstore...")
    
    settings = get_settings()
    loader = DocumentLoader()
    
    # Load PDF
    print(f"📄 Loading PDF from {settings.PDF_PATH}...")
    documents = loader.load_and_split_pdf()
    print(f"✅ Loaded and split into {len(documents)} chunks")
    
    # Create vectorstore
    print("🗂️  Creating vectorstore...")
    vs_manager = VectorstoreManager()
    vectorstore = vs_manager.create_vectorstore(documents, force_recreate=True)
    print(f"✅ Vectorstore created with {vectorstore._collection.count()} documents")
    
    return vectorstore


def run_evaluation():
    """Run RAGAS evaluation on Tesla 10-Q dataset."""
    print("\n🚀 Running RAGAS Evaluation...")
    
    # Load vectorstore
    vs_manager = VectorstoreManager()
    vectorstore = vs_manager.load_vectorstore()
    
    # Create RAG chain
    chain = RAGChain(retriever=vectorstore.as_retriever(search_kwargs={"k": 4}))
    
    # Generate answers
    print("⏳ Generating answers for test questions...")
    answers = []
    for question in TESLA_10Q_DATASET.questions:
        try:
            answer = chain.invoke(question)
            answers.append(answer)
        except Exception as e:
            print(f"⚠️  Error processing '{question}': {e}")
            answers.append("Error generating answer")
    
    # Evaluate
    evaluator = RAGEvaluator()
    eval_result = evaluator.evaluate(
        questions=TESLA_10Q_DATASET.questions,
        answers=answers,
        contexts=TESLA_10Q_DATASET.contexts,
        ground_truths=TESLA_10Q_DATASET.ground_truths,
    )
    
    print(eval_result)
    
    # Quality gates
    gates = evaluator.check_quality_gates(eval_result)
    print("\nQuality Gates:")
    for gate, passed in gates.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {gate}: {status}")
    
    return eval_result


def interactive_mode():
    """Run interactive chat mode."""
    print("\n💬 Interactive Mode")
    print("Type 'exit' to quit\n")
    
    # Load vectorstore
    vs_manager = VectorstoreManager()
    vectorstore = vs_manager.load_vectorstore()
    
    # Create chain
    chain = RAGChain(retriever=vectorstore.as_retriever(search_kwargs={"k": 4}))
    
    while True:
        question = input("You: ").strip()
        
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            answer = chain.invoke(question)
            print(f"\n🤖 Assistant: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    """Main entry point."""
    print("=" * 60)
    print("🚗 Modularized Automotive RAG System")
    print("=" * 60)
    
    settings = get_settings()
    print(f"📍 Environment: {settings.ENV}")
    print(f"📍 Primary LLM: {settings.PRIMARY_LLM.upper()}")
    print(f"📍 Embeddings Model: {settings.EMBEDDING_MODEL}")
    
    try:
        vectorstore = setup_vectorstore()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"⚠️  Ensure PDF exists at: {settings.PDF_PATH}")
        sys.exit(1)
    
    # Run evaluation
    try:
        eval_result = run_evaluation()
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        import traceback
        traceback.print_exc()
    
    # Interactive mode
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")


if __name__ == "__main__":
    main()
