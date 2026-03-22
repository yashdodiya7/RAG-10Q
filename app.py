# Simple RAG 10-Q Analyzer - All-in-one file

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper, llm_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from litellm import OpenAI as LiteLLMClient

import os


def load_dotenv(dotenv_path: str = ".env"):
    """Load simple KEY=VALUE pairs from a .env file into os.environ."""
    if not os.path.exists(dotenv_path):
        return

    with open(dotenv_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_dotenv()

def format_docs(docs):  # Add this function
    return "\n\n".join(doc.page_content for doc in docs)

# Configuration
PDF_PATH = "./data/10q.pdf"  # Place your 10-Q PDF here
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
MODEL_NAME = "llama3:8b"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# dataset samples for testing
test_questions = [
        'Which company filed the Form 10-K annual report for the fiscal year ended December 31, 2025?',
        'What is the Commission File Number listed in the document?',
        'Where is the principal executive office of the registrant located?',
        'What is the telephone number of the registrant?',
        'What is the trading symbol for the common stock?'
    ]
test_ground_truths = ["Tesla, Inc.", "001-34756", "1 Tesla Road, Austin, Texas 78725", "(512) 516-8177", "TSLA"]
data_samples = {
    'question': [
        'Which company filed the Form 10-K annual report for the fiscal year ended December 31, 2025?',
        'What is the Commission File Number listed in the document?',
        'Where is the principal executive office of the registrant located?',
        'What is the telephone number of the registrant?',
        'What is the trading symbol for the common stock?',
        'On which exchange is the common stock registered?',
        'Is the registrant indicated as a well-known seasoned issuer?',
        'What state is listed as the jurisdiction of incorporation or organization?',
        'Are there any securities registered pursuant to Section 12(g) of the Act?',
        'Is the registrant required to file reports pursuant to Section 13 or 15(d)?'
    ],
    'answer': [
        'Tesla, Inc. filed the Form 10-K annual report for the fiscal year ended December 31, 2025.',
        'The Commission File Number listed is 001-34756.',
        'The principal executive office is located at 1 Tesla Road, Austin, Texas 78725.',
        'The registrant’s telephone number is (512) 516-8177.',
        'The trading symbol for the common stock is TSLA.',
        'The common stock is registered on The Nasdaq Global Select Market.',
        'Yes, the registrant is indicated as a well-known seasoned issuer.',
        'The jurisdiction of incorporation or organization listed is Texas.',
        'No, there are no securities registered pursuant to Section 12(g) of the Act.',
        'Yes, the registrant is required to file reports pursuant to Section 13 or 15(d).'
    ],
    'contexts': [
        [
            'FORM 10-K ... For the fiscal year ended December 31, 2025 ... Tesla, Inc. (Exact name of registrant as specified in its charter)'
        ],
        [
            'Commission File Number: 001-34756'
        ],
        [
            '1 Tesla Road Austin, Texas 78725 (Address of principal executive offices) (Zip Code)'
        ],
        [
            '(512) 516-8177 (Registrant’s telephone number, including area code)'
        ],
        [
            'Common stock TSLA The Nasdaq Global Select Market'
        ],
        [
            'Title of each class Trading Symbol(s) Name of each exchange on which registered Common stock TSLA The Nasdaq Global Select Market'
        ],
        [
            'Indicate by check mark whether the registrant is a well-known seasoned issuer ... Yes x No o'
        ],
        [
            'Texas 91-2197729 (State or other jurisdiction of incorporation or organization)'
        ],
        [
            'Securities registered pursuant to Section 12(g) of the Act: None'
        ],
        [
            'Indicate by check mark if the registrant is not required to file reports pursuant to Section 13 or 15(d) of the Act. Yes o No x'
        ]
    ],
    'ground_truth': [
        'Tesla, Inc.',
        '001-34756',
        '1 Tesla Road, Austin, Texas 78725',
        '(512) 516-8177',
        'TSLA',
        'The Nasdaq Global Select Market',
        'Yes',
        'Texas',
        'None',
        'Yes'
    ]
}

def main():

    if not GEMINI_API_KEY:
        raise ValueError("Set GEMINI_API_KEY in your environment before running this script.")

    print("🚀 Starting RAG 10-Q Analyzer...")

    # Step 1: Load PDF
    print(f"📄 Loading PDF from {PDF_PATH}...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    
    # Step 2: Split into chunks
    print("✂️  Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings and vector store
    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("🧠 Creating Chroma vector store...")
    PERSIST_DIR = "./chroma_10q_db"
    vectorstore = Chroma.from_documents(
        chunks, 
        embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="10q_filings"
    )
    print("✅ Persistent Chroma DB ready!")

    # getting chunks
    def show_first_chunks(vectorstore, num_chunks=2):
        """
        Show the first N chunks from your vectorstore.
        Use these to manually write good questions + ground_truths.
        """
        # Get ALL documents from collection (first 2)
        all_docs = vectorstore.get()["documents"][:num_chunks]
        
        print("📄 FIRST 2 CHUNKS FOR ANALYSIS:\n")
        for i, doc in enumerate(all_docs, 1):
            print(f"🔹 CHUNK {i} ({len(doc)} chars):")
            print("-" * 80)
            print(doc[:2000] + "..." if len(doc) > 2000 else doc)
            print("-" * 80)
            print("\n" + "="*80 + "\n")
    
    show_first_chunks(vectorstore, num_chunks=2)

    # Step 4: Setup RAG chain
    print("⚙️  Setting up modern LCEL RAG pipeline...")
    
    llm = ChatGoogleGenerativeAI(google_api_key=GEMINI_API_KEY, model="gemini-2.0-flash")

    prompt = PromptTemplate.from_template(
        "Answer using ONLY this context (be concise):\n\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 10}) | format_docs, 
        "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ Modern LCEL RAG ready (faster than RetrievalQA!)")

    answers = []
    contexts = []
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    for q in test_questions:
        docs = retriever.invoke(q)  # your vectorstore retriever
        contexts.append([d.page_content for d in docs])
        answers.append(rag_chain.invoke(q))  # your LCEL chain
    
    dataset_dict = {
        "question": test_questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": test_ground_truths,
    }
    ragas_ds = Dataset.from_dict(dataset_dict)
    judge_llm_client = LiteLLMClient(api_key=GEMINI_API_KEY, model="gemini-2.0-flash")
    judge_llm = llm_factory("gemini-2.0-flash", client=judge_llm_client, adapter="litellm")
    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
    )
    print(result)
    
    # Step 5: Interactive Q&A
    print("\n💬 Ask questions about the 10-Q (type 'quit' to exit):\n")
    while True:
        query = input("Question: ")
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        print("🤔 Thinking...")
        response = rag_chain.invoke(query)
        print(f"\n📝 Answer: {response}\n")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(PDF_PATH):
        print(f"❌ PDF not found at {PDF_PATH}")
        print("Download sample 10-Q: https://www.sec.gov/Archives/edgar/data/.../10q.pdf")

    main()