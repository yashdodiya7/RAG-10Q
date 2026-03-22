Start with the financial SEC 10-Q RAG analyzer using public EDGAR data—it's beginner-friendly, highly relevant to jobs, and has ready tutorials/repos. The full dev flow fits in a weekend MVP: setup → data → pipeline → eval → demo. Use free local tools (LangChain, FAISS, Ollama) for your WSL/Ubuntu setup. [harini](https://harini.blog/2024/12/15/building-a-10-q-analyzer-extracting-financial-insights-with-ai/)

## Environment Setup
Install via pip (new venv):
```
pip install langchain langchain-community faiss-cpu sentence-transformers ragas streamlit ollama datasets
```
Download Ollama + Llama3: `ollama pull llama3`. Clone a starter repo like wisemachine/pb-rag-10q-analyzer or narayana-3005/RAG-based-Financial-Analysis-Chatbot for structure. [github](https://github.com/mrizwanakram/RAG-Application-using-LangChain-OpenAI-and-FAISS)

## Step 1: Data Ingestion
Download 10-Q PDFs from SEC EDGAR (e.g., Tesla/NVIDIA Q2 2025 via sec.gov/edgar). Use `datasets` HF lib or direct:
```
from datasets import load_dataset
dataset = load_dataset("docugami/kg-rag-datasets")  # Or "financial_phrasebank" for sentiment
```
Chunk PDFs (RecursiveCharacterTextSplitter, 500 chars/overlap 50). Embed with `HuggingFaceEmbeddings('all-MiniLM-L6-v2')`. Build FAISS index. [github](https://github.com/wisemachine/pb-rag-10q-analyzer)

## Step 2: RAG Pipeline
```
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

llm = OllamaLLM(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever(search_kwargs={"k":5}))
```
Query: "What are Tesla's Q2 risks?" → Retrieve top-5 chunks → Generate grounded answer. Add hybrid (BM25 + semantic) for +10-15% recall. [docs.langchain](https://docs.langchain.com/oss/python/langchain/rag)

## Step 3: Evaluation
Use Ragas (gold standard, auto-generates test cases):
```
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

result = evaluate(test_dataset, metrics=[faithfulness, answer_relevancy, context_precision])
# E.g., 0.85 faithfulness, plot in notebook
```
Test on 50-200 queries from dataset (or generate via LLM). Ablate: baseline vs reranking. Log to MLflow/WandB (free tier). [docs.ragas](https://docs.ragas.io/en/stable/concepts/components/eval_sample/)

## Step 4: Demo & Polish
Streamlit app: Chat interface + eval dashboard (metrics table/chart). Dockerize. GitHub README: Problem, arch diagram, results (e.g., "82% relevancy vs 65% naive LLM"), video demo. [github](https://github.com/mrizwanakram/RAG-Application-using-LangChain-OpenAI-and-FAISS)

## Timeline & Tips
| Day | Focus | Output |
|-----|-------|--------|
| 1 | Setup + Data/Pipeline | Working QA chain |
| 2 | Eval + Ablations | Metrics report/table |
| 3 | Streamlit + Deploy | GitHub repo ready |

Prioritize eval report—it screams "production thinker" to recruiters. Track compute (free Colab/Runpod trial if needed). [runpod](https://www.runpod.io/articles/guides/deploying-rag-pipelines-faiss-langchain-cloud-gpu)

What dataset/ticker (e.g., Tesla 10-Q) or metric focus first?