import os
import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# Load environment variables from .env file
# load_dotenv()
# === Setup: Local Embedding ===
local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Initialize the Chroma client with persistence

# Vector Store (ChromaDB)

chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name = "document_qa_collection"
collection = chroma_client.get_or_create_collection(
    name=collection_name, embedding_function=local_ef
)

# === Setup: Local Language Model (Flan-T5) ===
flan_model_name = "google/flan-t5-large"
flan_tokenizer = AutoTokenizer.from_pretrained(flan_model_name)
flan_model = AutoModelForSeq2SeqLM.from_pretrained(flan_model_name)


# Function to load documents from a directory
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(
                os.path.join(directory_path, filename), "r", encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Load documents from the directory
# === Index Local Documents ===
def index_documents(directory_path="./news_articles"): #################################
    documents = load_documents_from_directory(directory_path)
    print(f"Loaded {len(documents)} documents")
    
    # Split documents into chunks
    chunked_documents = []
    for doc in documents:
        chunks = split_text(doc["text"])
        print(f"==== Splitting {doc['id']} into {len(chunks)} chunks ====")
        for i, chunk in enumerate(chunks):
            chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

    print(f"Total chunks: {len(chunked_documents)}")
    
    # Upsert documents with embeddings into Chroma
    # save each chunk + its vector in the Chroma collection.
    # Chroma is like a smart database that can search "semantic similarity" instead of exact words.

    for doc in chunked_documents:
        print(f"==== Inserting {doc['id']} into Chroma ====")
        collection.upsert(ids=[doc["id"]], documents=[doc["text"]])





#######################################
########################################

# === Query Relevant Chunks from Chroma ===


# Function to query documents
# Converts the question into a vector (embedding)
# Finds the top-N (n_results) most similar chunks from the database
# Returns those chunks as retrieval context
def query_documents(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("==== Retrieved relevant chunks ====")
    return relevant_chunks

# Function to generate a response 
# Generate Final Answer 
# 
# === Generate Answer using Local Flan-T5 ===
def generate_response(question, relevant_chunks):
    context = "\n".join(relevant_chunks)

    prompt = f"""Answer the question based on the context below. 
If the answer cannot be found in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

    input_ids = flan_tokenizer(prompt, return_tensors="pt").input_ids
    outputs = flan_model.generate(input_ids, max_new_tokens=300)
    answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer



# Example query
# query_documents("tell me about AI replacing TV writers strike.")
# Example query and response generation
# === Main Execution ===
if __name__ == "__main__":
    # Step 1: Index documents (do this only once)
    index_documents() ####################################################

    # Step 2: Ask a question
    question = "Tell me about Databricks"
    chunks = query_documents(question)
    answer = generate_response(question, chunks)

    print("\n🤖 Answer:", answer)

'''
 This performs full RAG pipeline:

1. Search documents (via Chroma)

2. Pick top 2 relevant chunks

3. Send chunks + question to model

4. Get and print the answer


'''