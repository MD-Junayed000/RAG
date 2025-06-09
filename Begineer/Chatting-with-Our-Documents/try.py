import os
#from dotenv import load_dotenv
import chromadb
#from openai import OpenAI
#from chromadb.utils import embedding_functions

from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch




'''
# Load environment variables from .env file
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
## Create Embedding using the Embedding Function from OpenAI
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key, model_name="text-embedding-3-small"
)



'''


# Create a local embedding function using SentenceTransformer
local_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

















# Load Flan-T5 (small model for CPU, factual answers)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define QA-style chat function
def chat_with_bot(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_new_tokens=100)
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply
'''
# 🔹 Example: Ask the chatbot
if __name__ == "__main__":
    chat_history = None
    user_input = "What is human life expectancy in the United States?"
    reply, chat_history = chat_with_bot(user_input, chat_history) ## here user input is the prompt
    print("🤖 Bot:", reply)
    
'''
if __name__ == "__main__":

    while True:
        user_input = input("👤 You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = chat_with_bot(user_input) ## here user input is the prompt
        print("🤖 Bot:", reply)
