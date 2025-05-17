import ollama
import sys
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pymilvus import MilvusClient
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel

# Append custom C++ module path
sys.path.append(r"cpp_db_saver.pyd")
import cpp_db_saver  # Assuming this is a valid Python C++ extension

# Initialize FastAPI app
app = FastAPI()

# Milvus and Async MongoDB Clients
milvus_client = MilvusClient(uri="http://localhost:19530")
collection_name = "AI_info"

mongo_client = AsyncIOMotorClient("mongodb://localhost:27017/")
db = mongo_client["planitly"]
messages_collection = db["AI_message"]


# Request Model
class ChatRequest(BaseModel):
    message: str


# Caching Milvus search results
@lru_cache(maxsize=100)
def search_milvus(query_text: str):
    query_vector = ollama.embeddings(model="nomic-embed-text", prompt=query_text).embedding
    search_results = milvus_client.search(
        collection_name, data=[query_vector], output_fields=["text"], limit=3
    )
    return [hit["entity"]["text"] for hit in search_results[0]]


# Async function to save message to MongoDB
async def save_message(message: str, response: str):
    doc = {"user_message": message, "ai_response": response}
    await messages_collection.insert_one(doc)  # Non-blocking insert


# Async AI chat function
async def ask_ai(message: str) -> str:
    retrieved_data = search_milvus(message)
    system_message = "\n".join(retrieved_data) if retrieved_data else ""

    response = await ollama.chat(
        model='llama3.1',
        messages=[
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': message}
        ]
    )

    ai_response = response['message']['content']

    # Save response to MongoDB asynchronously
    await save_message(message, ai_response)

    return ai_response


# API Route for Chat
@app.post('/chat')
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="No message provided")

    ai_response = await ask_ai(request.message)
    return {"message": ai_response}

# Run with: uvicorn filename:app --reload
