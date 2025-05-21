import ollama
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from pymilvus import MilvusClient
from mongoengine import connect, Document, StringField
from pydantic import BaseModel
import requests
import json
from dotenv import load_dotenv
import os
from mongoengine import DateTimeField
import datetime
import uuid

load_dotenv()
API_KEY = os.getenv("API_KEY")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

# Connect to MongoDB using MongoEngine
connect(host=MONGO_URI, db="Cluster0")

# Initialize FastAPI app
app = FastAPI()

# Milvus Client
milvus_client = MilvusClient(uri="http://localhost:19530")
collection_name = "user_info"

# Pydantic model for chat request
class ChatRequest(BaseModel):
    message: str



# MongoEngine Document for messages
class AIMessage(Document):
    user_message = StringField(required=True)
    ai_response = StringField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)
    meta = {
        'collection': 'ai_messages',
        'indexes': [
            {'fields': ['created_at'], 'expireAfterSeconds': 60 * 60 * 24 * 7}  # 7 days
        ]
    }
    

async def call_gemini_api(apikey, content):
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={apikey}"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            url,
            headers=headers,
            json=content,
            timeout=30
        )
        data = response.json()
        print('Gemini API response:', data)
        if (data and 
            'candidates' in data and 
            len(data['candidates']) > 0 and
            'content' in data['candidates'][0] and
            'parts' in data['candidates'][0]['content']):
            bot_message_text = ' '.join([
                part.get('text', '') 
                for part in data['candidates'][0]['content']['parts']
                if 'text' in part
            ])
            return bot_message_text
        else:
            print('Invalid response from Gemini API:', data)
            return 'Sorry, I could not process your request.'
    except requests.exceptions.RequestException as error:
        print('Error communicating with Gemini API:', error)
        return 'Sorry, something went wrong. Please try again later.'
    except json.JSONDecodeError as error:
        print('Error parsing JSON response:', error)
        return 'Sorry, received an invalid response. Please try again later.'
    except Exception as error:
        print('Unexpected error:', error)
        return 'Sorry, an unexpected error occurred. Please try again later.'



# Caching Milvus search results
@lru_cache(maxsize=100)
def search_milvus(query_text: str):
    milvus_client.load_collection(collection_name)
    query_vector = ollama.embeddings(model="nomic-embed-text", prompt=query_text).embedding
    search_results = milvus_client.search(
        collection_name,
        data=[query_vector],
        output_fields=["user_message"],
        limit=3
    )
    # Include the score (heat) in the result
    return [
        {
            "user_message": hit["user_message"],
            "heat": hit.get("score", hit.get("distance", 0))  # Use score or distance
        }
        for hit in search_results[0]
    ]

# Save message using MongoEngine (sync, but can be run in threadpool if needed)
def save_message(message: str, response: str):
    AIMessage(user_message=message, ai_response=response).save()

def save_message_to_milvus(user_message: str, ai_response: str):
    # Create embedding for the user message
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=user_message).embedding
    milvus_client.insert(
        collection_name,
        data={
            "user_message": user_message,
            "vector": embedding
        }
    )

# Async AI chat function
async def ask_ai(message: str) -> str:
    retrieved_data = search_milvus(message)

    # Get the last 10 messages from MongoDB, most recent last
    last_messages = list(AIMessage.objects.order_by('-created_at').limit(10))

    # Format chat history efficiently for context
    chat_history = ""
    if last_messages:
        chat_history = "\n".join(
            f"User: {m.user_message}\nAI: {m.ai_response}" for m in last_messages
        )

    # Define the assistant instructions and program context
    instructions = (
        "You are an AI assistant named Planitly. "
        "You are helpful, concise, and always polite. "
        "Answer user questions to the best of your ability. "
        "If you do not know the answer, say so honestly. "
        "Do not provide medical, legal, or financial advice. "
        "Always keep responses clear and friendly.\n\n"
        "Context about this application:\n"
        "Here's a summary of its core concepts and structure:\n"
        "A unified productivity and life management app where everything is a 'subject' and interactions between them are 'connections'. "
        "The model mimics how people naturally think about life: as a network of interrelated topics, events, and responsibilities.\n"
        "Key Concepts:\n"
        "1. Subjects: Each aspect of your life is represented as a subject (e.g., 'Me', 'Tasks', 'Finance', 'University', 'Mental Health', 'Hobbies', 'People'). "
        "Subjects have customizable attributes, can contain widgets, and are modular.\n"
        "2. Connections: Subjects interact via Connections, defining input/output relationships. Connections can be manual or automated.\n"
        "3. Input/Output Flow: Data flows between subjects through connections (e.g., a Study Task outputs to 'Me' and boosts 'Knowledge' or 'Mood').\n"
        "4. Main Built-in Subjects: Me,  Finance , HabitTracker.\n"
        "5. Visual Widgets: Some subjects render data visually (graphs, flowcharts) and support manual or automated input.\n"
        "AI Integration: The app includes AI-based features such as personalized suggestions, a natural language interface, and automated subject/connection generation based on usage."
        "Below is the recent conversation history for context:\n"
    )
    user_info = ""
    if retrieved_data:
        user_info = "\n".join(
            f"User: {item['user_message']} (Relevance: {item['heat']:.2f})"
            for item in retrieved_data
        )

    # Combine instructions, chat history, context, and user message
    user_prompt = instructions
    if chat_history:
        user_prompt += chat_history + "\n"
    if user_info:
        user_prompt += "\nRelevant knowledge about the user they may have said in previous messages (with relevance score):\n" + user_info + "\n"
    user_prompt += "\n and here is the current User input answer it based on the previous set of instructions and info provided: " + message

    print ("User prompt for Gemini API:", user_prompt)
    # Prepare content for Gemini API (user role only)
    content = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]}
        ]
    }

    ai_response = await call_gemini_api(API_KEY, content)

    save_message(message, ai_response)
    save_message_to_milvus(message, ai_response) 

    return ai_response
# API Route for Chat
@app.post('/chat')
async def chat(request: ChatRequest):
    if not request.message:
        raise HTTPException(status_code=400, detail="No message provided")

    ai_response = await ask_ai(request.message)
    return {"message": ai_response}

# Run with: uvicorn filename:app --reload
