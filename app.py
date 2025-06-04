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
import re
import numpy as np
from typing import List, Dict, Any, Optional
from consts import available_functions

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
    user_id: str
    available_functions: Optional[List[Dict[str, Any]]] = []

# MongoEngine Document for messages
class AIMessage(Document):
    user_message = StringField(required=True)
    ai_response = StringField(required=True)
    user_id = StringField(required=True)
    created_at = DateTimeField(default=datetime.datetime.utcnow)
    meta = {
        'collection': 'ai_messages',
        'indexes': [
            {'fields': ['created_at'], 'expireAfterSeconds': 60 * 60 * 24 * 7},  # 7 days
            {'fields': ['user_id']}, 
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

def preprocess_text_for_embedding(text: str) -> str:
    """
    Preprocess text to improve embedding quality for personal information
    """
    # Convert to lowercase for consistency
    text = text.lower().strip()
    
    # Expand common contractions
    contractions = {
        "i'm": "i am",
        "my name's": "my name is",
        "i've": "i have",
        "i'll": "i will",
        "i'd": "i would",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "doesn't": "does not",
        "didn't": "did not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "wouldn't": "would not",
        "shouldn't": "should not",
        "couldn't": "could not"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Add context keywords for better semantic matching
    if re.search(r'\bmy name is\b|\bi am called\b|\bcall me\b', text):
        text = f"personal information name identity: {text}"
    elif re.search(r'\bi live\b|\bi am from\b|\bmy location\b', text):
        text = f"personal information location residence: {text}"
    elif re.search(r'\bi work\b|\bmy job\b|\bmy profession\b', text):
        text = f"personal information work career: {text}"
    elif re.search(r'\bi like\b|\bi love\b|\bmy favorite\b|\bi enjoy\b', text):
        text = f"personal information preferences interests: {text}"
    
    return text

def calculate_text_similarity(query: str, target: str) -> float:
    """
    Calculate text-based similarity for exact matches and keyword overlap
    """
    query_lower = query.lower().strip()
    target_lower = target.lower().strip()
    
    # Exact match gets highest score
    if query_lower == target_lower:
        return 1.0
    
    # Check for substring matches
    if query_lower in target_lower or target_lower in query_lower:
        return 0.8
    
    # Keyword overlap scoring
    query_words = set(re.findall(r'\b\w+\b', query_lower))
    target_words = set(re.findall(r'\b\w+\b', target_lower))
    
    if not query_words or not target_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = query_words.intersection(target_words)
    union = query_words.union(target_words)
    
    jaccard_score = len(intersection) / len(union) if union else 0.0
    
    # Boost score for important personal keywords
    important_keywords = {'name', 'called', 'identity', 'am', 'is'}
    if intersection.intersection(important_keywords):
        jaccard_score = min(jaccard_score + 0.3, 1.0)
    
    return jaccard_score

def extract_category_from_text(text: str) -> str:
    """
    Extracts a category from the text using simple keyword matching.
    You can expand this logic as needed.
    """
    categories = {
        "name": ["name", "identity", "called"],
        "location": ["location", "live", "residence", "from"],
        "work": ["work", "job", "profession", "career"],
        "preferences": ["like", "love", "favorite", "enjoy", "interest"],
        "other": []
    }
    text_lower = text.lower()
    for cat, keywords in categories.items():
        if any(kw in text_lower for kw in keywords):
            return cat
    return "other"

def search_milvus(query_text: str, user_id: str, category: str = None):
    milvus_client.load_collection(collection_name)
    
    # Preprocess query for better embedding
    processed_query = preprocess_text_for_embedding(query_text)
    query_vector = ollama.embeddings(model="nomic-embed-text", prompt=processed_query).embedding
    
    filter_str = f"user_id == '{user_id}'"
    if category:
        filter_str += f" && category == '{category}'"

    search_results = milvus_client.search(
        collection_name,
        data=[query_vector],
        output_fields=["user_message", "category"],
        limit=10,  
        filter=filter_str
    )
    
    print("Milvus search results:", search_results)
    
    enhanced_results = []
    for hit in search_results[0]:
        vector_score = hit.get("score", hit.get("distance", 0))  
        enhanced_results.append({
            "user_message": hit["user_message"],
            "vector_score": vector_score,
            "category": hit.get("category", "other")
        })
        print(f"Message: {hit['user_message'][:50]}...")
        print(f"Vector score: {vector_score:.4f}")
    
    enhanced_results.sort(key=lambda x: x["vector_score"], reverse=True)
    return enhanced_results[:10]

def save_message(message: str, response: str, user_id: str):
    AIMessage(user_message=message, ai_response=response, user_id=user_id).save()

def save_message_to_milvus(user_message: str, ai_response: str, user_id: str, category: str = None):
    # Preprocess message before creating embedding
    processed_message = preprocess_text_for_embedding(user_message)
    embedding = ollama.embeddings(model="nomic-embed-text", prompt=processed_message).embedding

    if not category:
        category = extract_category_from_text(user_message)

    # 1. Check for exact duplicate
    filter_str = f"user_id == '{user_id}' && category == '{category}'"
    existing = milvus_client.query(
        collection_name,
        filter=filter_str,
        output_fields=["user_message", "vector"]
    )
    for item in existing:
        # Exact match
        if item["user_message"] == user_message:
            print("Exact duplicate found in Milvus, skipping insert:", user_message)
            return

    search_results = milvus_client.search(
        collection_name,
        data=[embedding],
        output_fields=["user_message"],
        limit=3,
        filter=filter_str
    )
    for hit in search_results[0]:
        distance = hit.get("score", hit.get("distance", 0))
        if distance >= 0.85:
            print(f"Milvus built-in similarity found (cosine={distance:.2f}), skipping insert:", user_message)
            return

    print(f"Storing - Original: {user_message}")
    print(f"Storing - Processed: {processed_message}")
    print(f"Storing - Category: {category}")
    print("Embedding length (insert):", len(embedding))
    
    milvus_client.insert(
        collection_name,
        data={
            "user_message": user_message,  
            "vector": embedding,           
            "user_id": user_id,
            "category": category
        }
    )

async def get_user_name(user_id: str) -> str:
    """
    Fetch the user's name from stored information
    """
    try:
        # Search for name-related information in Milvus
        name_queries = ["name", "my name is", "called", "i am"]
        
        for query in name_queries:
            results = search_milvus(query, user_id, category="name")
            if results:
                # Look for the highest scoring result that contains name information
                for result in results:
                    message = result['user_message'].lower()
                    # Extract name from common patterns
                    name_patterns = [
                        "user's name is (.+)",
                        "user name is (.+)",
                        "user called (.+)",
                        "user is (.+)",
                    ]
                    
                    for pattern in name_patterns:
                        match = re.search(pattern, message)
                        if match:
                            return match.group(1).capitalize()
        
        return None
    except Exception as e:
        print(f"Error fetching user name: {e}")
        return None

# Add function to parse function calls from AI response
def parse_function_calls(ai_response: str) -> List[Dict[str, Any]]:
    """
    Parse function calls from AI response in format:
    <<FUNCTION_CALL: function_name | parameters: {"param1": "value1", "param2": "value2"}>>
    """
    function_calls = []
    
    # Pattern to match function calls
    pattern = r"<<FUNCTION_CALL:\s*(.*?)\s*\|\s*parameters:\s*({.*?})>>"
    matches = re.findall(pattern, ai_response, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        function_name = match[0].strip()
        try:
            parameters = json.loads(match[1])
            function_calls.append({
                "function_name": function_name,
                "parameters": parameters
            })
            print(f"Parsed function call: {function_name} with parameters: {parameters}")
        except json.JSONDecodeError as e:
            print(f"Error parsing function parameters: {e}")
            continue
    
    return function_calls

# Update the format_functions_for_ai function to handle your parameter structure
def format_functions_for_ai(functions: List[Dict[str, Any]]) -> str:
    """
    Format available functions for AI context
    """
    if not functions:
        return ""
    
    function_descriptions = []
    for func in functions:
        func_desc = f"- {func.get('name', 'unknown')}: {func.get('description', 'No description')}"
        
        # Add parameters info if available - updated for your structure
        if 'parameters' in func:
            params = func['parameters']
            param_list = []
            
            # Handle your parameter structure (dict with param_name: description)
            for param_name, param_desc in params.items():
                # Check if description contains type and requirement info
                if "(required)" in param_desc.lower():
                    req_str = " (required)"
                    param_desc = param_desc.replace("(required)", "").strip()
                elif "(optional)" in param_desc.lower():
                    req_str = " (optional)"
                    param_desc = param_desc.replace("(optional)", "").strip()
                else:
                    req_str = ""
                
                param_list.append(f"{param_name}{req_str}: {param_desc}")
            
            if param_list:
                func_desc += f"\n  Parameters: {'; '.join(param_list)}"
        
        function_descriptions.append(func_desc)
    
    return "\n".join(function_descriptions)

# Update the ask_ai function
async def ask_ai(message: str, user_id: str, available_functions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Use functions from consts if none provided
    if available_functions is None:
        available_functions = available_functions
    
    user_name = await get_user_name(user_id)
    
    retrieved_data = search_milvus(message, user_id)
    last_messages = list(AIMessage.objects(user_id=user_id).order_by('-created_at').limit(10))

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
    )
    
    # Add personalized greeting if name is available
    if user_name:
        instructions += f"The user's name is {user_name}. You can address them by name when appropriate.\n\n"
    
    # Add function calling instructions if functions are available
    if available_functions:
        function_list = format_functions_for_ai(available_functions)
        instructions += (
            "Function Calling Instructions:\n"
            "You have access to the following functions that you can call when appropriate:\n"
            f"{function_list}\n\n"
            "To call a function, use this exact format:\n"
            "<<FUNCTION_CALL: function_name | parameters: {\"param1\": \"value1\", \"param2\": \"value2\"}>>\n"
            "Only call functions when the user's request clearly requires that specific functionality. "
            "Do not call functions unnecessarily. You can call multiple functions if needed.\n"
            "Always provide a natural response to the user along with any function calls.\n\n"
            
            "Function Usage Guidelines:\n"
            "- create_subject: Use when user wants to create/add a new subject/area of life\n"
            "- add_component_to_subject: Use when user wants to add attributes/properties to an existing subject\n"
            "- create_connection: Use when user wants to link two subjects or create relationships\n"
            "- create_category: Use when user wants to organize subjects into categories\n"
            "- create_data_transfer: Use when user wants to set up data flow between components\n\n"
        )
    
    instructions += (
        "Context about this application:\n"
        "Here's a summary of its core concepts and structure:\n"
        "A unified productivity and life management app where everything is a 'subject' and interactions between them are 'connections'. "
        "The model mimics how people naturally think about life: as a network of interrelated topics, events, and responsibilities.\n"
        "Key Concepts:\n"
        "1. Subjects: Each aspect of your life is represented as a subject (e.g., 'Me', 'Tasks', 'Finance', 'University', 'Mental Health', 'Hobbies', 'People'). "
        "Subjects have customizable attributes, can contain widgets, and are modular.\n"
        "2. Connections: Subjects interact via Connections, defining input/output relationships. Connections can be manual or automated.\n"
        "3. Input/Output Flow: Data flows between subjects through connections (e.g., a Study Task outputs to 'Me' and boosts 'Knowledge' or 'Mood').\n"
        "4. Main Built-in Subjects: Me, Finance, HabitTracker.\n"
        "5. Visual Widgets: Some subjects render data visually (graphs, flowcharts) and support manual or automated input.\n"
        "AI Integration: The app includes AI-based features such as personalized suggestions, a natural language interface, and automated subject/connection generation based on usage.\n\n"
        "Conflict Handling Instructions:\n"
        "If you notice any conflicting or inconsistent information in the user's previous messages or stored knowledge, do the following:\n"
        "- Politely point out the conflict or inconsistency.\n"
        "- Output the previous (conflicting) information in this format for resolution.\n"
        "- If the user provides clarification or resolves the conflict, output the resolved information in this format:\n"
        "  <<RESOLVED: [the clarified/correct information about the user] | CONFLICTS_WITH: [brief description of what this replaces] | category: [category]>>\n"
        "- Always remain neutral and helpful when addressing conflicts.\n\n"
        "Memory Instructions:\n"
        "If the user provides information that should be remembered for future conversations (such as facts about themselves, preferences, corrections, or resolved conflicts), "
        "output the information to be stored in this format on a new line: <<STORE: [the information about the user]>>\n"
        "If no new information needs to be stored, do not output this marker.\n"
        "For both Conflicts and Memory storage, always phrase the stored information from a third-person perspective referring to 'the user' (e.g., 'The user prefers...', 'The user is studying...', 'The user lives in...').\n\n"
        "Below is the recent conversation history for context:\n"
    )
    
    user_info = ""
    if retrieved_data:
        user_info = "\n".join(
            f"User: {item['user_message']} (Category: {item['category']}, Vector Score: {item['vector_score']:.4f})"
            for item in retrieved_data
        )

    # Combine instructions, chat history, context, and user message
    user_prompt = instructions
    if chat_history:
        user_prompt += chat_history + "\n"
    if user_info:
        user_prompt += "\nRelevant knowledge about the user they may have said in previous messages (with relevance score):\n" + user_info + "\n"
    user_prompt += "\nand here is the current User input answer it based on the previous set of instructions and info provided: " + message

    print("User prompt for Gemini API:", user_prompt[:500] + "..." if len(user_prompt) > 500 else user_prompt)
    if user_name:
        print(f"User name retrieved: {user_name}")
    
    content = {
        "contents": [
            {"role": "user", "parts": [{"text": user_prompt}]}
        ]
    }

    ai_response = await call_gemini_api(API_KEY, content)

    # Parse function calls from AI response
    function_calls = parse_function_calls(ai_response)

    # Check for resolved marker (conflict resolution)
    resolved_match = re.search(
        r"<<RESOLVED:\s*(.*?)\s*\|\s*CONFLICTS_WITH:\s*(.*?)\s*\|\s*category:\s*(.*?)>>",
        ai_response, re.DOTALL | re.IGNORECASE
    )
    if resolved_match:
        resolved_info = resolved_match.group(1).strip()
        conflicts_with = resolved_match.group(2).strip()
        resolved_category = resolved_match.group(3).strip() if resolved_match.group(3) else extract_category_from_text(resolved_info)
        print("LLM resolved conflict:", resolved_info, "Conflicts with:", conflicts_with, "Category:", resolved_category)
        milvus_client.delete(
            collection_name,
            filter=f"user_id == '{user_id}' && category == '{resolved_category}' && user_message == '{conflicts_with}'"
        )
        save_message_to_milvus(resolved_info, ai_response, user_id, resolved_category)

    # Check for store marker (general memory)
    store_match = re.search(r"<<STORE:(.*?)(?:\|category:(.*?))?>>", ai_response, re.DOTALL)
    if store_match:
        store_info = store_match.group(1).strip()
        store_category = store_match.group(2).strip() if store_match.group(2) else extract_category_from_text(store_info)
        print("LLM chose to store:", store_info, "Category:", store_category)
        save_message_to_milvus(store_info, ai_response, user_id, store_category)

    save_message(message, ai_response, user_id)

    # Remove the markers from the response shown to the user
    ai_response_clean = re.sub(r"<<STORE:.*?>>", "", ai_response, flags=re.DOTALL)
    ai_response_clean = re.sub(r"<<RESOLVED:.*?>>", "", ai_response_clean, flags=re.DOTALL)
    ai_response_clean = re.sub(r"<<FUNCTION_CALL:.*?>>", "", ai_response_clean, flags=re.DOTALL)
    ai_response_clean = ai_response_clean.strip()
    
    return {
        "message": ai_response_clean,
        "function_calls": function_calls
    }

# API Route for Chat
@app.post('/chat')
async def chat(request: ChatRequest):
    if not request.message or not request.user_id:
        raise HTTPException(status_code=400, detail="No message or user_id provided")
    
    # Use functions from request or fall back to consts
    functions_to_use = request.available_functions if request.available_functions else available_functions
    
    result = await ask_ai(request.message, request.user_id, functions_to_use)
    
    return {
        "message": result["message"],
        "function_calls": result.get("function_calls", [])
    }

# Run with: uvicorn app:app --reload