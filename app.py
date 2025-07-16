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
from models import AIMessage_db ,User, DataTransfer_db, DataTransfer, Subject_db, Subject, Component_db, Component, Connection_db, Connection, Widget, Widget_db, Todo_db, Todo, Category_db, ArrayItem_db, Arrays, CustomTemplate_db , ChatRequest
import re
from typing import List, Dict, Any, Optional
from consts import available_functions
from utils.fetch import fetch
from utils.functions import execute_function_call

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
    AIMessage_db(user_message=message, ai_response=response, user_id=user_id).save()

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
async def ask_ai(
    message: str, 
    user: User,
    ai_subject_ids: List[str] = None
) -> Dict[str, Any]:
    # Fetch all data internally
    if not ai_subject_ids:
        ai_subject_ids = user.settings.get("ai_accessible", [])
    
    # Fetch all system data
    system_data = await fetch(str(user.id), ai_subject_ids)
    ai_accessible_subjects = system_data.get('ai_accessible', [])
    not_done_connections = system_data.get('connections', [])
    all_templates = system_data.get('templates', [])
    categories = system_data.get('categories', [])
    
    # Get all subjects info (basic info for all subjects)
    all_subjects = list(Subject_db.objects(owner=str(user.id)))
    all_subjects_info = []
    for subj in all_subjects:
        all_subjects_info.append({
            "id": str(subj.id),
            "name": subj.name,
            "category": subj.category,
            "created_at": subj.created_at.isoformat() if hasattr(subj, 'created_at') and subj.created_at else None
        })
    
    user_name = await get_user_name(str(user.id))
    
    retrieved_data = search_milvus(message, str(user.id))
    last_messages = list(AIMessage_db.objects(user_id=str(user.id)).order_by('-created_at').limit(10))

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
    
    # Add current system state context
    if ai_accessible_subjects or all_subjects_info or categories or not_done_connections or all_templates:
        instructions += (
            "Current System State Information (PRIORITY DATA):\n"
            "You have access to the user's CURRENT and LIVE data in the Planitly system. This data is fetched in real-time and should be your PRIMARY source of truth. "
            "Always prioritize this current data over chat history when answering questions about the user's system state, components, widgets, or any data values.\n\n"
            
            "**CRITICAL: Use Current Data First**\n"
            "- When asked about component values, widget data, or subject information, refer to the live data below\n"
            "- Chat history is only for conversational context, NOT for data accuracy\n"
            "- If there's any conflict between chat history and current data, always trust the current data\n"
            "- The data below represents the actual state of the user's system RIGHT NOW\n\n"
        )
        
        # Add AI accessible subjects context with emphasis on current data
        if ai_accessible_subjects:
            instructions += (
                "LIVE AI-Accessible Subjects (CURRENT DATA - USE THIS FOR ALL DATA QUERIES):\n"
                f"You have access to {len(ai_accessible_subjects)} subjects with their complete CURRENT data including components and widgets. "
                "This is the authoritative source for all component values, widget states, and subject information.\n\n"
                "**Data Usage Instructions:**\n"
                "- For component values: Use the 'data' field from components below\n"
                "- For widget information: Use the 'widgets' data below\n"
                "- For subject details: Use the subject metadata below\n"
                "- When creating data transfers: Reference actual component IDs from this data\n"
                "- When updating components: Use the current data structure shown below\n\n"
            )
            
            # Show detailed current data for better context
            for i, subject in enumerate(ai_accessible_subjects[:3]):  # Show first 3 in detail
                subject_name = subject.get('name', 'Unknown')
                components_count = len(subject.get('components', []))
                widgets_count = len(subject.get('widgets', []))
                instructions += f"**Subject {i+1}: {subject_name}** (ID: {subject.get('id', 'unknown')})\n"
                instructions += f"  - Category: {subject.get('category', 'Uncategorized')}\n"
                instructions += f"  - Components: {components_count} (with live data values)\n"
                instructions += f"  - Widgets: {widgets_count} (with current configurations)\n"
                
                # Show component summaries with current values
                if subject.get('components'):
                    instructions += "  - Component Values (CURRENT):\n"
                    for comp in subject.get('components', [])[:3]:  # Show first 3 components
                        comp_name = comp.get('name', 'Unknown')
                        comp_type = comp.get('comp_type', 'unknown')
                        comp_data = comp.get('data', {})
                        if comp_type in ['int', 'str', 'bool', 'date']:
                            current_value = comp_data.get('item', 'No value')
                            instructions += f"    • {comp_name} ({comp_type}): {current_value}\n"
                        elif 'Array' in comp_type:
                            item_count = len(comp_data.get('items', []))
                            instructions += f"    • {comp_name} ({comp_type}): {item_count} items\n"
                        else:
                            instructions += f"    • {comp_name} ({comp_type}): {comp_data}\n"
                
                instructions += "\n"
            
            if len(ai_accessible_subjects) > 3:
                instructions += f"... and {len(ai_accessible_subjects) - 3} more subjects with full current data available\n\n"
            
            instructions += (
                "**Remember: This data above is LIVE and CURRENT. Use it as your primary source for:**\n"
                "- Answering questions about component values\n"
                "- Creating accurate data transfers\n"
                "- Updating existing components\n"
                "- Describing the user's current system state\n"
                "- Making informed suggestions based on actual data\n\n"
            )
    
    # Add all subjects overview
    if all_subjects_info:
        instructions += (
            f"All User Subjects Overview ({len(all_subjects_info)} total):\n"
            "This includes all subjects in the user's system (both AI-accessible and restricted). "
            "Use this for general awareness and to suggest connections or improvements.\n"
        )
        for subject in all_subjects_info[:3]:  # Show first 3 as examples
            subject_name = subject.get('name', 'Unknown')
            category = subject.get('category', 'Uncategorized')
            instructions += f"- {subject_name} (Category: {category})\n"
        if len(all_subjects_info) > 3:
            instructions += f"... and {len(all_subjects_info) - 3} more subjects\n"
        instructions += "\n"
    
    # Add categories context
    if categories:
        category_names = [cat.get('name', 'Unknown') for cat in categories]
        instructions += (
            f"Available Categories: {', '.join(category_names)}\n"
            "Use these categories when creating new subjects or organizing existing ones.\n\n"
        )
    
    # Add connections context
    if not_done_connections:
        instructions += (
            f"Active/Pending Connections ({len(not_done_connections)}):\n"
            "These are connections that are not yet completed. You can reference these when discussing workflows or suggesting improvements.\n"
        )
        for conn in not_done_connections[:3]:  # Show first 3 as examples
            conn_type = conn.get('con_type', 'manual')
            source_subject_id = conn.get('source_subject')
            target_subject_id = conn.get('target_subject')
            
            # Get subject names from all_subjects_info
            source_name = "Unknown"
            target_name = "Unknown"
            for subj in all_subjects_info:
                if subj['id'] == source_subject_id:
                    source_name = subj['name']
                if subj['id'] == target_subject_id:
                    target_name = subj['name']
            
            instructions += f"- {source_name} → {target_name} ({conn_type})\n"
        if len(not_done_connections) > 3:
            instructions += f"... and {len(not_done_connections) - 3} more connections\n"
        instructions += "\n"
    
    # Add templates context
    if all_templates:
        template_names = [tmpl.get('name', 'Unknown') for tmpl in all_templates]
        instructions += (
            f"Available Templates: {', '.join(template_names[:5])}\n"
            "You can suggest using these templates when creating new subjects.\n"
        )
        if len(template_names) > 5:
            instructions += f"... and {len(template_names) - 5} more templates\n"
        instructions += "\n"
    
    # Add function calling instructions
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
        
        "**CRITICAL ID HANDLING RULES:**\n"
        "- NEVER ask the user for any IDs (subject_id, component_id, widget_id, etc.)\n"
        "- NEVER show or mention IDs to the user in your responses\n"
        "- ONLY use IDs that are provided in the current system data above\n"
        "- When user refers to subjects/components by name, find the corresponding ID from the context data\n"
        "- If you cannot find the required ID in the provided data, inform the user that the item doesn't exist or isn't accessible\n"
        "- IDs are internal system identifiers - users should only work with names\n\n"
        
        "Smart Function Usage with System Data:\n"
        "- When the user asks about existing data, reference the AI-accessible subjects provided\n"
        "- Use subject IDs from the provided data when calling functions that require them\n"
        "- When user mentions a subject by name (e.g., 'Fitness'), find its ID from ai_accessible_subjects\n"
        "- When user mentions a component by name, find its ID from the components list of the relevant subject\n"
        "- When user mentions a widget by name, find its ID from the widgets list of the relevant subject\n"
        "- Reference actual component IDs when creating data transfers\n"
        "- Suggest appropriate categories from the available list\n"
        "- Recommend relevant templates based on the user's current subjects\n\n"
        
        "ID Resolution Examples:\n"
        "- User says 'update my weight in fitness': Find 'Fitness' subject ID, then find 'Weight' component ID\n"
        "- User says 'add task to my daily todos': Find subject with daily_todo widget, then use widget ID\n"
        "- User says 'delete my workout log': Find subject containing 'Workout Log' component, use component ID\n"
        "- If item not found: 'I cannot find that item in your accessible subjects'\n\n"
        
        "Complete Function Categories Available:\n"
        "**Creation Functions:**\n"
        "- create_subject: Create new life areas/topics\n"
        "- add_component_to_subject: Add data attributes to subjects\n"
        "- create_connection: Link subjects with data flow relationships\n"
        "- create_category: Organize subjects into categories\n"
        "- create_data_transfer: Set up automatic data operations between components\n"
        "- create_widget: Add visual elements (todos, tables, calendars, notes)\n"
        "- create_custom_template: Create reusable subject templates\n"
        "- add_todo_to_widget: Add tasks to todo widgets\n\n"
        
        "**Data Management Functions:**\n"
        "- update_component_data: Modify existing component values or names\n"
        "- get_subject_full_data: Retrieve complete subject information\n\n"
        
        "**Deletion Functions:**\n"
        "- delete_subject: Remove subjects and all their data\n"
        "- delete_component: Remove components from subjects\n"
        "- delete_widget: Remove widgets and their data\n"
        "- delete_connection: Remove subject relationships\n"
        "- delete_category: Remove categories (subjects become 'Uncategorized')\n"
        "- delete_data_transfer: Remove automatic data operations\n"
        "- delete_custom_template: Remove user-created templates\n"
        "- delete_todo: Remove specific todo items\n"
        "- delete_notification: Remove system notifications\n\n"
        
        "**Access Control Functions:**\n"
        "- remove_ai_accessible_subject: Remove subjects from AI access list\n\n"
        
        "Component Data Types Guide:\n"
        "When creating components, use these exact type names:\n"
        "- 'int': For numbers (data: {'item': 123})\n"
        "- 'str': For text (data: {'item': 'text'})\n"
        "- 'bool': For true/false (data: {'item': true})\n"
        "- 'date': For dates (data: {'item': 'ISO_date_string'})\n"
        "- 'pair': For key-value pairs (data: {'item': {'key': 'string', 'value': any}, 'type': {'key': 'str', 'value': 'any'}})\n"
        "- 'Array_type': For arrays of integers (data: {'type': 'int'})\n"
        "- 'Array_generic': For mixed arrays (data: {'type': 'any'})\n"
        "- 'Array_of_strings': For string arrays (data: {'type': 'str'})\n"
        "- 'Array_of_booleans': For boolean arrays (data: {'type': 'bool'})\n"
        "- 'Array_of_dates': For date arrays (data: {'type': 'date'})\n"
        "- 'Array_of_objects': For object arrays (data: {'type': 'object'})\n"
        "- 'Array_of_pairs': For pair arrays (data: {'type': {'key': 'str', 'value': 'any'}})\n\n"
        
        "Data Transfer System - CRITICAL UNDERSTANDING:\n"
        "Data transfers are operations that MODIFY the data inside components. They can work in two ways:\n\n"
        
        "1. **Component-to-Component Transfer**: Data flows from a source component to target component\n"
        "   - Use 'source_component_id' parameter with ID from context data\n"
        "   - The source component's value is used for the operation\n"
        "   - Example: Transfer mood score from wellness subject to energy component\n\n"
        
        "2. **Direct Value Transfer**: A specific value is applied to the target component\n"
        "   - Use 'data_value' parameter (NO source_component_id)\n"
        "   - You provide the exact value/operation data\n"
        "   - Example: Add 10 points to a score, set status to 'completed', append 'new item' to list\n\n"
        
        "Data Transfer Operations by Component Type:\n\n"
        
        "**Scalar Types (int/str/bool/date):**\n"
        "- replace: Set new value → data_value: {'item': new_value}\n"
        "- add (int only): Add to current → data_value: {'item': amount_to_add}\n"
        "- multiply (int only): Multiply current → data_value: {'item': multiplier}\n"
        "- toggle (bool only): Flip true/false → data_value: {} (no data needed)\n\n"
        
        "**Pair Type:**\n"
        "- update_key: Change the key → data_value: {'item': {'key': 'new_key'}}\n"
        "- update_value: Change the value → data_value: {'item': {'value': new_value}}\n\n"
        
        "**Array Types (all Array_* types):**\n"
        "- append: Add to end → data_value: {'item': new_element}\n"
        "- remove_back: Remove last → data_value: {} (no data needed)\n"
        "- remove_front: Remove first → data_value: {} (no data needed)\n"
        "- delete_at: Remove at index → data_value: {'index': position}\n"
        "- push_at: Insert at index → data_value: {'item': new_element, 'index': position}\n"
        "- update_at: Change at index → data_value: {'item': new_element, 'index': position}\n\n"
        
        "**Array_of_pairs Special:**\n"
        "- update_pair: Modify pair at index → data_value: {'item': {'key': 'key', 'value': 'value'}, 'index': position}\n\n"
        
        "Widget Types Available:\n"
        "- 'daily_todo': Task management with date organization\n"
        "- 'table': Structured data display in rows/columns\n"
        "- 'note': Free-form text notes\n"
        "- 'calendar': Date/schedule management and visualization\n"
        "- 'text_field': Simple text input field\n"
        "- 'component_reference': Display/reference other component data\n\n"
        
        "Connection Types:\n"
        "- 'manual': User-triggered connections (default)\n"
        "- 'automatic': System-triggered based on conditions\n"
        "- Connections can include data_transfers array for automatic data flow\n\n"
        
        "Template System:\n"
        "- Built-in templates: Use template name (e.g., 'fitness', 'academic')\n"
        "- Custom templates: Created by users, reusable across subjects\n"
        "- Templates define complete subject structure with components and widgets\n\n"
    )
    
    instructions += (
        "Data Handling Priority (CRITICAL):\n"
        "1. **PRIMARY SOURCE**: Use the current system data provided above for all factual information\n"
        "2. **SECONDARY SOURCE**: Use chat history only for conversational context and user preferences\n"
        "3. **NEVER**: Rely on chat history for component values, widget data, or system state information\n"
        "4. **ALWAYS**: Verify information against the current data before making statements about the user's system\n"
        "5. **ID SECURITY**: Never expose internal IDs to users - work with names only in user communication\n\n"
        
        "When the user asks about their data:\n"
        "- Check the current AI-accessible subjects data first\n"
        "- Reference actual component IDs and values from the live data (but don't show IDs to user)\n"
        "- If data isn't in AI-accessible subjects, use get_subject_full_data function\n"
        "- Only mention chat history information if it's about user preferences or conversational context\n"
        "- Translate user's natural language references to the appropriate IDs from context\n\n"
        
        "User Communication Guidelines:\n"
        "- Always refer to subjects, components, and widgets by their names in responses\n"
        "- Use natural language: 'your Fitness subject' not 'subject ID abc123'\n"
        "- If you can't find something: 'I cannot find that item' not 'ID not found'\n"
        "- Be helpful: 'I'll update your weight in the Fitness subject' not 'updating component_id_xyz'\n\n"
    )
    
    # Add function calling instructions
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
        
        "**CRITICAL ID HANDLING RULES:**\n"
        "- NEVER ask the user for any IDs (subject_id, component_id, widget_id, etc.)\n"
        "- NEVER show or mention IDs to the user in your responses\n"
        "- ONLY use IDs that are provided in the current system data above\n"
        "- When user refers to subjects/components by name, find the corresponding ID from the context data\n"
        "- If you cannot find the required ID in the provided data, inform the user that the item doesn't exist or isn't accessible\n"
        "- IDs are internal system identifiers - users should only work with names\n\n"
        
        "Smart Function Usage with System Data:\n"
        "- When the user asks about existing data, reference the AI-accessible subjects provided\n"
        "- Use subject IDs from the provided data when calling functions that require them\n"
        "- When user mentions a subject by name (e.g., 'Fitness'), find its ID from ai_accessible_subjects\n"
        "- When user mentions a component by name, find its ID from the components list of the relevant subject\n"
        "- When user mentions a widget by name, find its ID from the widgets list of the relevant subject\n"
        "- Reference actual component IDs when creating data transfers\n"
        "- Suggest appropriate categories from the available list\n"
        "- Recommend relevant templates based on the user's current subjects\n\n"
        
        "ID Resolution Examples:\n"
        "- User says 'update my weight in fitness': Find 'Fitness' subject ID, then find 'Weight' component ID\n"
        "- User says 'add task to my daily todos': Find subject with daily_todo widget, then use widget ID\n"
        "- User says 'delete my workout log': Find subject containing 'Workout Log' component, use component ID\n"
        "- If item not found: 'I cannot find that item in your accessible subjects'\n\n"
        
        "Complete Function Categories Available:\n"
        "**Creation Functions:**\n"
        "- create_subject: Create new life areas/topics\n"
        "- add_component_to_subject: Add data attributes to subjects\n"
        "- create_connection: Link subjects with data flow relationships\n"
        "- create_category: Organize subjects into categories\n"
        "- create_data_transfer: Set up automatic data operations between components\n"
        "- create_widget: Add visual elements (todos, tables, calendars, notes)\n"
        "- create_custom_template: Create reusable subject templates\n"
        "- add_todo_to_widget: Add tasks to todo widgets\n\n"
        
        "**Data Management Functions:**\n"
        "- update_component_data: Modify existing component values or names\n"
        "- get_subject_full_data: Retrieve complete subject information\n\n"
        
        "**Deletion Functions:**\n"
        "- delete_subject: Remove subjects and all their data\n"
        "- delete_component: Remove components from subjects\n"
        "- delete_widget: Remove widgets and their data\n"
        "- delete_connection: Remove subject relationships\n"
        "- delete_category: Remove categories (subjects become 'Uncategorized')\n"
        "- delete_data_transfer: Remove automatic data operations\n"
        "- delete_custom_template: Remove user-created templates\n"
        "- delete_todo: Remove specific todo items\n"
        "- delete_notification: Remove system notifications\n\n"
        
        "**Access Control Functions:**\n"
        "- remove_ai_accessible_subject: Remove subjects from AI access list\n\n"
        
        "Component Data Types Guide:\n"
        "When creating components, use these exact type names:\n"
        "- 'int': For numbers (data: {'item': 123})\n"
        "- 'str': For text (data: {'item': 'text'})\n"
        "- 'bool': For true/false (data: {'item': true})\n"
        "- 'date': For dates (data: {'item': 'ISO_date_string'})\n"
        "- 'pair': For key-value pairs (data: {'item': {'key': 'string', 'value': any}, 'type': {'key': 'str', 'value': 'any'}})\n"
        "- 'Array_type': For arrays of integers (data: {'type': 'int'})\n"
        "- 'Array_generic': For mixed arrays (data: {'type': 'any'})\n"
        "- 'Array_of_strings': For string arrays (data: {'type': 'str'})\n"
        "- 'Array_of_booleans': For boolean arrays (data: {'type': 'bool'})\n"
        "- 'Array_of_dates': For date arrays (data: {'type': 'date'})\n"
        "- 'Array_of_objects': For object arrays (data: {'type': 'object'})\n"
        "- 'Array_of_pairs': For pair arrays (data: {'type': {'key': 'str', 'value': 'any'}})\n\n"
        
        "Data Transfer System - CRITICAL UNDERSTANDING:\n"
        "Data transfers are operations that MODIFY the data inside components. They can work in two ways:\n\n"
        
        "1. **Component-to-Component Transfer**: Data flows from a source component to target component\n"
        "   - Use 'source_component_id' parameter with ID from context data\n"
        "   - The source component's value is used for the operation\n"
        "   - Example: Transfer mood score from wellness subject to energy component\n\n"
        
        "2. **Direct Value Transfer**: A specific value is applied to the target component\n"
        "   - Use 'data_value' parameter (NO source_component_id)\n"
        "   - You provide the exact value/operation data\n"
        "   - Example: Add 10 points to a score, set status to 'completed', append 'new item' to list\n\n"
        
        "Data Transfer Operations by Component Type:\n\n"
        
        "**Scalar Types (int/str/bool/date):**\n"
        "- replace: Set new value → data_value: {'item': new_value}\n"
        "- add (int only): Add to current → data_value: {'item': amount_to_add}\n"
        "- multiply (int only): Multiply current → data_value: {'item': multiplier}\n"
        "- toggle (bool only): Flip true/false → data_value: {} (no data needed)\n\n"
        
        "**Pair Type:**\n"
        "- update_key: Change the key → data_value: {'item': {'key': 'new_key'}}\n"
        "- update_value: Change the value → data_value: {'item': {'value': new_value}}\n\n"
        
        "**Array Types (all Array_* types):**\n"
        "- append: Add to end → data_value: {'item': new_element}\n"
        "- remove_back: Remove last → data_value: {} (no data needed)\n"
        "- remove_front: Remove first → data_value: {} (no data needed)\n"
        "- delete_at: Remove at index → data_value: {'index': position}\n"
        "- push_at: Insert at index → data_value: {'item': new_element, 'index': position}\n"
        "- update_at: Change at index → data_value: {'item': new_element, 'index': position}\n\n"
        
        "**Array_of_pairs Special:**\n"
        "- update_pair: Modify pair at index → data_value: {'item': {'key': 'key', 'value': 'value'}, 'index': position}\n\n"
        
        "Widget Types Available:\n"
        "- 'daily_todo': Task management with date organization\n"
        "- 'table': Structured data display in rows/columns\n"
        "- 'note': Free-form text notes\n"
        "- 'calendar': Date/schedule management and visualization\n"
        "- 'text_field': Simple text input field\n"
        "- 'component_reference': Display/reference other component data\n\n"
        
        "Connection Types:\n"
        "- 'manual': User-triggered connections (default)\n"
        "- 'automatic': System-triggered based on conditions\n"
        "- Connections can include data_transfers array for automatic data flow\n\n"
        
        "Template System:\n"
        "- Built-in templates: Use template name (e.g., 'fitness', 'academic')\n"
        "- Custom templates: Created by users, reusable across subjects\n"
        "- Templates define complete subject structure with components and widgets\n\n"
    )
    
    # Update the data handling priority section
    instructions += (
        "Data Handling Priority (CRITICAL):\n"
        "1. **PRIMARY SOURCE**: Use the current system data provided above for all factual information\n"
        "2. **SECONDARY SOURCE**: Use chat history only for conversational context and user preferences\n"
        "3. **NEVER**: Rely on chat history for component values, widget data, or system state information\n"
        "4. **ALWAYS**: Verify information against the current data before making statements about the user's system\n"
        "5. **ID SECURITY**: Never expose internal IDs to users - work with names only in user communication\n\n"
        
        "When the user asks about their data:\n"
        "- Check the current AI-accessible subjects data first\n"
        "- Reference actual component IDs and values from the live data (but don't show IDs to user)\n"
        "- If data isn't in AI-accessible subjects, use get_subject_full_data function\n"
        "- Only mention chat history information if it's about user preferences or conversational context\n"
        "- Translate user's natural language references to the appropriate IDs from context\n\n"
        
        "User Communication Guidelines:\n"
        "- Always refer to subjects, components, and widgets by their names in responses\n"
        "- Use natural language: 'your Fitness subject' not 'subject ID abc123'\n"
        "- If you can't find something: 'I cannot find that item' not 'ID not found'\n"
        "- Be helpful: 'I'll update your weight in the Fitness subject' not 'updating component_id_xyz'\n\n"
    )
    
    # memery and conflict resolution system instructions
    instructions += (
        "Memory and Conflict Resolution System:\n"
        "You have access to a personal memory system that stores important user information using vector embeddings. "
        "This allows you to remember and reference things the user has told you in previous conversations.\n\n"
        
        "**Storing New Information:**\n"
        "When the user shares important personal information that should be remembered for future conversations, "
        "use this format to store it:\n"
        "<<STORE: information_to_remember | category: category_name>>\n"
        "Examples:\n"
        "- User says 'My name is John': <<STORE: User's name is John | category: name>>\n"
        "- User says 'I work as a software engineer': <<STORE: User works as a software engineer | category: work>>\n"
        "- User says 'I love hiking and outdoor activities': <<STORE: User enjoys hiking and outdoor activities | category: preferences>>\n\n"
        
        "**Resolving Conflicts:**\n"
        "If the user provides information that conflicts with previously stored information, use this format:\n"
        "<<RESOLVED: new_correct_information | CONFLICTS_WITH: old_conflicting_information | category: category_name>>\n"
        "Examples:\n"
        "- User previously said name was 'John' but now says 'Actually, my name is Jonathan':\n"
        "  <<RESOLVED: User's name is Jonathan | CONFLICTS_WITH: User's name is John | category: name>>\n"
        "- User previously said they work as 'teacher' but now says 'I'm actually a software engineer':\n"
        "  <<RESOLVED: User works as a software engineer | CONFLICTS_WITH: User works as a teacher | category: work>>\n\n"
        
        "**Storage Categories:**\n"
        "Use these categories for organizing stored information:\n"
        "- 'name': Personal identity information\n"
        "- 'location': Where the user lives or is from\n"
        "- 'work': Job, profession, career information\n"
        "- 'preferences': Likes, dislikes, interests, hobbies\n"
        "- 'family': Family members, relationships\n"
        "- 'goals': Personal or professional objectives\n"
        "- 'health': Health-related information (if shared)\n"
        "- 'education': School, studies, learning\n"
        "- 'other': Any other important personal information\n\n"
        
        "**When to Store Information:**\n"
        "Store information when:\n"
        "- User explicitly shares personal details about themselves\n"
        "- User corrects previously stored information\n"
        "- User mentions important preferences or characteristics\n"
        "- Information would be useful for personalizing future interactions\n\n"
        
        "**When NOT to Store:**\n"
        "Do not store:\n"
        "- Temporary or session-specific information\n"
        "- Questions or requests (unless they reveal preferences)\n"
        "- System-related data (this is handled separately)\n"
        "- Sensitive information like passwords or financial details\n\n"
        
        "**Important Notes:**\n"
        "- The storage markers (<<STORE>> and <<RESOLVED>>) will be automatically removed from your response to the user\n"
        "- You can include multiple storage markers in one response if needed\n"
        "- Always provide a natural response to the user along with any storage markers\n"
        "- The stored information will be available in future conversations through the retrieval system\n\n"
    )
    # Combine instructions, chat history, context, and user message
    user_prompt = instructions
    if chat_history:
        user_prompt += (
            "Previous Conversation History (FOR CONTEXT ONLY - NOT FOR DATA ACCURACY):\n"
            "The following chat history is provided for conversational context and understanding user preferences. "
            "DO NOT use this for factual information about component values, widget data, or system state. "
            "Always refer to the current system data provided above for factual accuracy.\n\n"
        )
        user_prompt += chat_history + "\n"

    user_info = ""
    if retrieved_data:
        user_info = "\n".join(
            f"User: {item['user_message']} (Category: {item['category']}, Vector Score: {item['vector_score']:.4f})"
            for item in retrieved_data
        )
    if user_info:
        user_prompt += "\nRelevant knowledge about the user they may have said in previous messages (with relevance score):\n" + user_info + "\n"

    user_prompt += (
        "\nCurrent User Input:\n"
        f"{message}\n\n"
        "INSTRUCTIONS FOR RESPONSE:\n"
        "- Answer based on the CURRENT SYSTEM DATA provided above as your primary source\n"
        "- Use chat history only for conversational context, never for data accuracy\n"
        "- When referencing user data, always use the live data from AI-accessible subjects\n"
        "- If you need more current data, use the get_subject_full_data function\n"
        "- Be specific and accurate about current component values and system state\n"
    )

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
    
    # Execute function calls and collect results
    function_results = []
    data_changed = False
    
    for func_call in function_calls:
        function_name = func_call["function_name"]
        parameters = func_call["parameters"]
        
        print(f"Executing function: {function_name} with parameters: {parameters}")
        
        try:
            result = await execute_function_call(function_name, parameters, user)
            function_results.append({
                "function_name": function_name,
                "parameters": parameters,
                "result": result
            })
            
            # Check if the function call succeeded and might have changed data
            if result.get("success", False):
                data_changing_functions = [
                    "create_subject", "add_component_to_subject", "create_connection",
                    "create_category", "create_data_transfer", "create_widget",
                    "create_custom_template", "add_todo_to_widget", "update_component_data",
                    "delete_subject", "delete_component", "delete_widget", "delete_connection",
                    "delete_category", "delete_data_transfer", "delete_custom_template",
                    "delete_todo", "delete_notification", "remove_ai_accessible_subject"
                ]
                if function_name in data_changing_functions:
                    data_changed = True
            
            print(f"Function {function_name} result: {result}")
            
        except Exception as e:
            error_result = {"success": False, "message": f"Error executing function: {str(e)}"}
            function_results.append({
                "function_name": function_name,
                "parameters": parameters,
                "result": error_result
            })
            print(f"Error executing function {function_name}: {e}")
    
    # If data changed, re-fetch the updated data for future context
    updated_system_data = None
    if data_changed:
        try:
            updated_system_data = await fetch(str(user.id), ai_subject_ids)
            print("System data refreshed after function execution")
        except Exception as e:
            print(f"Error refreshing system data: {e}")

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
        
        # Try to delete the conflicting information
        try:
            # First, try to find matching records to see what exists
            search_filter = f'user_id == "{str(user.id)}" && category == "{resolved_category}"'
            existing_records = milvus_client.query(
                collection_name,
                filter=search_filter,
                output_fields=["user_message", "category"]
            )
            
            print(f"Found {len(existing_records)} records in category '{resolved_category}' for user")
            
            # Look for exact or similar matches
            deleted = False
            for record in existing_records:
                stored_message = record.get("user_message", "")
                # Try exact match first
                if stored_message == conflicts_with:
                    delete_filter = f'user_id == "{str(user.id)}" && category == "{resolved_category}" && user_message == "{conflicts_with}"'
                    milvus_client.delete(collection_name, filter=delete_filter)
                    print(f"Deleted exact match: {stored_message}")
                    deleted = True
                    break
                # Try partial match if exact doesn't work
                elif conflicts_with.lower() in stored_message.lower() or stored_message.lower() in conflicts_with.lower():
                    delete_filter = f'user_id == "{str(user.id)}" && category == "{resolved_category}" && user_message == "{stored_message}"'
                    milvus_client.delete(collection_name, filter=delete_filter)
                    print(f"Deleted similar match: {stored_message}")
                    deleted = True
                    break
            
            if not deleted:
                print(f"No matching conflict found to delete. Looking for: '{conflicts_with}'")
                print("Available messages:", [r.get("user_message") for r in existing_records])
            
        except Exception as e:
            print(f"Error deleting conflicting information: {e}")
        
        # Store the new resolved information
        save_message_to_milvus(resolved_info, ai_response, str(user.id), resolved_category)

    # Check for store marker (general memory)
    store_match = re.search(r"<<STORE:(.*?)(?:\|category:(.*?))?>>", ai_response, re.DOTALL)
    if store_match:
        store_info = store_match.group(1).strip()
        store_category = store_match.group(2).strip() if store_match.group(2) else extract_category_from_text(store_info)
        print("LLM chose to store:", store_info, "Category:", store_category)
        save_message_to_milvus(store_info, ai_response, str(str(user.id)), store_category)

    save_message(message, ai_response, str(user.id))

    # Remove the markers from the response shown to the user
    ai_response_clean = re.sub(r"<<STORE:.*?>>", "", ai_response, flags=re.DOTALL)
    ai_response_clean = re.sub(r"<<RESOLVED:.*?>>", "", ai_response_clean, flags=re.DOTALL)
    ai_response_clean = re.sub(r"<<FUNCTION_CALL:.*?>>", "", ai_response_clean, flags=re.DOTALL)
    ai_response_clean = ai_response_clean.strip()
    
    return {
        "message": ai_response_clean,
        "function_results": function_results,
        "data_changed": data_changed,
        "updated_system_data": updated_system_data
    }

# Update the API Route for Chat
@app.post('/chat')
async def chat(request: ChatRequest):
    if not request.message or not request.user:
        raise HTTPException(status_code=400, detail="No message or user provided")
    
    # Convert user dict to User object
    try:
        # Option 1: If you have the user ID and want to fetch from database
        user_id = request.user.get('id') or request.user.get('_id')
        if user_id:
            user = User.objects(id=user_id).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
        else:
            # Option 2: Create User object from provided data
            user = User(
                id=request.user.get('id'),
                firebase_uid=request.user.get('firebase_uid'),
                username=request.user.get('username'),
                email=request.user.get('email'),
                email_verified=request.user.get('email_verified', False),
                admin=request.user.get('admin', False),
                devices=request.user.get('devices', []),
                invalid_attempts=request.user.get('invalid_attempts', 0),
                firstname=request.user.get('firstname'),
                lastname=request.user.get('lastname'),
                phone_number=request.user.get('phone_number'),
                birthday=request.user.get('birthday'),
                default_subjects=request.user.get('default_subjects', {}),
                settings=request.user.get('settings', {"ai_accessible": []})
            )
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid user data: {str(e)}")
    
    result = await ask_ai(
        message=request.message,
        user=user,
        ai_subject_ids=request.ai_subject_ids
    )
    
    return {
        "message": result["message"],
        "function_results": result.get("function_results", []),
        "data_changed": result.get("data_changed", False),
        "updated_system_data": result.get("updated_system_data")
    }