from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

import ollama
import json

client = MilvusClient(uri="http://localhost:19530")

# --- AI_info collection setup ---
collection_name = "AI_info"

# Load AI info about self from a file
with open('AI_info.json', 'r') as file:
    data = json.load(file)

# Define schema for AI_info
ai_info_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=256)
]
ai_info_schema = CollectionSchema(fields=ai_info_fields, description="AI knowledge base")

# Drop and recreate collection if it exists
if client.has_collection(collection_name):
    client.drop_collection(collection_name)
client.create_collection(collection_name=collection_name, schema=ai_info_schema)

def store_ai_info(info_id, text, subject):
    """Store AI information with embeddings in the collection"""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    if isinstance(response, dict) and "embedding" in response:
        embedding = response["embedding"]
    elif hasattr(response, "embedding"):
        embedding = response.embedding
    else:
        print("Warning: Unexpected response format from ollama:", response)
        embedding = response  # Fallback, assuming response is directly the embedding
    data = [{
        "id": info_id,
        "vector": embedding,
        "text": text,
        "subject": subject
    }]
    res = client.insert(collection_name=collection_name, data=data)
    return res

# Store all AI info
for index, info in enumerate(data):
    store_ai_info(index, info["text"], info["subject"])

client.flush(collection_name)
print("AI_info stats:", client.get_collection_stats(collection_name))

# --- user_info collection setup (with auto id) ---
user_collection = "user_info"

# Drop and recreate user collection
if client.has_collection(user_collection):
    client.drop_collection(user_collection)

user_fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="user_message", dtype=DataType.VARCHAR, max_length=1024),
]
user_schema = CollectionSchema(fields=user_fields, description="User chat history with auto id")

client.create_collection(collection_name=user_collection, schema=user_schema)
print(f"Collection '{user_collection}' created with auto-generated id.")

index_params = MilvusClient.prepare_index_params()

index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="IVF_FLAT",
    index_name="vector_index",
    params={ "nlist": 128 }
)

client.create_index(
    collection_name=user_collection,
    index_params=index_params,
)
print(f"Index created on '{user_collection}.embedding'.")
