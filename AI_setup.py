#this script stores all the info that the AI knows about itself for future use
from pymilvus import MilvusClient
import ollama
import json

client = MilvusClient(uri="http://localhost:19530")
collection_name = "AI_info"
#load AI info about self from a file
with open('AI_info.json', 'r') as file:
    data = json.load(file)

def store_ai_info(info_id, text, subject):
    #if info abut AI subject = self
    #if about the planitly subject = planitly
    response = ollama.embeddings(model="nomic-embed-text", prompt=text) #embed the text and make it vectors
    data = [{"id":info_id, "vector": response.embedding, "text": text, "subject": subject}]
    res = client.insert(
        collection_name=collection_name,
        data=data
    )
    print(res)
# Check if the collection exists
if client.has_collection(collection_name):
    client.delete(collection_name, filter="id >= 0")
    for index, info in enumerate(data):
        store_ai_info(index,info["text"],info["subject"])
else:
    print(f"Collection '{collection_name}' does not exist.")
    client.create_collection(
         collection_name=collection_name,
         dimension=768  # The vectors we will use in this demo has 768 dimensions
     )
    for index, info in data:
        store_ai_info(index,info["text"],info["subject"])
client.flush(collection_name)
print(client.get_collection_stats(collection_name))