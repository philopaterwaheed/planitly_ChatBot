import ollama
from pymilvus import MilvusClient

milvus_client = MilvusClient(uri="http://localhost:19530")
collection_name = "AI_info"


def search_milvus(query_text):
    query_vector =  ollama.embeddings(model="nomic-embed-text", prompt=query_text).embedding

    search_results = milvus_client.search(
        collection_name,
        data=[query_vector],
        output_fields=["text"],
        limit=3
    )

    retrieved_info = [hit["entity"]["text"] for hit in search_results[0]]
    return retrieved_info

# User message
message = "hello what is your name?"

retrieved_data = search_milvus(message)
system_message = ""
if retrieved_data:
    system_message += "Here is some extra knowledge that may help you:\n"
    system_message += "\n".join(retrieved_data)
response = ollama.chat(
    model='llama3.1',
    messages=[
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': message}
    ]
)

# Print AI response
print(response['message']['content'])

#
# client.create_collection(
#     collection_name="AI_info",
#     dimension=768  # The vectors we will use in this demo has 768 dimensions
# )

# print(client.get_collection_stats("AI_info"))


# client.get_collection_stats
# docs = [
#     "Artificial intelligence was founded as an academic discipline in 1956.",
#     "Alan Turing was the first person to conduct substantial research in AI.",
#     "Born in Maida Vale, London, Turing was raised in southern England.",
# ]
#
# vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
# data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
#          for i in range(len(vectors)) ]
# res = client.insert(
#     collection_name="demo_collection",
#     data=data
# )
#
# # noinspection PyRedeclaration
# res = client.search(
#     collection_name="demo_collection",
#     data=[vectors[0]],
#     filter="subject == 'history'",
#     limit=2,
#     output_fields=["text", "subject"],
# )
# print(res)
#
# res = client.query(
#     collection_name="demo_collection",
#     filter="subject == 'history'",
#     output_fields=["text", "subject"],
# )
# print(res)
# #
# res = client.delete(
#     collection_name="AI_info",
#     filter="subject == 'self'",
# )
# # print(res)
#
# import ollama
# import torch
#
# # Get embedding for given texts
# text = "my parents named me love ; philo means love in roman"
# # text1 = "Philo adores apples."
# text1 = "my name is Philo"
#
# # Fetch embeddings from Ollama
# response = ollama.embeddings(model="nomic-embed-text", prompt=text)
# response1 = ollama.embeddings(model="nomic-embed-text", prompt=text1)
#
# # Extract embeddings from response dictionary
# embedding1 = torch.tensor(response["embedding"])  # Convert to tensor
# embedding2 = torch.tensor(response1["embedding"])  # Convert to tensor
#
# # Compute cosine similarity
# similarity = torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0)
#
# # Print results
# print(f"Similarity Score: {similarity.item():.4f}")
# print("Embedding Sample:", embedding1[:5])  # Print first 5 elements
# https://robkerr.ai/chunking-text-into-vector-databases/

