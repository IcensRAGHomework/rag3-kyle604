import datetime
import chromadb
import traceback
import csv

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    collection = demo("TRAVEL")
    if collection.count() > 0:
        return collection

    file_name = "COA_OpenData.csv"

    ids = []
    documents = []
    metadatas = []
    r = csv.DictReader(open(file_name, encoding="UTF-8-sig"))
    for row in r:
        ids.append(row['ID'])
        documents.append(row['HostWords'])
        metadatas.append({
                "file_name": file_name,
                "name": row['Name'],
                "type": row['Type'],
                "address": row['Address'],
                "tel": row['Tel'],
                "city": row['City'],
                "town": row['Town'],
                "date": datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp(),
            })
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    ret = collection.query(
        query_texts=[question],
        n_results=10,
        where={"$and" : [
                {"city": {"$in": city}},
                {"type": {"$in": store_type}},
                {"date": {"$gte": start_date.timestamp()}},
                {"date": {"$lte": end_date.timestamp()}},
        ]},
        include=["metadatas", "distances"],
    )
    t = zip(ret["distances"][0], ret["metadatas"][0])
    return [i[1]["name"] for i in t if i[0] <= 1 - 0.8]
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    pass
    
def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

if __name__ == "__main__":
    #generate_hw01()
    print(generate_hw02("我想要找有關茶餐點的店家", ["宜蘭縣", "新北市"], ["美食"], datetime.datetime(2024, 4, 1), datetime.datetime(2024, 5, 1)))