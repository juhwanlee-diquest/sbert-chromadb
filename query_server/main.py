__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict


app = FastAPI()
# Allow CORS
origins = [
    "*",  # Allow all origins (for testing purposes, specify your domains in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 'cuda:0'


# ChromaDB 클라이언트 및 모델 설정
database_path = '/data/juhwan/sbert-chromadb/chroma'
client = chromadb.PersistentClient(path=database_path)
collection_name = "dqchat"
dqchat = client.get_collection(collection_name)
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
model.to(device)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class QueryResult(BaseModel):
    query: str
    answer: str

class QueryResponse(BaseModel):
    query: str
    results: List[dict]

@app.post("/post/query/", response_model=QueryResponse)
async def query_db(request: QueryRequest):
    query_embedding = model.encode(request.query).tolist()
    
    results = dqchat.query(
        query_embeddings=[query_embedding],
        n_results=request.top_k
    )

    if not results:
        raise HTTPException(status_code=404, detail="No results found")
    
    response = QueryResponse(
        query=request.query,
        results=[
            QueryResult(
                query=result_meta['query'],
                answer=result_meta['answer']
            ).dict()
            for result_meta in results['metadatas'][0]
        ]
    )


    return response


