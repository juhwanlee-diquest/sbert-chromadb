__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from sentence_transformers import SentenceTransformer
import chromadb
import pathlib
import json
import pandas as pd
import uuid

def load_data_and_insert_to_chromadb(folder_path, data_files):
    client = chromadb.PersistentClient()

    # Collection 이름
    collection_name = "dqchat"

    # 기존 컬렉션이 있는지 확인
    existing_collections = client.list_collections()

    if collection_name in [collection.name for collection in existing_collections]:
        # 이미 존재하는 컬렉션 가져오기
        dqchat = client.get_collection(collection_name)
    else:
        # 컬렉션이 없으면 새로 생성
        dqchat = client.create_collection(name=collection_name)

    model = SentenceTransformer('jhgan/ko-sroberta-multitask')

    all_data = []
    for file_name in data_files:
        
        data_path = pathlib.Path(folder_path, file_name).with_suffix('.json')
        
        #
        with data_path.open('r', encoding='utf-8') as file:
            data = json.load(file)
            
            # 각 파일의 데이터를 리스트에 추가
            for entry in data:
                entry['카테고리'] = file_name
                all_data.append(entry)

    # 모든 데이터를 하나의 DataFrame으로 변환
    df = pd.DataFrame(all_data)
    
    # 데이터프레임에서 각 행을 가져와 임베딩 및 컬렉션에 추가
    for _, row in df.iterrows():
        questions = row['질문']  # '질문' 컬럼의 내용 가져오기 (리스트 형태)
        answer = row['답변']  # '답변' 컬럼의 내용 가져오기
        clause = row['조항']  # '조항' 컬럼의 내용 가져오기
        category = row['카테고리']  # 'category' 컬럼의 내용 가져오기
        
        # 질문이 리스트 형태임을 가정하고 각각의 질문을 처리
        for question in questions:
            # 질문을 임베딩
            embedding = model.encode(question)
            
            # 메타정보 설정
            metadata = {
                'query': question,
                'answer': answer,
                'clause': clause,
                'category': category
            }
            doc_id = str(uuid.uuid4())
            # 컬렉션에 추가
            dqchat.add(
                ids=[doc_id],
                documents=[question],
                embeddings=[embedding.tolist()],
                metadatas=[metadata]
            )
            
if __name__ == "__main__":
    folder_path = '/ml_data/DQChat/'
    data_files = ['employment', 'expenditure', 'privacy', 'retirement', 'traveling', 'welfare']
    load_data_and_insert_to_chromadb(folder_path, data_files)
