import mysql.connector
import chardet
import faiss
import os
import json
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from config import OPEN_API_KEY
from typing import List
from pathlib import Path

#os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
index_name = 'hedge'
# 初始化 MySQL 連接

db_config = {
    "host" : "host",
    "user" : "user",
    "password" : "password",
    "database" : "database",
    "pool_name" : "pool_name",
    "pool_size" : 10
}

try:
    pool = mysql.connector.pooling.MySQLConnectionPool(**db_config)
    print("Successfully initialized connection pool for VectorDB database")
except Exception as e:
    print(f"An error occurred while initializing the connection pool: {e}")



embeddings = OpenAIEmbeddings()

# 獲取當前 Python 腳本的絕對路徑
#current_script_path = Path(__file__).resolve().parent

# 在當前目錄下檢查 "faiss_index" 資料夾是否存在
#faiss_index_path = current_script_path / 'faiss_index'
#faiss_index_path.mkdir(parents=True, exist_ok=True)

# 檢查 "index.faiss" 文件是否存在
#index_path = faiss_index_path / 'hedge.faiss' #存別名稱的資料庫，這邊名稱也要改
#if not index_path.exists():
#    texts = ['這是一個測試文本']
#    db = FAISS.from_texts(texts, embeddings)
    #db.save_local(folder_path=str(faiss_index_path), index_name="index_sql")
#    db.save_local(folder_path=str(faiss_index_path), index_name=index_name)

def process_and_store_documents(file_paths: List[str], db_table_name: str, original_filename: str) -> None:
    print("Starting process_and_store_documents function...")

    conn = None
    mycursor = None
    try:
        conn = pool.get_connection()
        mycursor = conn.cursor()

        if db_table_name == "q":
            index_name = "Q"
        elif db_table_name == "index":
            index_name = "index"
        elif db_table_name == "indexqa":
            index_name = "index_QA"
        elif db_table_name == "indexsynonym":
            index_name = "index_synonym"
        elif db_table_name == "flowchart":
            index_name = "flowchart"


        embeddings = OpenAIEmbeddings()

        # 獲取當前 Python 腳本的絕對路徑
        current_script_path = Path(__file__).resolve().parent

        # 在當前目錄下檢查 "faiss_index" 資料夾是否存在
        faiss_index_path = current_script_path / 'faiss_index'
        faiss_index_path.mkdir(parents=True, exist_ok=True)

        # 檢查 "index.faiss" 文件是否存在
        index_path = faiss_index_path / f'{index_name}.faiss' #存別名稱的資料庫，這邊名稱也要改

        if not index_path.exists():
            texts = ['這是一個測試文本']
            db = FAISS.from_texts(texts, embeddings)
            #db.save_local(folder_path=str(faiss_index_path), index_name="index_sql")
            db.save_local(folder_path=str(faiss_index_path), index_name=index_name)

        
        def init_txt(file_pth: str):
            loader = TextLoader(f'{file_pth}',  encoding = "utf-8")
            documents = loader.load()
            return documents
        
        def init_csv(file_pth: str):
            # my_csv_loader = CSVLoader(file_path=f'{file_pth}',encoding="utf-8", 
            #                           csv_args={'delimiter': ','
            # })
            loader = CSVLoader(f'{file_pth}', encoding = "utf-8")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=5000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
            )
            split_docs_csv = text_splitter.split_documents(documents)
            return split_docs_csv
        
        def init_xlsx(file_pth: str):
            loader = UnstructuredExcelLoader(file_pth,mode="elements")
            documents = loader.load() 
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
            )
            split_docs_xlsx = text_splitter.split_documents(documents)
            return split_docs_xlsx
        
        def init_pdf(file_pth: str):
            loader = PyPDFLoader(file_pth)
            documents = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
            )
            split_docs_pdf = text_splitter.split_documents(documents)        
            return split_docs_pdf
        
        def init_word(file_pth: str):
            loader = Docx2txtLoader(file_pth)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
            )
            split_docs_word = text_splitter.split_documents(documents)
            # Extract individual columns
            page_contents = [doc.page_content for doc in split_docs_word]

            # Create the DataFrame
            df = pd.DataFrame(page_contents)

            # Set the columns to empty names
            df.columns = ['' for _ in df.columns]
            # Get the base name from the file path (without extension)
            base_name = os.path.splitext(os.path.basename(file_pth))[0]
            
            # Save the DataFrame to a CSV with the same name as the input Word document
            csv_output_path = os.path.join(os.path.dirname(file_pth), f"{base_name}.csv")
            print(csv_output_path)
            df.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
            split_docs_word = init_csv(csv_output_path)
            return split_docs_word
        
        def init_ustruc(file_pth: str):
            loader = UnstructuredFileLoader(file_pth)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=50, separators=["\n", "\n\n", "\t", "。"]
            )
            split_docs_ustruc= text_splitter.split_documents(documents)  
            return split_docs_ustruc  

        doc_chunks = []
        for file_path in file_paths:
            extension = os.path.splitext(file_path)[-1].lower()  # Get the file extension
            if extension == '.txt':
                txt_docs = init_txt(file_path)
                doc_chunks.extend(txt_docs)
            elif extension == '.csv':
                csv_docs = init_csv(file_path)
                doc_chunks.extend(csv_docs)
            elif extension == '.xlsx':
                xlsx_docs = init_xlsx(file_path)
                doc_chunks.extend(xlsx_docs)
            elif extension == '.pdf':
                continue
            elif extension == '.docx':
                word_docs = init_word(file_path)
                doc_chunks.extend(word_docs)
            elif extension == '.gif':
                continue
            elif extension == '.jpg':
                continue
            else:
                ustruc_docs = init_ustruc(file_path)
                doc_chunks.extend(ustruc_docs)
            

        # 獲取當前 Python 腳本的絕對路徑
        current_script_path = Path(__file__).resolve().parent
        # 在當前目錄下找 "faiss_index" 資料夾
        faiss_index_path = current_script_path / 'faiss_index'
        # 加載FAISS
        try:
            docsearch = FAISS.load_local(str(faiss_index_path), embeddings, index_name=index_name)
        except Exception as e:
            print(f"Error while loading FAISS: {str(e)}")


        #過濾已存在的資料
        """
        new_doc_chunk = []
        for i in doc_chunks:
            similarity = docsearch.similarity_search_with_score(i.page_content)[0][1]
            if similarity >= 0.001:
                new_doc_chunk.append(i)
        new_clean_metadatas = [doc.metadata for doc in new_doc_chunk]
        """
        new_metadatas = [{"source": original_filename} for doc in doc_chunks]
        #print(new_metadatas)
        # 提取每個文檔的實際來源
        docsearch.add_texts([t.page_content for t in doc_chunks], metadatas = new_metadatas)
        # 查詢目前最大的 id
        mycursor.execute(f"SELECT MAX(id) FROM `{db_table_name}`")
        max_id = mycursor.fetchone()[0]  # fetchone() 返回一個 tuple，最大 id 在第一個位置
        if max_id is None:
            max_id = 0
        # 計算新的 id
        new_id = max_id + 1

        alter_table_query = f"ALTER TABLE `{db_table_name}` AUTO_INCREMENT = {new_id}"
        mycursor.execute(alter_table_query)
        conn.commit()
        # 在添加新向量到 FAISS 的同時，將相應的 metadata 和 page_content 添加到 SQL 資料庫
        for chunk, metadata in zip(doc_chunks, new_metadatas):
            page_content = chunk.page_content  # 從 doc_chunk 獲取文本內容
            sql = f"INSERT INTO `{db_table_name}` (metadata, page_content) VALUES (%s, %s)"
            val = (json.dumps(metadata, ensure_ascii=False), page_content)  # 將 metadata 轉換為 JSON 字符串
            mycursor.execute(sql, val)
            conn.commit()

        # 確保兩個資料庫的 ID 是相同的（如果需要）
        # assert faiss_index.ntotal == mycursor.lastrowid, "IDs are not synchronized!"
        docsearch.save_local(folder_path=str(faiss_index_path), index_name=index_name)
        index = faiss.read_index(str(index_path))
        # 獲取向量數據庫中的向量數量
        num_vectors = index.ntotal
        print("向量數據庫中的向量數量：", num_vectors)
        print("Finished process_and_store_documents function.")

    except Exception as e:
        print("發生錯誤:", e)

    finally:
        if mycursor is not None:
            mycursor.close()
        if conn is not None:
            conn.close()


    


def get_data_from_table(table_name):

    if table_name == "q":
        index_name = "Q"
    elif table_name == "index":
        index_name = "index"
    elif table_name == "indexqa":
        index_name = "index_QA"
    elif table_name == "indexsynonym":
        index_name = "index_synonym"
    elif table_name == "flowchart":
        index_name = "flowchart"
    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent

    # 在當前目錄下檢查 "faiss_index" 資料夾是否存在
    faiss_index_path = current_script_path / 'faiss_index'
    # 檢查 "index.faiss" 文件是否存在
    index_path = faiss_index_path / f'{index_name}.faiss' #存別名稱的資料庫，這邊名稱也要改
    conn = pool.get_connection()
    if conn.is_connected():
        print("Successfully connected to the database")
    else:
        print("Failed to connect to the database")
        return {"error": "Failed to connect to the database"}, 500

    mycursor = conn.cursor()

    try:
        query = f"SELECT * FROM `{table_name}`"  # Directly format the table name into the query
        mycursor.execute(query)

        result = mycursor.fetchall()

        index = faiss.read_index(str(index_path))
        # 獲取向量數據庫中的向量數量
        num_vectors = index.ntotal
        print("向量數據庫中的向量數量：", num_vectors)

        mycursor.close()
        conn.close()

        return result
    except Exception as e:
        print("查詢資料失敗:", e)
        return {"error": str(e)}, 500



#process_and_store_documents ([r'D:\SabrinasFolder\files\Langchain_Faiss\Langchain_Faiss\flask-server\日月光財報.txt'])


def deleteid(table_name, id):

    if table_name == "q":
        index_name = "Q"
    elif table_name == "index":
        index_name = "index"
    elif table_name == "indexqa":
        index_name = "index_QA"
    elif table_name == "indexsynonym":
        index_name = "index_synonym"
    elif table_name == "flowchart":
        index_name = "flowchart"

    # 獲取當前 Python 腳本的絕對路徑
    current_script_path = Path(__file__).resolve().parent

    # 在當前目錄下檢查 "faiss_index" 資料夾是否存在
    faiss_index_path = current_script_path / 'faiss_index'
    # 檢查 "index.faiss" 文件是否存在
    index_path = faiss_index_path / f'{index_name}.faiss' #存別名稱的資料庫，這邊名稱也要改
    conn = pool.get_connection()
    mycursor = conn.cursor()

    

    try:
        # 獲取當前 Python 腳本的絕對路徑
        current_script_path = Path(__file__).resolve().parent

        # 在當前目錄下找 "faiss_index" 資料夾
        faiss_index_path = current_script_path / 'faiss_index'
        docsearch = FAISS.load_local(faiss_index_path, embeddings, index_name=index_name)
        docsearch.delete([docsearch.index_to_docstore_id[id]])
        docsearch.save_local(folder_path=str(faiss_index_path), index_name=index_name)
        # 刪除資料庫中的相對應項目
        mycursor.execute(f"DELETE FROM `{table_name}` WHERE id = {id}")
        conn.commit()
        # 
        # # 更新後續的 id
        mycursor.execute(f"UPDATE `{table_name}` SET id = id - 1 WHERE id > {id}")
        conn.commit()
        index = faiss.read_index(str(index_path))
        # 獲取向量數據庫中的向量數量
        num_vectors = index.ntotal
        print("向量數據庫中的向量數量：", num_vectors)
    except Exception as e:
        print(f"Error occurred: {e}")
        conn.rollback()  # 如果出現錯誤，回滾事務



def id_retrieval(record_id):
    conn = pool.get_connection()
    mycursor = conn.cursor()

    try:
        query = r"SELECT * FROM `index` WHERE id = %s"
        mycursor.execute(query, (record_id,))  # 注意参数是一个元组
        result = mycursor.fetchall()

        values = [row[2] for row in result]  # 假设字段名为 'value'

        mycursor.close()
        conn.close()
        print(values[0])
        return values[0]
    except Exception as e:
        print("查詢資料失敗:", e)
        return {"error": str(e)}, 500
    

def source_search(record_id):
    conn = pool.get_connection()
    mycursor = conn.cursor()

    try:
        query = r"SELECT * FROM `index` WHERE id = %s"
        mycursor.execute(query, (record_id,))  # 注意参数是一个元组
        result = mycursor.fetchall()

        values = [row[1] for row in result]  # 假设字段名为 'value'

        mycursor.close()
        conn.close()
        print(values[0])
        return values[0]
    except Exception as e:
        print("查詢資料失敗:", e)
        return {"error": str(e)}, 500


def max_id():
    conn = pool.get_connection()
    mycursor = conn.cursor()

    try:
        query = f"SELECT max(id) FROM `index`"
        mycursor.execute(query)
        result = mycursor.fetchone()

        mycursor.close()
        conn.close()
        
        return result[0]
    except Exception as e:
        print("查詢資料失敗:", e)
        return {"error": str(e)}, 500
    

def id_QQQ(record_id):
    conn = pool.get_connection()
    mycursor = conn.cursor()

    try:
        query = "SELECT * FROM `Q` WHERE id = %s"
        mycursor.execute(query, (record_id,))  # 注意参数是一个元组
        result = mycursor.fetchone()  # 使用 fetchone() 因为我们只期望一个记录

        if result:
            # 将结果映射为字段名和值的字典
            columns = [col[0] for col in mycursor.description]
            record = dict(zip(columns, result))

            mycursor.close()
            conn.close()
            return record
        else:
            return None
    except Exception as e:
        print("查詢資料失敗:", e)
        return {"error": str(e)}, 500
    

def id_QA(id):
    conn = pool.get_connection()
    mycursor = conn.cursor()

    try:
        query = "SELECT * FROM `indexqa` WHERE id = %s"
        mycursor.execute(query, (id,))
        result = mycursor.fetchone()

        if result:
            columns = [col[0] for col in mycursor.description]
            record = dict(zip(columns, result))

            mycursor.close()
            conn.close()
            return record
        else:
            return None
    except Exception as e:
        print(f"查询 AnotherTable 失败: {e}")
        return {"error": str(e)}, 500

# 獲取當前 Python 腳本的絕對路徑
#current_script_path = Path(__file__).resolve().parent

# 在當前目錄下找 "faiss_index" 資料夾
#faiss_index_path = current_script_path / 'faiss_index'
#docsearch = FAISS.load_local(faiss_index_path, embeddings, index_name=index_name)
#docsearch.delete([docsearch.index_to_docstore_id[id]])
#docsearch.save_local(folder_path=str(faiss_index_path), index_name=index_name)


    
    
#deleteid('hedge', 9)
# 讀取已經創建的向量數據庫
#index = faiss.read_index(str(index_path))
# 獲取向量數據庫中的向量數量
#num_vectors = index.ntotal
#print("向量數據庫中的向量數量：", num_vectors)