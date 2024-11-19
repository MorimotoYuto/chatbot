import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
load_dotenv()

def initialize_vectorstore():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings,persist_directory="../DB/vectorstore")  
    return vectorstore

if __name__ == "__main__":


    
    file_path = sys.argv[1]
    loader = UnstructuredFileLoader(file_path) 
    raw_docs = loader.load()
    print(raw_docs.page_content)

    text_splitter = CharacterTextSplitter(separator="\n\n",chunk_size=200, chunk_overlap=10)
    docs = text_splitter.split_documents(raw_docs)

    vectorstore = initialize_vectorstore()
    #vectorstore.add_documents(docs)

    # 追加されたドキュメントの件数を確認
    doc_count = vectorstore._collection.count()
    print(f"ベクターストアに追加されたドキュメント数: {doc_count}")
    doc = vectorstore.similarity_search("AI")  # top_kはデフォルト4
    print("\n1\n",doc[0].page_content)  
    print("\n2\n",doc[1].page_content)  

    doc = vectorstore.similarity_search_with_score("AI")
    print(f"content: {doc[0][0].page_content}", f"score: {doc[0][1]}") 
    print(f"content: {doc[1][0].page_content}", f"score: {doc[1][1]}") 

