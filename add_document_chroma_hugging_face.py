import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # 修正済み
import nltk

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
load_dotenv()

def initialize_vectorstore():

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings,persist_directory="../data/DB/vectorstore")  
    return vectorstore

if __name__ == "__main__":
"""    # サンプルテキスト
    text = """生成AIは、近年急速に発展している技術の一つであり、多くの分野で注目を集めています。
    生成AIは、機械学習の手法を用いて、新しいコンテンツを生成する能力を持っています。
    特に、自然言語処理（NLP）や画像生成などの領域でその能力を発揮しています。
    例えば、GPTシリーズやDALL-Eなどのモデルは、テキストや画像を生成するために大量のデータを学習し、その結果として非常にリアルで創造的なアウトプットを提供します。"""

    # CharacterTextSplitterを初期化
    text_splitter = CharacterTextSplitter(
        chunk_size=15,         # 各チャンクの最大文字数を100に設定
        chunk_overlap=10,       # 10文字の重複を設定
        separator="\n"          # 改行で区切り
    )

    # テキストを分割
    chunks = text_splitter.split_text(text)

    # チャンクの内容を確認
    for i, chunk in enumerate(chunks):
        print(f"チャンク {i+1}:")
        print(chunk)
        print("\n---\n")"""

    
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

