import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from add_document_chroma_hugging_face import initialize_vectorstore 

# .env ファイルから環境変数を読み込む
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    huggingfacehub_api_token=huggingface_token,
)

model = ChatHuggingFace(llm=llm)
vectorstore = initialize_vectorstore()

# メッセージを (タイプ, コンテンツ) ペアのリストに変換する関数
def convert_messages(messages):
    formatted_messages = []
    for message in messages:
        if isinstance(message, SystemMessage):
            formatted_messages.append(("system", message.content))
        elif isinstance(message, HumanMessage):
            formatted_messages.append(("human", message.content))
        elif isinstance(message, AIMessage):
            formatted_messages.append(("ai", message.content))
    return formatted_messages

# プロンプトのテンプレートを定義
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        ("user","Bob"),
        ("placeholder", "{agent_scratchpad}"),
        ("placeholder", "{messeages}")
        
    ]
)




if __name__ == "__main__":
    # チャット履歴を保持するオブジェクト
    demo_ephemeral_chat_history_for_chain = ChatMessageHistory()


    # 初期メッセージを渡して実行
    config = {"configurable": {"session_id": "unused"}}
    i=1

    while True:
        user_input = input(str(i)+"\nYou: ")
        if user_input.lower() == "exit":
            break  # "exit"と入力された場合、ループを終了

        demo_ephemeral_chat_history_for_chain.add_user_message(user_input)
        history = demo_ephemeral_chat_history_for_chain.messages
        retriever = vectorstore.similarity_search(user_input)  # top_kはデフォルト4
        #print("\n1\n",retriever[0].page_content)  
        #print("\n2\n",retriever[1].page_content)  

        #print("Message History:", history)
        
        prompt_value = prompt.invoke(
            {"messeages": history,
            "agent_scratchpad": [retriever[0].page_content,retriever[1].page_content]
            },
            config
        )
        #print("\nprompt_value:", prompt_value, "\n\n")
        formatted_prompt = convert_messages(prompt_value.messages)
        print("\nformatted_prompt:", formatted_prompt, "\n\n")

        ai_msg = model.invoke(

            formatted_prompt

        )
        print("AI:", ai_msg.content, "\n")
        demo_ephemeral_chat_history_for_chain.add_ai_message(ai_msg.content)
        i+=1
        