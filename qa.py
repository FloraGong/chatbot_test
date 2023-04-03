import streamlit as st
from streamlit_chat import message as st_message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import os

# 加载文档
loader = DirectoryLoader('data', glob='**/*.txt')
documents = loader.load()

# 切分文本
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 获取文档嵌入
os.environ[
    'OPENAI_API_KEY'] = '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'
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)

# 创建 QA 模型
qa = VectorDBQA.from_chain_type(llm=OpenAI(),
                                chain_type="stuff",
                                vectorstore=docsearch)
# 创建历史记录
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit 应用程序
st.title("问答机器人")
st.write("您好！我是问答机器人。请问您有什么问题需要我回答吗？")


# 构建回答
def generate_answer():

    user_message = st.session_state.input_text
    message_bot = qa.run(user_message)

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})


st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

for chat in st.session_state.history:
    st_message(**chat)
