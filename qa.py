from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import streamlit as st
from streamlit_chat import message as st_message
import os

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

# 构建回答
def generate_answer(QA_module):
    user_message = st.session_state.input_text
    message_bot = QA_module.run(user_message)

    st.session_state.history.append({"message": user_message, "is_user": True})
    st.session_state.history.append({"message": message_bot, "is_user": False})
        
# 构建Streamlit 应用程序
st.set_page_config(page_title="问答机器人", page_icon="👀")
st.header("Chat With Your Document")
st.write("您好！我是问答机器人。请问您有什么问题需要我回答吗？")

OpenAI_key = st.text_input('Input your OpenAI API Key here:')

if st.button('Submit',use_container_width=True):
    if 'sk' not in OpenAI_key:
        st.write('Please input your API Key.')
    else:
        os.environ['OPENAI_API_KEY'] = OpenAI_key
        print(OpenAI_key)
        st.write('API Key submitted successfully.')


uploaded_file = st.file_uploader(
    "Upload a document you would like to chat about",
    type=None,
    accept_multiple_files=False,
    key=None,
    help=None,
    on_change=None,
    args=None,
    kwargs=None,
    disabled=False,
    label_visibility="visible")

if uploaded_file is not None and uploaded_file.name not in os.listdir("data"):
    ls = os.listdir('data')
    for i in ls:
        c_path = os.path.join('data', i)                
        os.remove(c_path)
        
    with open("data/" + uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File uploaded successfully")
    loader = DirectoryLoader('data', glob='**/*.txt')
    print('data/' + uploaded_file.name)
    documents = loader.load()

    # 切分文本
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 获取文档嵌入
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    # 创建 QA 模型
    QA_module = VectorDBQA.from_chain_type(llm=OpenAI(),
                                        chain_type="stuff",
                                        vectorstore=docsearch)
        # 创建历史记录
    if "history" not in st.session_state:
        st.session_state.history = []
        
    # 构建回答
    st.text_input("Talk to the bot",
                    key="input_text",
                    on_change=generate_answer(QA_module))

    for chat in st.session_state.history:
        st_message(**chat)
        
elif uploaded_file is not None and uploaded_file.name in os.listdir("data"):
    st.write("File is existed.")
    loader = DirectoryLoader('data' , glob='**/*.txt')
    documents = loader.load()

    # 切分文本
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # 获取文档嵌入
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    # 创建 QA 模型
    QA_module = VectorDBQA.from_chain_type(llm=OpenAI(),
                                        chain_type="stuff",
                                        vectorstore=docsearch)
        # 创建历史记录
    if "history" not in st.session_state:
        st.session_state.history = []
        
    # 构建回答
    st.text_input("Talk to the bot",
                    key="input_text",
                    on_change=generate_answer(QA_module))

    for chat in st.session_state.history:
        st_message(**chat)
        