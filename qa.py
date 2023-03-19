from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import gradio as gr
import os

loader = DirectoryLoader('data/', glob='**/*.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
os.environ['OPENAI_API_KEY'] = '<your token>'
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)


def answer_question(question):
    # 使用训练好的模型回答问题
    answer = qa.run(question)
    return answer


gradio_interface = gr.Interface(
    fn=answer_question,
    inputs="text",
    outputs="text",
    title="问答机器人",
    description="输入您的问题，机器人将回答您的问题。"
)

gradio_interface.launch()