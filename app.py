import streamlit as st
import os
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    # load_dotenv()
    st.set_page_config(page_title="Smart Chat", page_icon=":briefcase:")
    st.write(css, unsafe_allow_html=True)

    st.header("Lecture Chatbot")
    company = "Lec1"
    # company = st.sidebar.selectbox("Select Lecture", companies)

    # if "selected_company" not in st.session_state or st.session_state.selected_company != company:
    st.session_state.selected_company = company
    vectorstore_path = f"./annual_reports/Transcripts/{company}/faiss_index"
    vectorstore = FAISS.load_local(vectorstore_path, allow_dangerous_deserialization=True, embeddings=OpenAIEmbeddings())
    st.session_state.conversation = get_conversation_chain(vectorstore)
    st.session_state[f'chat_history_{company}'] = []

    user_question = st.text_input("Ask a question about " + company + "'s documents:")
    if user_question:
        handle_userinput(user_question)

    if st.button('Reset Chat'):
        st.session_state[f'chat_history_{company}'] = []
        vectorstore_path = f"./annual_reports/Transcripts/{company}/faiss_index"
        vectorstore = FAISS.load_local(vectorstore_path, allow_dangerous_deserialization=True, embeddings=OpenAIEmbeddings())
        st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()
