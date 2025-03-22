import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
# from langchain_community.text_splitter import CharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.chat_models import ChatOpenAI
# from langchain_community.memory import ConversationBufferMemory
# from langchain_community.chains import ConversationalRetrievalChain
# from htmlTemplates import css, bot_template, user_template
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


# Function to extract text from a list of PDF documents
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split the extracted text into smaller chunks
def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create a vector store from text chunks using OpenAI embeddings and FAISS
def create_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# Function to initialize a conversation chain using a vector store
def initialize_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True,  # Ensure source documents are returned
        output_key='answer'  # Explicitly set the output key
    )
    return conversation_chain


# Function to process user input and generate a response
def process_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Check if the response is relevant to the document content
    if response['source_documents']:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.write(bot_template.replace(
            "{{MSG}}", "Sorry, I can't assist with that. The question is outside the contents of the uploaded documents."), unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        process_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = extract_text_from_pdfs(pdf_docs)
                st.write("Extracted text:", raw_text)  # Debugging statement

                # get the text chunks
                text_chunks = split_text_into_chunks(raw_text)
                st.write("Text chunks:", text_chunks)  # Debugging statement

                # create vector store
                if text_chunks:
                    vectorstore = create_vector_store(text_chunks)
                    # create conversation chain
                    st.session_state.conversation = initialize_conversation_chain(
                        vectorstore)
                else:
                    st.write("No text chunks found. Please check the PDF content.")


if __name__ == '__main__':
    main()