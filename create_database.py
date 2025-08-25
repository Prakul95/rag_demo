from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import shutil
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from setup import  initialize_system
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

CHROMA_PATH = "chroma"
data_path = "data/books/"
load_dotenv()

def main():
    generate_data_store()
GOOGLE_API_KEY = initialize_system()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    loader = DirectoryLoader(data_path, glob="*.md", loader_cls = TextLoader, show_progress=True,)
    if loader:
        print(os.getcwd())
    documents = loader.load()
    return documents



def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 500,length_function = len,
    add_start_index = True)

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks
def save_to_chroma(chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", key = GOOGLE_API_KEY)
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

    return 
if __name__ == "__main__":
    main()