from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. Starting PDF reading...")
loader = PyMuPDFLoader("data/manual509.pdf")
documents = loader.load()
print(f"Read {len(documents)} pages.")

for doc in documents:
    try:
        doc.page_content = doc.page_content.encode('latin1', errors='ignore').decode('cp1251', errors='ignore')
    except Exception as e:
        print(f"Ошибка на странице: {e}")

print("\n--- ТЕСТОВЫЙ КУСОК ТЕКСТА ИЗ PDF ---")
print(documents[10].page_content[:300])
print("------------------------------------\n")

print("2. Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
print(f"Text split into {len(chunks)} chunks.")

print("3. Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("4. Creating vector database and saving data...")
persist_directory = "chroma_db"
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Done! Vector database successfully created.")