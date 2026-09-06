import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

print("Loading environment variables...")
load_dotenv()


print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

print("Connecting to vector database...")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Initializing LLM...")
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)

system_prompt = (
    "You are a helpful assistant for question-answering tasks based on the provided car manual. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't find the answer in the context, say that you don't know. "
    "Answer in the same language as the user's question."
    "\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("\nБот готов! Задавай вопросы по мануалу (или напиши 'exit' для выхода).")
while True:
    query = input("\nВопрос: ")
    if query.lower() in ['exit', 'выход', 'quit']:
        print("Завершение работы.")
        break

    print("Searching for answer...")
    response = rag_chain.invoke({"input": query})

    print("\n=== ОТВЕТ ===")
    print(response["answer"])
    print("=============\n")