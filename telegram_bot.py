import os
import asyncio
from dotenv import load_dotenv

# Aiogram (Telegram Bot Framework)
from aiogram import Bot, Dispatcher, F, types
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup

# LangChain (Core & Chains)
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain (Data structures & Prompts)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain (Community extensions & Integrations)
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


# --- 1. State Machine Definitions ---
class UserState(StatesGroup):
    waiting_for_car_model = State()
    waiting_for_pdf = State()
    chatting = State()


# --- 2. Initialization & Configuration ---
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:postgres@localhost:5433/kia_bot_db")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)

# Set custom timeout for downloading large PDF files
session = AiohttpSession(timeout=600)
bot = Bot(token=TELEGRAM_TOKEN, session=session)
dp = Dispatcher()


# --- 3. Database & Memory Management ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return PostgresChatMessageHistory(
        connection_string=DB_CONNECTION_STRING,
        session_id=session_id,
        table_name="chat_history"
    )


# --- 4. RAG Pipeline Builder (Multi-Tenant) ---
def get_user_chain(chat_id: str, car_model: str):
    user_vector_retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"chat_id": chat_id}, "k": 3}
    )

    user_docs = vectorstore.get(where={"chat_id": chat_id}, include=["documents", "metadatas"])

    if not user_docs["documents"]:
        raise ValueError("Your database is empty. Please upload a PDF manual first.")

    bm25_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(user_docs["documents"], user_docs["metadatas"])
    ]
    user_bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    user_bm25_retriever.k = 3

    user_ensemble_retriever = EnsembleRetriever(
        retrievers=[user_bm25_retriever, user_vector_retriever],
        weights=[0.3, 0.7]
    )

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, user_ensemble_retriever, contextualize_q_prompt)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "page"],
        template="[Page in database: {page}]\n{page_content}"
    )

    qa_system_prompt = (
        f"You are a helpful AI assistant, an expert on the {car_model} vehicle. "
        "Use the following pieces of retrieved context to answer the question. "
        "Each paragraph is preceded by its 'Page in database'. "
        "YOUR TASK: Formulate an answer. If you use facts from the text, YOU MUST add to the end of your answer on a new line: "
        "'📖 *Source: page X*', where X is the 'Page in database' number PLUS 1 (since the database is zero-indexed). "
        "Cite only the pages from which you ACTUALLY took facts. Ignore pages with irrelevant garbage. "
        "If the question is just a greeting or small talk, do not cite any pages. "
        "Answer concisely and in the EXACT SAME LANGUAGE as the user's question.\n\n"
        "Context:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt, document_prompt=document_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# --- 5. Telegram Handlers ---
@dp.message(CommandStart(), StateFilter('*'))
async def send_welcome(message: types.Message, state: FSMContext):
    session_id = str(message.chat.id)
    history = get_session_history(session_id)
    history.clear()

    await state.set_state(UserState.waiting_for_car_model)
    await message.answer(
        "Hello! I'm a smart car assistant.\n\n"
        "Please write the make and model of your car (e.g., BMW X5 or Mazda 3):"
    )


@dp.message(Command("clear"), StateFilter('*'))
async def clear_history(message: types.Message, state: FSMContext):
    session_id = str(message.chat.id)
    history = get_session_history(session_id)
    history.clear()
    await message.answer("Chat context has been deleted from the database! You can start asking new questions.")


@dp.message(UserState.waiting_for_car_model)
async def process_car_model(message: types.Message, state: FSMContext):
    await state.update_data(car_model=message.text)
    await state.set_state(UserState.waiting_for_pdf)
    await message.answer(f"Great, {message.text}! Now please send me the PDF manual for this car.")


@dp.message(UserState.waiting_for_pdf, F.document)
async def process_pdf_document(message: types.Message, state: FSMContext):
    document = message.document

    if not document.file_name.lower().endswith('.pdf'):
        await message.answer("Please send a file in PDF format.")
        return

    status_msg = await message.answer(
        "Downloading the file and processing the manual... This will take a couple of minutes ⏳"
    )

    os.makedirs("user_manuals", exist_ok=True)
    file_path = f"user_manuals/{message.chat.id}_{document.file_name}"

    await bot.download(document, destination=file_path)

    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        for doc in docs:
            try:
                doc.page_content = doc.page_content.encode('latin1', errors='ignore').decode('cp1251', errors='ignore')
            except Exception:
                pass

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(docs)

        chat_id_str = str(message.chat.id)
        for chunk in chunks:
            chunk.metadata["chat_id"] = chat_id_str

        vectorstore.add_documents(chunks)

        await state.set_state(UserState.chatting)
        await status_msg.edit_text(
            "✅ The manual has been successfully loaded and added to your personal database! Ask your questions."
        )

    except Exception as e:
        await status_msg.edit_text(f"❌ Error processing PDF: {e}")


@dp.message(UserState.chatting)
async def handle_message(message: types.Message, state: FSMContext):
    session_id = str(message.chat.id)
    user_data = await state.get_data()
    car_model = user_data.get("car_model", "your car")

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        user_chain = get_user_chain(chat_id=session_id, car_model=car_model)

        response = await user_chain.ainvoke(
            {"input": message.text},
            config={"configurable": {"session_id": session_id}}
        )

        await message.answer(response["answer"], parse_mode="Markdown")

    except ValueError as ve:
        await message.answer(str(ve))
        await state.set_state(UserState.waiting_for_pdf)
    except Exception as e:
        await message.answer(f"An error occurred while processing your request: {e}")


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())