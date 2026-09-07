import os
import asyncio
from dotenv import load_dotenv

from aiogram import F
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart, Command

# LangChain импорты
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# 1. Определение состояний FSM
class UserState(StatesGroup):
    waiting_for_car_model = State()
    waiting_for_pdf = State()
    chatting = State()


print("Загрузка переменных окружения...")
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING", "postgresql://postgres:postgres@localhost:5433/kia_bot_db")

# 2. Глобальные подключения (только те, что общие для всех)
print("Подключение к векторной базе ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


# 3. Управление историей сессий в PostgreSQL
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return PostgresChatMessageHistory(
        connection_string=DB_CONNECTION_STRING,
        session_id=session_id,
        table_name="chat_history"
    )


# 4. ДИНАМИЧЕСКАЯ СБОРКА ЦЕПИ ПОЛЬЗОВАТЕЛЯ (Изоляция данных)
def get_user_chain(chat_id: str, car_model: str):
    # Векторный поиск строго по chat_id
    user_vector_retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"chat_id": chat_id}, "k": 3}
    )

    # Выгружаем документы конкретного юзера для BM25
    user_docs = vectorstore.get(where={"chat_id": chat_id}, include=["documents", "metadatas"])

    if not user_docs["documents"]:
        raise ValueError("Ваша база данных пуста. Пожалуйста, загрузите PDF-мануал.")

    bm25_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(user_docs["documents"], user_docs["metadatas"])
    ]
    user_bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    user_bm25_retriever.k = 3

    # Персональный гибридный поиск
    user_ensemble_retriever = EnsembleRetriever(
        retrievers=[user_bm25_retriever, user_vector_retriever],
        weights=[0.3, 0.7]
    )

    # Промпты
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Учитывая историю чата и последний вопрос пользователя, сформулируй самостоятельный вопрос, который можно понять без истории чата. НЕ отвечай на вопрос, просто переформулируй его, если нужно, иначе верни как есть."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, user_ensemble_retriever, contextualize_q_prompt)

    document_prompt = PromptTemplate(
        input_variables=["page_content", "page"],
        template="[Страница в базе: {page}]\n{page_content}"
    )

    # Динамический системный промпт с маркой авто пользователя
    qa_system_prompt = (
        f"Ты — полезный ИИ-ассистент, эксперт по автомобилю {car_model}. "
        "Используй следующий контекст для ответа на вопрос. Перед каждым абзацем указана его 'Страница в базе'. "
        "ТВОЯ ЗАДАЧА: Сформируй ответ. Если ты брал факты из текста, ОБЯЗАТЕЛЬНО добавь в конец ответа с новой строки: "
        "'📖 *Источник: стр. X*', где X — это номер 'Страницы в базе' ПЛЮС 1 (так как база считает с нуля). "
        "Указывай только те страницы, из которых ты РЕАЛЬНО взял факты. Проигнорируй страницы с нерелевантным мусором. "
        "Если вопрос — это просто приветствие или болтовня, не указывай страницы вообще. "
        "Отвечай кратко на русском языке.\n\n"
        "Контекст:\n{{context}}"
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


# 5. Хэндлеры Telegram

@dp.message(CommandStart())
async def send_welcome(message: types.Message, state: FSMContext):
    session_id = str(message.chat.id)
    history = get_session_history(session_id)
    history.clear()

    await state.set_state(UserState.waiting_for_car_model)
    await message.answer(
        "Привет! Я умный ассистент по автомобилям.\n\nНапиши марку и модель твоего авто (например: BMW X5 или Mazda 3):")


@dp.message(Command("clear"))
async def clear_history(message: types.Message, state: FSMContext):
    session_id = str(message.chat.id)
    history = get_session_history(session_id)
    history.clear()
    await message.answer("Контекст диалога удален из базы данных! Можешь задавать новые вопросы.")


@dp.message(UserState.waiting_for_car_model)
async def process_car_model(message: types.Message, state: FSMContext):
    await state.update_data(car_model=message.text)
    await state.set_state(UserState.waiting_for_pdf)
    await message.answer(f"Отлично, {message.text}! Теперь отправь мне PDF-файл с мануалом для этой машины.")


@dp.message(UserState.waiting_for_pdf, F.document)
async def process_pdf_document(message: types.Message, state: FSMContext):
    document = message.document

    if not document.file_name.lower().endswith('.pdf'):
        await message.answer("Пожалуйста, отправь файл в формате PDF.")
        return

    status_msg = await message.answer("Скачиваю файл и нарезаю мануал... Это займет пару минут ⏳")

    os.makedirs("user_manuals", exist_ok=True)
    file_path = f"user_manuals/{message.chat.id}_{document.file_name}"
    await bot.download(document, destination=file_path)

    try:
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Исправление кодировки
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

        # ИЗОЛЯЦИЯ ДАННЫХ: Привязываем каждый кусок к текущему chat_id
        chat_id_str = str(message.chat.id)
        for chunk in chunks:
            chunk.metadata["chat_id"] = chat_id_str

        # Загрузка в общую векторную базу
        vectorstore.add_documents(chunks)

        await state.set_state(UserState.chatting)
        await status_msg.edit_text(
            "✅ Мануал успешно загружен, изучен и добавлен в твою личную базу! Задавай свои вопросы.")

    except Exception as e:
        await status_msg.edit_text(f"❌ Ошибка при обработке PDF: {e}")


@dp.message(UserState.chatting)
async def handle_message(message: types.Message, state: FSMContext):
    session_id = str(message.chat.id)
    user_data = await state.get_data()
    car_model = user_data.get("car_model", "твоего автомобиля")

    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        # Динамическая генерация цепи под текущего пользователя
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
        await message.answer(f"Произошла ошибка при обработке запроса: {e}")


async def main():
    print("Telegram-бот на Aiogram запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())