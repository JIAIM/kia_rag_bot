import os
import asyncio
from dotenv import load_dotenv

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

print("Загрузка переменных окружения...")
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Строка подключения к PostgreSQL (замени postgres:postgres на свои логин и пароль)
# Формат: postgresql://пользователь:пароль@хост:порт/имя_базы
DB_CONNECTION_STRING = "postgresql://postgres:postgres@localhost:5433/kia_bot_db"

print("Подключение к RAG-конвейеру...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("Сборка лексического индекса (BM25)...")
db_data = vectorstore.get(include=["documents", "metadatas"])
all_documents = [
    Document(page_content=doc, metadata=meta)
    for doc, meta in zip(db_data["documents"], db_data["metadatas"])
]
bm25_retriever = BM25Retriever.from_documents(all_documents)
bm25_retriever.k = 3

print("Настройка EnsembleRetriever...")
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)

contextualize_q_system_prompt = (
    "Учитывая историю чата и последний вопрос пользователя, "
    "который может ссылаться на контекст в истории чата, "
    "сформулируй самостоятельный вопрос, который можно понять "
    "без истории чата. НЕ отвечай на вопрос, "
    "просто переформулируй его, если нужно, иначе верни как есть."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])
history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, contextualize_q_prompt)

document_prompt = PromptTemplate(
    input_variables=["page_content", "page"],
    template="[Страница в базе: {page}]\n{page_content}"
)

qa_system_prompt = (
    "Ты — полезный ИИ-ассистент, эксперт по автомобилям Kia Cee'd. "
    "Используй следующий контекст для ответа на вопрос. Перед каждым абзацем указана его 'Страница в базе'. "
    "ТВОЯ ЗАДАЧА: Сформируй ответ. Если ты брал факты из текста, ОБЯЗАТЕЛЬНО добавь в конец ответа с новой строки: "
    "'📖 *Источник: стр. X*', где X — это номер 'Страницы в базе' ПЛЮС 1 (так как база считает с нуля). "
    "Указывай только те страницы, из которых ты РЕАЛЬНО взял факты. Проигнорируй страницы с нерелевантным мусором. "
    "Если вопрос — это просто приветствие или болтовня, не указывай страницы вообще. "
    "Отвечай кратко на русском языке.\n\n"
    "Контекст:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(
    llm,
    qa_prompt,
    document_prompt=document_prompt
)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- БЛОК ДОЛГОВРЕМЕННОЙ ПАМЯТИ (PostgreSQL) ---
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return PostgresChatMessageHistory(
        connection_string=DB_CONNECTION_STRING,
        session_id=session_id,
        table_name="chat_history" # LangChain сам создаст эту таблицу
    )

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(CommandStart())
async def send_welcome(message: types.Message):
    await message.answer("Привет! Я умный RAG-бот. Задавай вопросы по мануалу, и я найду ответ с учетом контекста нашей беседы.")

@dp.message(Command("clear"))
async def clear_history(message: types.Message):
    session_id = str(message.chat.id)
    # Прямое обращение к БД для удаления истории конкретного юзера
    history = get_session_history(session_id)
    history.clear()
    await message.answer("Контекст диалога удален из базы данных! Начнем с чистого листа.")

@dp.message()
async def handle_message(message: types.Message):
    session_id = str(message.chat.id)
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        response = await conversational_rag_chain.ainvoke(
            {"input": message.text},
            config={"configurable": {"session_id": session_id}}
        )

        await message.answer(response["answer"], parse_mode="Markdown")

    except Exception as e:
        await message.answer(f"Произошла ошибка при обработке запроса: {e}")

async def main():
    print("Telegram-бот на Aiogram запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())