# Multi-Tenant RAG Telegram Bot

A scalable, multi-tenant Telegram bot that allows users to upload their own vehicle PDF manuals and interact with them using a Retrieval-Augmented Generation (RAG) pipeline. The system ensures complete data isolation between users and maintains individual chat histories.

## Features
* **Multi-Tenancy & Data Isolation:** Each user's uploaded PDF is tagged with their unique Telegram `chat_id`. The vector database strictly filters context, ensuring users only retrieve information from their own documents.
* **Dynamic Ingestion (On-the-fly):** Users upload PDFs directly in Telegram. The bot automatically extracts text, fixes encoding issues, chunks the data, and embeds it into the vector store.
* **Hybrid Search (Ensemble Retriever):** Combines BM25 (lexical search) and ChromaDB (semantic search) for highly accurate retrieval.
* **Long-Term Memory:** Chat history is persistently stored in a PostgreSQL database, allowing the LLM to understand follow-up questions.
* **State Machine (FSM):** Structured onboarding flow (Car Model -> PDF Upload -> Chatting) using Aiogram 3.
* **Source Attribution:** The LLM strictly cites the exact page numbers from the uploaded PDF for factual verification.
* **Adaptive Language:** The bot responds in the exact language used by the user in their query.

## Tech Stack
* **LLM & Embeddings:** Groq API (gpt-oss-20b), HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Frameworks:** LangChain, Aiogram 3
* **Databases:** ChromaDB (Vector Store), PostgreSQL (Relational Memory)
* **Deployment:** Docker, Docker-Compose

## Quick Start
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/multi-tenant-rag-bot.git](https://github.com/yourusername/multi-tenant-rag-bot.git)
   cd multi-tenant-rag-bot
   
2. Create a .env file in the root directory and add your API keys:
    ```bash
   TELEGRAM_BOT_TOKEN=your_telegram_token
    GROQ_API_KEY=your_groq_api_key
   
3. Run the application using Docker Compose:
    ```bash
   docker-compose up -d --build
   
4. Open Telegram, send /start to your bot, and follow the instructions!
    ```bash
   Before committing, make sure the `.env` file (containing your tokens) is added to `.gitignore`
    so that your Telegram and Groq keys are not exposed publicly.
    Run `git add .`, commit the changes, and push to the `main` branch.
    The project is ready for release.