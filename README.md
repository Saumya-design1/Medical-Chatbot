# 🩺 Medical AI Chatbot

### Retrieval-Augmented Generation for Medical Question Answering

An AI-powered medical chatbot built using **Retrieval-Augmented Generation (RAG)**. The application retrieves relevant information from a medical knowledge base and uses a Large Language Model to generate concise, context-aware answers.

> ⚠️ **Disclaimer:** This project is for educational and experimental purposes only. It is **not a substitute for professional medical advice, diagnosis, or treatment.**

---

## ✨ Features

* 🤖 **AI-powered medical Q&A**
* 🔎 **Retrieval-Augmented Generation (RAG)**
* 📚 Uses a medical PDF as the knowledge source
* 🧠 **HuggingFace embeddings** for semantic search
* 🗄️ **Pinecone** vector database for efficient retrieval
* ⚡ **Groq LLM** for fast response generation
* 🌐 **Flask** backend
* 💬 Simple web-based chat interface
* 🎯 Retrieves the top **3 most relevant** knowledge chunks for each question
* 🔐 API keys managed through environment variables

---

## 🧠 How It Works

The chatbot follows a simple RAG pipeline:

```text
                 ┌──────────────────┐
                 │   Medical PDF    │
                 └────────┬─────────┘
                          │
                          ▼
                ┌────────────────────┐
                │   Text Extraction  │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │  Text Chunking     │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ HuggingFace        │
                │ Embeddings         │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │ Pinecone Vector DB │
                └─────────┬──────────┘
                          │
                    User Question
                          │
                          ▼
                ┌────────────────────┐
                │ Semantic Retrieval │
                └─────────┬──────────┘
                          │
                          ▼
                ┌────────────────────┐
                │     Groq LLM       │
                └─────────┬──────────┘
                          │
                          ▼
                   🤖 Final Answer
```

---

## 🛠️ Tech Stack

| Technology                 | Purpose                        |
| -------------------------- | ------------------------------ |
| 🐍 **Python 3.11**         | Core programming language      |
| 🌐 **Flask**               | Web application backend        |
| 🦜 **LangChain**           | RAG pipeline orchestration     |
| 🗄️ **Pinecone**           | Vector database                |
| 🧠 **HuggingFace**         | Text embeddings                |
| ⚡ **Groq**                 | Large Language Model inference |
| 📄 **PyPDF**               | Medical PDF document loading   |
| 🎨 **HTML / Tailwind CSS** | Frontend                       |
| ⚡ **jQuery**               | Frontend interaction           |

The project uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings.

---

## 📁 Project Structure

```text
Medical-Chatbot/
│
├── 📂 data/
│   └── Medical_book.pdf
│
├── 📂 templates/
│   └── chat.html
│
├── 📜 app.py
├── 📜 ingest.py
├── 📜 .env.example
├── 📜 .gitignore
├── 📜 README.md
└── 📜 requirements.txt
```

### `app.py`

Runs the Flask application and handles the chatbot requests.

The application connects to Pinecone, retrieves relevant documents, and passes the retrieved context to the Groq-powered LLM.

### `ingest.py`

Responsible for preparing the medical knowledge base.

It:

1. Loads the medical PDF
2. Splits the document into chunks
3. Generates embeddings
4. Creates/configures the Pinecone index
5. Uploads the vectors to Pinecone

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Medical-Chatbot.git

cd Medical-Chatbot
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Configure Environment Variables

Create a `.env` file in the project root:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

**Never commit your `.env` file to GitHub.**

The project already includes `.env` in `.gitignore` to prevent API keys from being committed.

---

# 📚 Build the Knowledge Base

Before running the chatbot for the first time, ingest the medical document into Pinecone.

Make sure your PDF is located at:

```text
data/Medical_book.pdf
```

Then run:

```bash
python ingest.py
```

The ingestion script creates the `medical-chatbot` Pinecone index if it doesn't already exist and uploads the generated document vectors.

You should eventually see:

```text
Ingestion complete!
```

---

# ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

The application runs on:

```text
http://localhost:8080
```

Open the URL in your browser and start chatting with the medical assistant.

---

# 🔍 RAG Pipeline

The chatbot uses the following retrieval configuration:

```text
User Question
      ↓
Query Embedding
      ↓
Pinecone Similarity Search
      ↓
Top 3 Relevant Chunks
      ↓
Context + User Question
      ↓
Groq LLM
      ↓
Concise Answer
```

The retriever is configured to perform similarity search and return the **top 3 results**.

The system prompt also instructs the model to answer using retrieved context and keep responses concise.

---

# 🧩 Key Components

### Embeddings

```python
HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

Converts text into numerical vector representations that can be compared semantically.

### Vector Database

```text
Pinecone
     ↓
Stores document embeddings
     ↓
Performs similarity search
```

### LLM

The chatbot uses Groq's API through LangChain to generate the final response.

---

# 🎯 Why RAG?

Instead of relying entirely on the model's pre-trained knowledge, RAG allows the chatbot to retrieve information from a specific knowledge base before generating an answer.

This helps the application:

* 📚 Ground responses in the provided medical material
* 🔎 Retrieve relevant information dynamically
* 🧠 Combine semantic search with LLM generation
* 🔄 Update the knowledge base without retraining the LLM

---

# ⚠️ Disclaimer

This chatbot is an **educational AI project**.

It should **not** be used to:

* Diagnose medical conditions
* Prescribe medication
* Replace a doctor or healthcare professional
* Make emergency medical decisions

Always consult a qualified healthcare professional for medical advice.

---

# 👨‍💻 Author

**Saumya**

Built with ❤️ using Python, LangChain, Pinecone, HuggingFace, Groq and Flask.

⭐ If you found this project interesting, consider giving the repository a star!

