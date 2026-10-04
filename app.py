import os
import warnings
import logging
from flask import Flask, render_template, request
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Mute unnecessary warnings in terminal
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

load_dotenv()

app = Flask(__name__)

# 1. Initialize Embeddings & Vector Store
print("Initializing Embeddings and Vector Store...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = PineconeVectorStore(index_name="medical-chatbot", embedding=embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# 2. Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0.3
)

# 3. System Prompt & RAG Chain
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.form["msg"]
        print(f"User Question: {user_input}")
        
        response = rag_chain.invoke({"input": user_input})
        answer = response.get("answer", "Sorry, I couldn't generate an answer.")
        
        print(f"Bot Answer: {answer}")
        return str(answer)
    except Exception as e:
        print(f"Error handling request: {e}")
        return f"An error occurred: {str(e)}"


if __name__ == "__main__":
    # use_reloader=False prevents duplicated terminal logs
    app.run(host="0.0.0.0", port=8080, debug=True, use_reloader=False)