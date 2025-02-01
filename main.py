import gradio as gr
import re
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM, ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage

from gradio import ChatMessage

# prompt Template
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA

from fastapi import FastAPI
import pdfplumber
import os
from pdf2image import convert_from_path
import cv2
import pytesseract

# cuda visible devices = -1
# 1) ПРЕДОБРАБОТКА: загрузка PDF, создание векторного хранилища и retriever


# 2) СИСТЕМНЫЙ ПРОМПТ ДЛЯ Llama 3.1
system_prompt = """Ты – виртуальный помощник, работающий в режиме RAG (Retrieval-Augmented Generation), который даёт ответы строго на основе предоставленных выдержек из загруженного документаы.

Основные правила:

1. Отвечай только по предоставленному контексту. Если нужная информация отсутствует во фрагментах, сообщай, что у тебя нет достаточных сведений.
2. Не добавляй никаких домыслов или внешних знаний. Избегай ссылок на источники вне текущего контекста.
3. Если вопрос выходит за пределы данных, предоставленных в контексте, прямо укажи, что «в предоставленном тексте нет информации по этому вопросу».
4. Старайся формулировать ответы лаконично, ясно выделяя ключевые моменты из контекста.
5. Не передавай конфиденциальных данных и не измышляй факты.
6. Оборачивай код в тройные обратные кавычки (Например, ```bash) и указывай источник, если это необходимо.
"""
# Пример поведения

# Вопрос: "как отменить погашение аварии?"
# Контекст: ... (выдержки из документа) ...
# Твой Ответ: «Для отмены погашения аварии выполните следующие шаги:

# Нажмите кнопку на панели режимов отображения.
# В выпадающем списке вверху окна выберите "Активные аварии".
# Выберите погашенную аварию, которую нужно вернуть в предыдущее состояние, и нажмите на неё правой кнопкой мыши.
# В открывшемся контекстном меню выберите "Отменить погашение".
# После выполнения этих действий авария перейдёт в состояние, которое было до погашения.
# Также отменить погашение аварии можно с помощью REST запроса "Отменить погашение аварии"
# Для этого запроса требуется ID аварии.
# ```bash
# login=<...>
# password=<...>
# incident_id=<...>
# saymon_hostname=<...>
# url=https://$saymon_hostname/node/api/incidents/$incident_id/undo-clear
# curl -X POST $url -u $login:$password
# ```
# Ссылка на источник: Руководство пользователя, страница 13.»
# """


# 3) ФУНКЦИИ ДЛЯ RAG
def format_docs(docs):
    """
    Форматирует список документов в один текстовый блок,
    чтобы передать его как контекст в Llama через Ollama.
    """
    return "\n\n".join(
        f"Текст контекста: {doc.page_content}\nСтраница: {doc.metadata.get('page_label', 'N/A')}"
        for doc in docs
    )


def ollama_llm(question, context, llm, stream):
    """
    Вызывает модель Llama 3.1 (через ollama.chat) с учётом системного промпта и контекста.
    """
    formatted_prompt = f"""Вопрос: "{question}"\nКонтекст: {context}"""
    sys_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=formatted_prompt)
    if stream:
        stream = llm.stream([sys_message, human_message])
        return stream
    else:
        print("Getting response..")
        response = llm.invoke([sys_message, human_message])
        return response


def rag_chain(question: str, retriever, llm, stream=False) -> str:
    """
    Основная функция: по вопросу извлекает релевантные документы и передаёт их в LLM.
    Возвращает финальный ответ.
    """
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)

    return ollama_llm(question, formatted_context, llm, stream)


IS_STREAM = True


# 4) GRADIO UI
def handle_user_question(user_question, chat_history, retriever, llm):
    """
    Генераторная функция. Сначала добавляет вопрос в чат,
    затем постепенно "достраивает" ответ.
    """

    # 1) Добавляем вопрос в историю (пустой ответ)
    chat_history.append(ChatMessage(role="user", content=user_question))
    # 2) Показываем плейсхолдер (например, «генерация…»)
    chat_history.append(ChatMessage(role="assistant", content="🧠 Генерация ответа..."))
    yield "", chat_history  # рендерим, чтобы пользователь видел placeholder

    # Получаем генератор ответа от rag_chain
    stream = rag_chain(user_question, retriever, llm, IS_STREAM)

    partial_answer = ""
    for chunk in stream:  # chunk -- это строка (часть ответа), приходящая от Ollama
        # response = chunk["message"]["content"]
        partial_answer += chunk
        # обновляем последний элемент chat_history на "текущий ответ"
        chat_history[-1] = ChatMessage(role="assistant", content=partial_answer)
        yield "", chat_history


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from pdf
    """
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    return text


def ocr_pdf(pdf_path):
    images = convert_from_path(
        pdf_path,
        poppler_path=r"C:\\Users\\Davron\\Desktop\\poppler-24.02.0\\Library\\bin",
    )
    full_text = ""
    for i, image in enumerate(images):
        if os.name == "nt":
            pytesseract.pytesseract.tesseract_cmd = (
                "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            )
        text = pytesseract.image_to_string(image, lang="rus+eng")
        full_text += text

    return full_text


def update_retriever(filepath, chat_history):

    gr.Info("Загружаем PDF-файл...")
    # update retriever
    text = extract_text_from_pdf(filepath)
    if len(text) < 2000:
        gr.Info("Нераспознанный текст. Попробуем распознать с помощью OCR...")
        text = ocr_pdf(filepath)
        init_docs = [Document(page_content=text)]
    else:
        gr.Info("Текст успешно распознан. Продолжаем...")
        loader = PyPDFLoader(filepath)
        init_docs = loader.load()

    # Разбиваем документы на chunks (пример с RecursiveCharacterTextSplitter)
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    split_docs = recursive_splitter.split_documents(init_docs)

    # Создаём векторное хранилище (FAISS). Вам нужен заранее определённый Embeddings

    # Create Embeddings using Ollama
    embeddings = OllamaEmbeddings(model="bge-m3", base_url="http://127.0.0.1:11434")

    vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)
    retriever = vector_store.as_retriever()
    gr.Info("PDF-файл успешно загружен! Можете задавать вопросы.")
    llm = OllamaLLM(
        model="llama3.1:8b",  # "deepseek-r1:8b",
        # num_gpu=0,
        base_url="http://127.0.0.1:11434",
        # num_thread=12,
    )
    chat_history.append(
        ChatMessage(
            role="assistant",
            content="Теперь можешь задавать мне вопросы по загруженному документу.",
        )
    )
    msg = gr.Textbox(
        label="Напишите свой вопрос, касающийся загруженного PDF", visible=True
    )
    submit = gr.Button("➤ Отправить", visible=True)
    return retriever, llm, chat_history, msg, submit


def delete_files():
    gr.Info("Файл удалён.")
    msg = gr.Textbox(
        label="Напишите свой вопрос, касающийся загруженного PDF", visible=False
    )
    submit = gr.Button("➤ Отправить", visible=False)
    return msg, submit


with gr.Blocks() as user_block:
    gr.Markdown("## Чат-бот (RAG) по документации")
    with gr.Row():
        pdf_file = gr.File(
            label="Загрузите PDF-файл ",
            type="filepath",
            file_count="single",
            file_types=[".pdf"],
            interactive=True,
        )
        retriever = gr.State()
        llm = gr.State()
    chatbot = gr.Chatbot(type="messages", label="DOC AI")

    msg = gr.Textbox(
        label="Напишите свой вопрос, касающийся загруженного PDF", visible=False
    )
    submit = gr.Button("➤ Отправить", visible=False)

    # chat_history будет хранить список (вопрос, ответ)
    # chat_history = gr.State([])

    # Привязываем функцию handle_user_question
    msg.submit(
        handle_user_question,
        inputs=[msg, chatbot, retriever, llm],
        outputs=[msg, chatbot],
    )
    submit.click(
        handle_user_question,
        inputs=[msg, chatbot, retriever, llm],
        outputs=[msg, chatbot],
    )
    pdf_file.upload(
        update_retriever,
        inputs=[pdf_file, chatbot],
        outputs=[retriever, llm, chatbot, msg, submit],
    )
    pdf_file.clear(delete_files, outputs=[msg, submit])
    # Кнопка очистки истории
    # clear_btn = gr.ClearButton([msg, chatbot], value="🗑️ Очистить историю")

if __name__ == "__main__":
    user_block.launch(share=False)
