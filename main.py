import gradio as gr
import re
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from fastapi import FastAPI

# 1) ПРЕДОБРАБОТКА: загрузка PDF, создание векторного хранилища и retriever
filepath = "data/user-guide.pdf"
loader = PyPDFLoader(filepath)
init_docs = loader.load()

# Разбиваем документы на chunks
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = recursive_splitter.split_documents(init_docs)


# Создаём векторизатор Ollama
embeddings = OllamaEmbeddings(
    model="bge-m3",
)
# Создаём векторное хранилище (FAISS).
vector_store = FAISS.from_documents(documents=split_docs, embedding=embeddings)
retriever = vector_store.as_retriever()

# 2) СИСТЕМНЫЙ ПРОМПТ
system_prompt = """Ты – виртуальный помощник, работающий в режиме RAG (Retrieval-Augmented Generation), который даёт ответы строго на основе предоставленных выдержек из «Руководства пользователя».

Основные правила:

1. Отвечай только по предоставленному контексту. Если нужная информация отсутствует во фрагментах, сообщай, что у тебя нет достаточных сведений.
2. Не добавляй никаких домыслов или внешних знаний. Избегай ссылок на источники вне текущего контекста.
3. Если вопрос выходит за пределы данных, предоставленных в контексте, прямо укажи, что «в предоставленном тексте нет информации по этому вопросу».
4. Старайся формулировать ответы лаконично, ясно выделяя ключевые моменты из контекста.
5. Не передавай конфиденциальных данных и не измышляй факты.
6. Оборачивай код в тройные обратные кавычки (Например, ```bash) и указывай источник, если это необходимо.

Пример поведения

Вопрос: «как отменить погашение аварии?»
Контекст: ... (выдержки из «Руководства пользователя») ...
Ответ: «Для отмены погашения аварии выполните следующие шаги:

Нажмите кнопку на панели режимов отображения.
В выпадающем списке вверху окна выберите "Активные аварии".
Выберите погашенную аварию, которую нужно вернуть в предыдущее состояние, и нажмите на неё правой кнопкой мыши.
В открывшемся контекстном меню выберите "Отменить погашение".
После выполнения этих действий авария перейдёт в состояние, которое было до погашения.
Также отменить погашение аварии можно с помощью REST запроса "Отменить погашение аварии"
Для этого запроса требуется ID аварии.
```bash
login=<...>
password=<...>
incident_id=<...>
saymon_hostname=<...>
url=https://$saymon_hostname/node/api/incidents/$incident_id/undo-clear
curl -X POST $url -u $login:$password
```
Ссылка на источник: Руководство пользователя, страница 13.»
"""


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


def ollama_llm(question, context):
    """
    Вызывает модель Llama 3.1 (через ollama.chat) с учётом системного промпта и контекста.
    """
    formatted_prompt = f"""Вопрос: "{question}"\nКонтекст: {context}"""
    stream = ollama.chat(
        model="llama3.1",
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_prompt},
        ],
    )
    return stream


def rag_chain(question: str) -> str:
    """
    Основная функция: по вопросу извлекает релевантные документы и передаёт их в LLM.
    Возвращает финальный ответ.
    """
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)


# 4) GRADIO UI
def handle_user_question(user_question, chat_history):
    """
    Генераторная функция. Сначала добавляет вопрос в чат,
    затем постепенно "достраивает" ответ.
    """
    # 1) Добавляем вопрос в историю (пустой ответ)
    chat_history.append((user_question, ""))
    yield "", chat_history  # рендерим, чтобы сразу увидеть вопрос

    # 2) Показываем плейсхолдер
    chat_history[-1] = (user_question, "печатает...")
    yield "", chat_history  # рендерим, чтобы пользователь видел placeholder

    # Получаем генератор ответа от rag_chain
    stream = rag_chain(user_question)

    partial_answer = ""
    for chunk in stream:  # chunk -- это строка (часть ответа), приходящая от Ollama
        response = chunk["message"]["content"]
        partial_answer += response
        # обновляем последний элемент chat_history на "текущий частичный ответ"
        chat_history[-1] = (user_question, partial_answer)
        yield "", chat_history


with gr.Blocks() as user_block:
    gr.Markdown("## Чат-бот (RAG) по Руководству пользователя")

    chatbot = gr.Chatbot(label="DOC AI")
    msg = gr.Textbox(label="Напишите свой вопрос, касающийся Руководства пользователя")
    submit = gr.Button("➤ Отправить")

    # Привязываем функцию handle_user_question
    msg.submit(handle_user_question, inputs=[msg, chatbot], outputs=[msg, chatbot])
    submit.click(handle_user_question, inputs=[msg, chatbot], outputs=[msg, chatbot])

    # Кнопка очистки истории
    clear_btn = gr.ClearButton([msg, chatbot, chatbot], value="🗑️ Очистить историю")

if __name__ == "__main__":
    user_block.launch(share=True)
