import asyncio
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
import time
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from pathlib import Path

# Obtener la ruta del directorio del script actual
current_dir = Path(__file__).resolve().parent

# Define la función asíncrona para llamar a ChatGoogleGenerativeAI

st.set_page_config(page_title="AVU")
st.header("AVU💬🤖")

# Configuración de la interfaz de Streamlit
st.title("Asistente virtual Universitario")
st.write("Hola estoy aquí para ayudarte en tu vida cotidiana, tienes alguna pregunta❓")


def stream_response(response):
    placeholder = st.empty()
    for i in range(len(response) + 1):
        placeholder.markdown(response[:i])
        print(placeholder)
        time.sleep(0.05) 

async def get_response(input_text, retriever):

#async def get_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    template = """
                Eres un asistente especializado en Ingeniería en Telecomunicaciones y Electrónica especificamente en el aréa de sistemas de comunicaciones. Actúa como un profesor universitario y responde acorde para ayudar a los alumnos. Tambien puedes responder a otras preguntas ya que eres un asistente.\nQuestion: {question} \nContext: {context} \nAnswer:
                """
    prompt_template = PromptTemplate(
        input_variables=["question", "context"],
        template=template
    )

    #prompt = hub.pull("rlm/rag-prompt")
    prompt = prompt_template

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    #return rag_chain



    #llm = ChatGoogleGenerativeAI(model="gemini-pro")
    #response = llm.invoke(input_text)
    response = rag_chain.invoke(input_text)
    #print(response)
    print(type(response))
    print(response)
    return response

def generate_retriever():
   embeddings = OllamaEmbeddings(model="nomic-embed-text")
   vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
   retriever = vectorstore.as_retriever()
   st.session_state.retriever = retriever
   st.session_state.retriever_ready = True
#
##async def get_rag_chain(retriever):
#    llm = ChatGoogleGenerativeAI(model="gemini-pro")
#    template = """
#                Eres un asistente especializado en Ingeniería en Telecomunicaciones y Electrónica especificamente en el aréa de sistemas de comunicaciones. Actúa como un profesor universitario y responde acorde para ayudar a los alumnos\nQuestion: {question} \nContext: {context} \nAnswer:
#                """
#    prompt_template = PromptTemplate(
#        input_variables=["question", "context"],
#        template=template
#    )
#
#    #prompt = hub.pull("rlm/rag-prompt")
#    prompt = prompt_template
#
#    def format_docs(docs):
#        return "\n\n".join(doc.page_content for doc in docs)
#
#
#    rag_chain = (
#        {"context": retriever | format_docs, "question": RunnablePassthrough()}
#        | prompt
#        | llm
#        | StrOutputParser()
#    )
#    #return rag_chain

def main():
    # Recibir entrada del usuario
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'retriever_ready' not in st.session_state:
        st.session_state.retriever_ready = False

    generate_retriever()
    #if 'retriever' not in st.session_state:
    #    st.session_state.retriever = None

    # Iniciar el hilo para generar el retriever
    #if not st.session_state.retriever_ready:
    #    generate_retriever()

    # Mostrar el estado del retriever
    #if st.session_state.retriever_ready:
    #    #   st.success("Retriever está listo")
    #    print("Retriever está listo")
    #else:
    #    st.info("Generando retriever...")

    # Mostrar la conversación
    for i, (speaker, text) in enumerate(st.session_state.conversation):
        if speaker == "Usuario":
            st.write(f"{speaker}: {text}")
        else:
            stream_response(text)
        # Añadir un espacio para el siguiente input y botón después de cada respuesta del chatbot
        if speaker == "Chatbot" and i == len(st.session_state.conversation) - 1:
            if st.session_state.retriever_ready:
                # Entrada de texto para la pregunta del usuario
                question = st.text_input("Escribe tu pregunta:", key=f"input_{i}")
                # Botón para enviar la pregunta
                if st.button("Enviar", key=f"button_{i}"):
                    with st.spinner("Pensando💭..."):
                        if question:
                            # Añadir la pregunta del usuario a la conversación
                            st.session_state.conversation.append(("Usuario", question))
                            # Obtener la respuesta y añadirla a la conversación
                            try:
                                #rag_chain = asyncio.run(get_rag_chain(st.session_state.retriever))
                                response = asyncio.run(get_response(question, st.session_state.retriever))
                                st.session_state.conversation.append(("Chatbot", response))
                                # Refrescar la página para mostrar la nueva conversación
                                st.experimental_rerun()
                            except Exception as e:
                                st.write("Parece que ocurrió un error")
                                st.write("Refresque la página o contacte a los desarrolladores")
                                st.warning(e)
                            

    # Si la conversación está vacía, mostrar el primer input y botón
    if st.session_state.retriever_ready:
        if len(st.session_state.conversation) == 0:
            question = st.text_input("Escribe tu pregunta:")
            if st.button("Enviar"):
                with st.spinner("Pensando💭..."):
                    if question:
                        st.session_state.conversation.append(("Usuario", question))
                        try:
                            #rag_chain = asyncio.run(get_rag_chain(st.session_state.retriever))
                            response = asyncio.run(get_response(question, st.session_state.retriever))
                            st.session_state.conversation.append(("Chatbot", response))
                            st.experimental_rerun()
                        except Exception as e:
                            st.write("Parece que ocurrió un error")
                            st.write("Refresque la página o contacte a los desarrolladores")
                            st.warning(e)
                    

if __name__ == "__main__":
    main()