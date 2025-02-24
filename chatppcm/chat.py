import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from chatppcm import rag

# Função para carregar ou criar o vector store usando FAISS
def load_vectorstore():
    persist_directory = 'ppcm/base'
    embedding = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectordb = FAISS.load_local(
            persist_directory,
            embedding,
            allow_dangerous_deserialization=True
        )
    else:
        st.write("Criando novo vector store...")
        chunks = rag.PrepararBaseDados()
        vectordb = FAISS.from_documents(chunks, embedding)
        vectordb.save_local(persist_directory)
    return vectordb

# Função para criar a cadeia de QA com histórico e prompt aprimorado, utilizando retriever base (sem compressão)
def create_qa_chain(vectordb, memory):
    # Inicializa o LLM
    llm = ChatOpenAI(model="gpt-4", openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # Cria o retriever base usando mmr com k=5 e lambda_mult=0.5
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )
    
    # Prompt template aprimorado, incluindo o histórico da conversa
    prompt_template = (
        "Contexto relevante:\n{context}\n\n"
        "Instruções para resposta:\n"
        "1. Responda APENAS com base no contexto fornecido.\n"
        "2. Se a informação não estiver no contexto, diga 'Não tenho essa informação no meu conhecimento atual'.\n"
        "3. Formate a resposta de maneira organizada, utilizando listas se necessário.\n"
        "4. Limite a resposta a 3 frases ou menos.\n"
        "5. Finalize com 'Obrigado por perguntar!'\n\n"
        "Pergunta: {question}\n"
        "Fala meu prezado(a), Resposta concisa e contextualizada:"
    )
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)
    
    # Cria a cadeia de QA utilizando o prompt customizado e o retriever base
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        verbose=True
    )
    return qa_chain

# --- Interface Streamlit ---
st.title("Oráculo PPCM")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Inicializa a memória (o chain pode atualizá-la internamente, mas usaremos as mensagens do session_state para o histórico)
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

try:
    vectordb = load_vectorstore()
    qa_chain = create_qa_chain(vectordb, memory)
    
    # Exibe o histórico de mensagens na interface
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Recebe a nova pergunta do usuário
    if user_input := st.chat_input("Faça sua pergunta:"):
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Constrói o histórico de conversa a partir das mensagens armazenadas
        chat_history_str = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state["messages"]])
        
        # Executa a cadeia RAG passando a pergunta e o histórico explicitamente
        result = qa_chain({"question": user_input, "chat_history": chat_history_str})
        answer = result["answer"]
        
        st.session_state["messages"].append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

except Exception as e:
    st.error(f"Erro crítico: {str(e)}")
    st.stop()
