import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from chatppcm import rag

def load_vectorstore():
    persist_directory = 'ppcm/base'
    embedding = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    if os.path.exists(persist_directory):
        vectordb = FAISS.load_local(
            persist_directory,
            embedding,
            allow_dangerous_deserialization=True
        )
    else:
        chunks = rag.PrepararBaseDados()
        vectordb = FAISS.from_documents(chunks, embedding)
        vectordb.save_local(persist_directory)
    return vectordb

def create_qa_chain(vectordb, memory):
    llm = ChatOpenAI(model="gpt-4", temperature=0.3, openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.5}
    )
    
    template = """Use os seguintes trechos de contexto para responder à pergunta no final. 
Se não souber a resposta, diga que não sabe. Use no máximo 3 frases. 
Formate a resposta de forma concisa. Finalize com "obrigado por perguntar meu preazado(a)!".
{context}
Pergunta: {question}
Resposta útil:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
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
