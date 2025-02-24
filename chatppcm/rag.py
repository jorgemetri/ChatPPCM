from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re
import os
import streamlit as st

def SplitManual(docs):
    """
    Args:
        docs: Um array de Document referente a cada página do manual de manutenção.
        
    Returns:
        Um array de Document contendo os chunks extraídos do manual.
    """
    generic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=20,
        chunk_overlap=10,
        separators=["\n"]
    )
    
    dados = []
    
    for doc in docs:
        chunks = generic_splitter.split_text(doc.page_content)  # Divide o texto
        for chunk in chunks:
            dados.append(Document(page_content=chunk.strip(), metadata=doc.metadata.copy()))  # Preserva os metadados
    
    return dados

def preprocess_text(text):
    """
    Aplica regex para identificar e marcar cabeçalhos no texto.
    """
    text = re.sub(
        r'\n(\d{1,2})(\s+[^\d\.])',
        r'\n||MAIN_HEADER||\1\2', 
        text
    )
    return text

def ChunkearConhecimento(documentos):
    """
    Recebe uma lista de Document (resultado de loader.load())
    e retorna uma lista de chunks com metadados preservados.
    """
    final_docs = []
    
    processed_pages = []
    
    # Processa cada página individualmente
    for page in documentos:
        processed_content = preprocess_text(page.page_content)
        processed_pages.append(Document(
            page_content=processed_content,
            metadata=page.metadata.copy()  # Preserva os metadados originais
        ))
    
    # Configurar o splitter com os separadores conforme o código original
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=0,
        separators=["\n||MAIN_HEADER||", "\n \n||MAIN_HEADER||"]
    )
    
    # Dividir os documentos em chunks
    split_chunks = text_splitter.split_documents(processed_pages)
    
    # Pós-processamento: remover os marcadores inseridos
    for chunk in split_chunks:
        chunk.page_content = re.sub(r'\|\|MAIN_HEADER\|\|', '', chunk.page_content).strip()
    
    final_docs.extend(split_chunks)
    
    return final_docs

def PrepararBaseDados():
    pasta = "chatppcm/dados"
    loaders = []
    dados_processados = []  # Inicializa a lista para armazenar os chunks processados
    
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".pdf"):
            caminho_completo = os.path.join(pasta, arquivo)
            if "Manual GMU rev4" in caminho_completo:
                # Esse documento tem um tratamento de dados diferente
                st.write(f"Processando: {caminho_completo}")
                manual = PyPDFLoader(caminho_completo)
                doc = manual.load()
                dados_processados.extend(SplitManual(doc))  # Garante que os chunks são adicionados corretamente
            else:
                loaders.append(PyPDFLoader(caminho_completo))
    
    docs = []            
    for loader in loaders:
        docs.extend(loader.load())  # Carrega todos os documentos
    
    # Transformando cada página dentro de docs em chunks
    for doc in docs:
        chunks = ChunkearConhecimento([doc])  # Passa como lista para manter a compatibilidade
        dados_processados.extend(chunks)  # Adiciona os chunks à lista final
    
    return dados_processados  # Retorna os dados processados para uso posterior

def CriandoDataBase():
    persist_directory = 'db/ppcm'