import streamlit as st
from chatppcm import rag

from langchain.embeddings.openai import OpenAIEmbeddings
persist_directory = 'ppcm/test/'
embedding = OpenAIEmbeddings()

chunks=rag.PrepararBaseDados()
st.write(len(chunks))
