import streamlit as st

# Configurações da página
st.set_page_config(
    page_title="PPCM AI",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para exibir o logo
def show_logo(url):
    st.image(url, width=200)

LOGO_URL_LARGE = "images/samarco.png"  # Substitua pelo caminho correto da sua logo

# Inicializa o estado de sessão
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Função para sair da aplicação
def logout():
    st.session_state["logged_in"] = False
    st.success("Você saiu da aplicação com sucesso!")
    st.rerun()

if st.session_state["logged_in"]:
    # Sidebar com logo e botão de logout
    with st.sidebar:
        show_logo(LOGO_URL_LARGE)  # Logo na sidebar
        st.write("")  # Espaçamento
        if st.button("Sair"):
            logout()

    # Navegação e páginas
    oraculo = st.Page("chatppcm/chat.py", title="Oráculo", icon="🤖")
    mes = st.Page("agentemes/mes.py", title="MES", icon=":material/swap_horiz:")
    extrator = st.Page("extrator/main.py", title="Extrator", icon=":material/search:")
    dash = st.Page("dash/dash.py", title="Dash Notas", icon=":material/list:")

    pg = st.navigation({
        "Oráculo": [oraculo],
        "Agentes": [mes, extrator],
        "Dashs LLM's": [dash]
    })
    
    pg.run()

else:
    # Esconde a sidebar e exibe logo na tela de login
    st.markdown("""
        <style>
            section[data-testid="stSidebar"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Exibe logo na tela principal de login
    show_logo(LOGO_URL_LARGE)
    
    # Tela de login
    st.title("Login")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    
    if st.button("Login"):
        if username == "admin" and password == "admin":
            st.session_state["logged_in"] = True
            st.success("Login realizado com sucesso!")
            st.rerun()
        else:
            st.error("Usuário ou senha incorretos.")