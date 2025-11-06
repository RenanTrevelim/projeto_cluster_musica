import streamlit as st
import pandas as pd
import joblib
import numpy as np
from scipy.spatial.distance import euclidean

# Configuração básica da página
st.set_page_config(page_title="Sugestão de Música", page_icon="🎧", layout="centered")

# Título e instruções
st.title("Sugestão de Música 🎧")
st.write("Digite o nome de uma música e veja 10 sugestões parecidas.")

# Explicação do algoritmo (apenas visual)
with st.expander("Como funciona a recomendação?"):
    st.write("""
    O sistema utiliza técnicas de **Machine Learning** para sugerir músicas similares. O processo é dividido em três etapas principais:

    1. **Pré-processamento**: As informações numéricas das músicas (como popularidade, valência, energia, etc.) são padronizadas.
    2. **Redução de Dimensionalidade (PCA)**: As músicas são colocadas em um espaço com poucas dimensões (2, neste caso), permitindo visualizar similaridades.
    3. **Agrupamento K-Means**: As músicas são separadas em grupos (clusters). A recomendação é feita buscando as músicas mais próximas da música escolhida dentro do mesmo grupo.

    A distância entre as músicas é calculada no espaço reduzido, encontrando as 10 mais parecidas.
    """)

# Carrega modelos/artefatos
scaler = joblib.load('scaler(2).pkl')
pca = joblib.load('pca(1).pkl')
kmeans = joblib.load('modelo_kmeans(2).pkl')

# Entrada do usuário
musica = st.text_input("Digite o nome de uma música:")
enviar = st.button("Enviar")

# Função original de recomendação (lógica inalterada)
def recomendacao(musica, df):
    nome_musica = musica

    cluster = df[df['artists_song'] == nome_musica]['cluster'].values[0]

    musicas_recomendadas = df[df['cluster'] == cluster]

    componentes_musica = musicas_recomendadas[
        musicas_recomendadas['artists_song'] == nome_musica
    ][['pca1', 'pca2']].values[0]

    musicas_recomendadas['Dist'] = musicas_recomendadas.apply(
        lambda row: euclidean(componentes_musica, [row['pca1'], row['pca2']]),
        axis=1
    )

    # remove a própria música
    musicas_recomendadas = musicas_recomendadas[
        musicas_recomendadas['artists_song'] != nome_musica
    ]

    recomendadas = musicas_recomendadas.sort_values('Dist').head(10)[['artists_song']]
    return recomendadas

# Quando clicar no botão
if enviar:
    if musica.strip() == "":
        st.warning("Por favor, digite o nome de uma música.")
    else:
        st.success("Música enviada!")
        st.write("Músicas Recomendadas:")

        # Prepara dados (mesma lógica)
        dados = pd.read_csv('dados_musicas.csv', sep=';')
        df = dados.copy()
        dados = dados.drop(['artists', 'id', 'name', 'artists_song'], axis=1)
        dados_escalados = scaler.transform(dados)
        dados_pca = pca.transform(dados_escalados)
        cluster = kmeans.predict(dados_pca)

        df[['pca1', 'pca2']] = dados_pca
        df['cluster'] = cluster

        # Recomendações
        try:
            musicas_recomendadas = recomendacao(musica, df)
            st.table(musicas_recomendadas.reset_index(drop=True))
        except IndexError:
            st.error("Não encontrei essa música no dataset. Verifique o nome exatamente como está cadastrado.")
