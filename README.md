# 🎧 Sistema de Sugestão de Músicas com Machine Learning

Este projeto utiliza técnicas de **Machine Learning** para recomendar músicas semelhantes com base em suas características numéricas (popularidade, energia, valência, etc.).  
Basta digitar o nome de uma música e o sistema retorna **10 músicas parecidas**.

---

## 🚀 Como funciona a recomendação?

O processo de sugestão é dividido em **três etapas principais**:

1. **Pré-processamento**  
   As informações numéricas das músicas são padronizadas para que todas as variáveis tenham peso equivalente.

2. **Redução de Dimensionalidade (PCA)**  
   As músicas são projetadas em um espaço reduzido (2 dimensões), permitindo visualizar a similaridade entre elas.

3. **Agrupamento K-Means**  
   As músicas são separadas em clusters. A recomendação é feita buscando as músicas mais próximas dentro do mesmo grupo da música consultada.

A distância entre as músicas é calculada no espaço reduzido, retornando as **10 mais parecidas**.

---

## 🖥️ Interface do Usuário

O sistema possui uma interface simples e intuitiva, onde o usuário digita o nome da música e recebe as recomendações instantaneamente.

📌 Exemplo de tela:

<img width="935" height="766" alt="image" src="https://github.com/user-attachments/assets/2c8a9ee5-e6d6-4227-8884-5d69fef9af50" />

<img width="936" height="747" alt="image" src="https://github.com/user-attachments/assets/376adcfb-7599-4aae-90c7-3ca0a342130c" />



---

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas / NumPy
- Scikit-Learn
- PCA (Principal Component Analysis)
- K-Means Clustering
- Streamlit (Interface Web)

---

## 📂 Estrutura do Projeto


