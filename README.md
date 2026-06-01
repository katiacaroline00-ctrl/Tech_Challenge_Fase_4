🧬 Predição de Níveis de Obesidade

Tech Challenge – Fase 4 | Pós-Graduação em Data Analytics

Este repositório contém o pipeline completo de Machine Learning desenvolvido para prever níveis de obesidade a partir de 17 variáveis preditoras. O projeto inclui desde a exploração e tratamento dos dados até o deploy de um aplicativo interativo com Streamlit, além de um dashboard analítico e uma apresentação gerencial.



📌 Descrição

O objetivo central deste desafio é construir um modelo preditivo (classificação) capaz de classificar corretamente o nível de obesidade de um indivíduo com base em características demográficas, alimentares e de estilo de vida. A meta estabelecida é atingir uma acurácia superior a 75% no conjunto de teste.

Para garantir a reprodutibilidade e a aplicação prática do modelo, foram desenvolvidos:





Pipeline completo de pré-processamento e treinamento (VScode)



Modelo serializado (modelo.pkl) e pipeline salvo (pipeline.pkl)



Aplicativo web preditivo em Streamlit



Dashboard analítico interativo



Apresentação gerencial (PDF)



🎯 Objetivos





Explorar e compreender o dataset, identificando padrões e possíveis problemas de qualidade.



Realizar feature engineering e tratamento dos dados (valores ausentes, outliers, codificação de variáveis categóricas, normalização).



Treinar e comparar diferentes algoritmos de classificação (Regressão Logística, Random Forest, XGBoost, etc.).



Selecionar o melhor modelo com base em métricas de acurácia, precisão, recall e F1-score.



Salvar o pipeline e o modelo final para uso em produção.



Desenvolver uma interface web amigável para que usuários possam inserir seus dados e obter a predição.



Criar um dashboard analítico que exiba insights do dataset e o desempenho do modelo.



Apresentar os resultados de forma clara e gerencial.



📊 Dicionário de Dados

O conjunto obesity.csv contém 16 variáveis preditoras e 1 variável alvo (Obesity_level). Abaixo a descrição detalhada de cada campo:

| Age | Numérica | Idade em anos | | Weight | Numérica | Peso em quilogramas | | FAVC | Binária | Consumo frequente de alimentos calóricos (yes / no) | | NCP | Numérica | Número de refeições principais por dia | | SMOKE | Binária | Fumante (yes / no) | | SCC | Binária | Monitoramento de calorias consumidas (yes / no) | | TUE | Numérica | Tempo de uso de dispositivos eletrônicos (horas/dia) | | MTRANS | Categórica | Meio de transporte mais utilizado (Automobile, Bike, Motorbike, Public_Transportation, Walking) |



ℹ️ A variável alvo Obesity_level considerou todas as classes presentes, excetp altura e peso.



🛠️ Tecnologias Utilizadas

| Ferramenta | Finalidade | Matplotlib / Seaborn | Visualização estática | | Scikit-learn | Pipeline de ML, pré-processamento e modelos | | Imbalanced-learn | Tratamento de desbalanceamento (SMOTE) | | Streamlit | Deploy do aplicativo preditivo |



📁 Estrutura do Projeto

├── Dataset/
│   ├── dicionario_obesity_fiap_tc4.pdf  # Dicionário Dataset
│   └── obesity.csv                      # Dataset
├── Modelo_e_pipeline/
│   ├── modelo.pkl                       # Modelo salvo
│   ├── pipeline.pkl                     # Pipeline salvo
│   └── tech_challenge_fase_4.ipynb      # Código do pipeline (Jupyter Notebook)
├── Streamlit_/
│   ├── config.toml                      # Configurações do Streamlit
│   ├── app.py                           # Código do aplicativo Streamlit
│   ├── estetoscopio.ico                 # Ícone do Streamlit
│   ├── stethoscope-medical-tool.png     # Imagem do Streamlit
│   └── streamlit                        # Arquivo de configuração/execução
└── README.md                            # Este arquivo
└── requirements.txt                     # bibliotecas utilizadas




🚀 Como Executar

1. Clonar o repositório

git clone https://seu-repositorio.git
cd tech-challenge-fase4


2. Criar e ativar ambiente virtual (opcional, mas recomendado)

python -m venv venv
source venv/bin/activate       # Linux / macOS
# ou
venv\Scripts\activate          # Windows


3. Instalar dependências

pip install -r requirements.txt


4. Executar o pipeline de treinamento

vscode notebook Modelo_e_pipeline/tech_challenge_fase_4.ipynb


O notebook gera os arquivos modelo.pkl e pipeline.pkl dentro da pasta Modelo_e_pipeline/.

5. Rodar o aplicativo Streamlit

cd Streamlit_
streamlit run app.py


O aplicativo abrirá no navegador em http://localhost:8501. Insira os dados nos campos e clique em Prever para obter a classificação.


🎥 Vídeo de Apresentação | https://youtu.be/ahwumbIUD1k |


👥 Autora 
Katia Caroline Wilkomm


Projeto desenvolvido como parte do Tech Challenge – Fase 4 da Pós-Graduação em Data Analytics da FIAP.


📝 Licença

Este projeto é de uso acadêmico e não possui fins comerciais. 
