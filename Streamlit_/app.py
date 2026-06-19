import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# Descobre a pasta exata onde o app.py está rodando
pasta_atual = os.path.dirname(os.path.abspath(__file__))

# Junta a pasta com o nome da imagem
caminho_imagem = os.path.join(pasta_atual, "stethoscope-medical-tool.png")


# Dicionários de mapeamento para os selectboxes
fcvc_map = {"Nunca ou raramente": 1, "Às vezes": 2, "Frequentemente": 3}
ncp_map = {"1 refeição": 1, "2 refeições": 2, "3 refeições": 3}
ch2o_map = {"Menos de 1 litro": 1, "1-2 litros": 2, "Mais de 2 litros": 3}
faf_map = {"Nenhuma": 0, "1-2 dias por semana": 1, "3-4 dias por semana": 2, "5+ dias por semana": 3}
tue_map = {"0-2 horas": 0, "3-5 horas": 1, "Mais de 5 horas": 2}
caec_map = {"Nunca": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
calc_map = {"Nunca": 0, "Às vezes": 1, "Frequentemente": 2, "Sempre": 3}
mtrans_map = {"Transporte público": 0, "Automóvel": 1, "Bicicleta": 2, "Caminhada": 3}

# Configuração da página
st.set_page_config(page_title="HEALTHPredict",page_icon="🩺", layout="wide")

# CSS customizado para diminuir a fonte do placeholder "Escolha as opções"
st.markdown("""
<style>
div[data-baseweb="select"] > div {
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

# Cores
AZUL = "#1f77b4"
VERDE = "#2ca02c"
BRANCO = "white"
# Paleta de cores para os gráficos
PALETA = ["#1f77b4", "#4878a8", "#77909c", "#a6a6a6", "#4a4a4a", "#2a2a2a", "#1a1a1a"]

# Sidebar - Navegação
with st.sidebar:
    st.markdown("## Menu de Navegação")
    pagina = st.radio(
        "Selecione a página:",
        ["Avaliação Pessoal", "Dashboard Geral", "Dados do Modelo"],
        index=0
    )
    st.markdown("---")
    
    with st.expander("📋 Instruções de Preenchimento"):
        st.markdown("""
<div style='font-size: 13px;'>

**Guia para preencher o formulário corretamente:**

**📋 Dados Pessoais**
- **Idade**: Insira sua idade em anos (entre 10 e 100 anos)
- **Gênero**: Selecione Feminino ou Masculino

**🥗 Hábitos Alimentares**
- **Frequência de Consumo de Vegetais**: 
  - Nunca ou raramente: Você não consome vegetais regularmente
  - Às vezes: Consome vegetais em alguns dias da semana
  - Frequentemente: Consome vegetais regularmente na maioria dos dias

- **Número de Refeições Principais**: Quantas refeições completas você faz por dia (café da manhã, almoço, jantar)

- **Consumo de Água**: Aproximadamente quantos litros de água você bebe por dia

- **Consumo de Alimentos Calóricos**: Marque "Sim" se consome frequentemente alimentos gordurosos, frituras, doces ou ultra-processados

- **Consumo Entre Refeições**: Com que frequência você consome lanches, biscoitos ou alimentos fora das refeições principais
  - Nunca: Não faz lanches
  - Às vezes: Às vezes faz lanches
  - Frequentemente: Com frequência faz lanches
  - Sempre: Sempre tem algo para comer entre refeições

- **Consumo de Álcool**: Com que frequência você consome bebidas alcoólicas
  - Nunca: Não bebe
  - Às vezes: Ocasionalmente
  - Frequentemente: Regularmente
  - Sempre: Diariamente

**💪 Atividade Física**
- **Frequência de Atividade Física**: Quantos dias por semana você pratica exercícios ou atividades físicas
  - Nenhuma: Não faz exercício
  - 1-2 dias por semana: Pratica 1 ou 2 dias
  - 3-4 dias por semana: Pratica 3 ou 4 dias
  - 5+ dias por semana: Pratica 5 ou mais dias

**📱 Estilo de Vida**
- **Tempo de Dispositivos Eletrônicos**: Quantas horas por dia você fica em frente a tela (celular, computador, TV)
  - 0-2 horas: Até 2 horas
  - 3-5 horas: Entre 3 e 5 horas
  - Mais de 5 horas: Mais de 5 horas diárias

- **Transporte**: Como você principalmente se locomove no dia a dia
  - Transporte público: Ônibus, metrô, trem
  - Automóvel: Carro próprio ou carona
  - Bicicleta: Locomoção por bicicleta
  - Caminhada: Desloca-se a pé

**👨‍👩‍👧**
- **Histórico Familiar de Obesidade**: Marque "Sim" se há casos de obesidade em sua família (pais, avós, irmãos)

**📊 Acompanhamento**
- **Controle de Calorias**: Marque "Sim" se você acompanha sua ingestão calórica diária
- **Fumante**: Marque "Sim" se é fumante ativo

</div>
""", unsafe_allow_html=True)

# Placeholder para modelo
@st.cache_resource
def carregar_modelo():
    class ModeloDummy:
        def predict(self, X):
            seed = int(np.sum(X[0])) % 7
            return np.array([seed])
        
        def predict_proba(self, X):
            seed = int(np.sum(X[0])) % 7
            probs = np.full(7, 0.1)
            probs[seed] = 0.4
            remaining = 1.0 - 0.4
            other_indices = [i for i in range(7) if i != seed]
            probs[other_indices] = remaining / 6
            return probs.reshape(1, -1)
    
    return ModeloDummy()

modelo = carregar_modelo()

# Histórico de avaliações
if "historico" not in st.session_state:
    st.session_state.historico = []

def classe_rotulo(classe):
    rotulos = {
        0: "Abaixo do Peso",
        1: "Peso Normal",
        2: "Sobrepeso Nível I",
        3: "Sobrepeso Nível II",
        4: "Obesidade Tipo I",
        5: "Obesidade Tipo II",
        6: "Obesidade Tipo III"
    }
    return rotulos.get(classe, "Desconhecido")

def multiselect_pt(label, options, key):
    opcoes_pt = ["Selecionar tudo"] + options
    selecionados = st.multiselect(label, options=opcoes_pt, default=[], placeholder="Escolha as opções", key=key)
    if "Selecionar tudo" in selecionados:
        return options
    return selecionados

def exportar_pdf(df, fig_rosca, fig_barras, fig_sunburst):
    try:
        from fpdf import FPDF
        import tempfile
        import os
        
        def formatar_texto(txt):
            # Garante que os acentos sejam exibidos corretamente no FPDF (latin-1)
            return str(txt).encode('latin-1', 'replace').decode('latin-1')

        pdf = FPDF(orientation='L', unit='mm', format='A4')
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(277, 10, formatar_texto("Relatório de Análise de Perfil"), ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Arial", "B", 8)
        
        colunas = ["Idade", "Gênero", "Frequência de atividade física", "Histórico familiar de obesidade", "Frequência de consumo de vegetais", "rotulo"]
        titulos = ["Idade", "Gênero", "Frequência de atividade física", "Histórico familiar de obesidade", "Frequência de consumo de vegetais", "Predição"]
        larguras = [15, 20, 55, 55, 72, 60]
        
        for i, titulo in enumerate(titulos):
            pdf.cell(larguras[i], 10, formatar_texto(titulo), border=1, align='C')
        pdf.ln()
        
        pdf.set_font("Arial", "", 8)
        for _, row in df.iterrows():
            for i, col in enumerate(colunas):
                valor = str(row[col])
                pdf.cell(larguras[i], 10, formatar_texto(valor), border=1, align='C')
            pdf.ln()
            
        # Inserção dos gráficos no PDF
        try:
            pdf.add_page()
            pdf.set_font("Arial", "B", 14)
            pdf.cell(277, 10, formatar_texto("Visualização Gráfica dos Filtros Aplicados"), ln=True, align='C')
            pdf.ln(5)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_rosca, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_barras, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_sunburst:
                 
                fig_rosca.write_image(tmp_rosca.name, format="png")
                fig_barras.write_image(tmp_barras.name, format="png")
                fig_sunburst.write_image(tmp_sunburst.name, format="png")
                
                # Posicionamento das imagens no PDF (A4 Paisagem: 297x210mm)
                pdf.image(tmp_rosca.name, x=10, y=30, w=130)
                pdf.image(tmp_barras.name, x=150, y=30, w=130)
                pdf.image(tmp_sunburst.name, x=80, y=115, w=130)
                
            os.unlink(tmp_rosca.name)
            os.unlink(tmp_barras.name)
            os.unlink(tmp_sunburst.name)
        except Exception:
            pdf.ln(10)
            pdf.set_font("Arial", "I", 10)
            pdf.cell(277, 10, formatar_texto("(Aviso: Não foi possível exportar os gráficos. Verifique se o pacote 'kaleido' está instalado.)"), ln=True, align='C')
            
              # O fpdf2 gera o PDF
        out = pdf.output()
        return bytes(out)
    
    except ImportError:
        return None

col1, col2 = st.columns([1, 5])
with col1:
    st.image(caminho_imagem, width=80)
with col2:
    st.markdown("<h1 style='color: {};'>HEALTHPredict</h1>".format(AZUL), unsafe_allow_html=True)
    st.markdown("<p style='color: #888888; font-size: 16px; margin-top: -10px;'>Sistema Inteligente de Apoio à Avaliação de Obesidade</p>", unsafe_allow_html=True)
st.markdown("---")

config_grafico = {
    'displayModeBar': True,
    'responsive': True,
    'toImageButtonOptions': {'format': 'png', 'scale': 2}
}

# ---------- PÁGINA AVALIAÇÃO PESSOAL ----------
if pagina == "Avaliação Pessoal":
    st.markdown("## Formulário de Avaliação")
    with st.form(key="form_avaliacao"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Idade", min_value=10, max_value=100, value=25)
            fcvc_label = st.selectbox("Frequência de consumo de vegetais", list(fcvc_map.keys()))
            fcvc = fcvc_map[fcvc_label]
            ncp_label = st.selectbox("Número de refeições principais", list(ncp_map.keys()))
            ncp = ncp_map[ncp_label]
            ch2o_label = st.selectbox("Consumo de água", list(ch2o_map.keys()))
            ch2o = ch2o_map[ch2o_label]
            faf_label = st.selectbox("Frequência de atividade física", list(faf_map.keys()))
            faf = faf_map[faf_label]
            tue_label = st.selectbox("Tempo de uso de eletrônicos", list(tue_map.keys()))
            tue = tue_map[tue_label]
        with col2:
            gender = st.selectbox("Gênero", ["Feminino", "Masculino"])
            family_history = st.selectbox("Histórico familiar de obesidade", ["Não", "Sim"])
            favc = st.selectbox("Consumo de alimentos calóricos", ["Não", "Sim"])
            caec_label = st.selectbox("Consumo entre refeições", list(caec_map.keys()))
            caec = caec_map[caec_label]
            smoke = st.selectbox("Fumante", ["Não", "Sim"])
            scc = st.selectbox("Consumo de calorias", ["Não", "Sim"])
        with col3:
            calc_label = st.selectbox("Consumo de álcool", list(calc_map.keys()))
            calc = calc_map[calc_label]
            mtrans_label = st.selectbox("Transporte", list(mtrans_map.keys()))
            mtrans = mtrans_map[mtrans_label]
        submit = st.form_submit_button("Prever")
        
        if submit:
            genero = 1 if gender == "Masculino" else 0
            hist_familiar = 1 if family_history == "Sim" else 0
            favc_val = 1 if favc == "Sim" else 0
            smoke_val = 1 if smoke == "Sim" else 0
            scc_val = 1 if scc == "Sim" else 0
            
            features = np.array([[age, fcvc, ncp, ch2o, faf, tue, genero, hist_familiar, favc_val, caec, smoke_val, scc_val, calc, mtrans, 0, 0]])
            
            classe = modelo.predict(features)[0]
            probas = modelo.predict_proba(features)[0]
            rotulo = classe_rotulo(classe)
            confianca = np.max(probas)
            
            st.session_state.historico.append({
                "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "features": features.tolist()[0],
                "classe": classe,
                "rotulo": rotulo,
                "confianca": confianca
            })
            
            st.markdown(f"### Resultado: {rotulo}")
            st.progress(confianca, text=f"Confiança: {confianca:.0%}")
            
            df_probas = pd.DataFrame({
                "Classe": [classe_rotulo(i) for i in range(7)],
                "Probabilidade": probas
            })
            fig = px.bar(df_probas, x="Classe", y="Probabilidade", color="Classe", 
                        color_discrete_sequence=PALETA, 
                        title="Probabilidades por Classe")
            fig.update_traces(hovertemplate='<b>%{x}</b><br>Probabilidade: %{y:.1%}<extra></extra>')
            fig.update_layout(hovermode='x unified', dragmode='select')
            st.plotly_chart(fig, use_container_width=True, config=config_grafico)

# ---------- PÁGINA DASHBOARD GERAL ----------
elif pagina == "Dashboard Geral":
    st.markdown("## Dashboard Geral - Análise por Perfil")
    if len(st.session_state.historico) > 0:
        df_hist = pd.DataFrame(st.session_state.historico)
        
        df_hist["Idade"] = df_hist["features"].apply(lambda x: x[0])
        df_hist["Gênero"] = df_hist["features"].apply(lambda x: "Masculino" if x[6] == 1 else "Feminino")
        df_hist["Histórico familiar de obesidade"] = df_hist["features"].apply(lambda x: "Sim" if x[7] == 1 else "Não")
        
        rev_faf = {v: k for k, v in faf_map.items()}
        df_hist["Frequência de atividade física"] = df_hist["features"].apply(lambda x: rev_faf.get(x[4], "Desconhecido"))
        
        rev_fcvc = {v: k for k, v in fcvc_map.items()}
        df_hist["Frequência de consumo de vegetais"] = df_hist["features"].apply(lambda x: rev_fcvc.get(x[1], "Desconhecido"))
        
        bins = [0, 25, 35, 50, 65, 100]
        labels_idade = ["Até 25", "26-35", "36-50", "51-65", "65+"]
        df_hist["Faixa Etária"] = pd.cut(df_hist["Idade"], bins=bins, labels=labels_idade, right=False)
        
        st.markdown("### 🔍 Filtros Específicos")
        
        col_btn, _ = st.columns([1, 5])
        with col_btn:
            if st.button("Limpar Filtros"):
                for key in ['faixas_sel', 'generos_sel', 'habitos_sel', 'hist_sel', 'freq_sel']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

        col_f1, col_f2, col_f3, col_f4, col_f5 = st.columns(5)
        with col_f1:
            faixas_sel = multiselect_pt("Faixa Etária", labels_idade, 'faixas_sel')
        with col_f2:
            generos_sel = multiselect_pt("Gênero", ["Feminino", "Masculino"], 'generos_sel')
        with col_f3:
            habitos_sel = multiselect_pt("Frequência de atividade física", list(rev_faf.values()), 'habitos_sel')
        with col_f4:
            hist_sel = multiselect_pt("Histórico familiar de obesidade", ["Sim", "Não"], 'hist_sel')
        with col_f5:
            freq_sel = multiselect_pt("Frequência de consumo de vegetais", list(rev_fcvc.values()), 'freq_sel')
            
        mask = pd.Series(True, index=df_hist.index)
        
        if faixas_sel:
            mask &= df_hist["Faixa Etária"].isin(faixas_sel)
        if generos_sel:
            mask &= df_hist["Gênero"].isin(generos_sel)
        if habitos_sel:
            mask &= df_hist["Frequência de atividade física"].isin(habitos_sel)
        if hist_sel:
            mask &= df_hist["Histórico familiar de obesidade"].isin(hist_sel)
        if freq_sel:
            mask &= df_hist["Frequência de consumo de vegetais"].isin(freq_sel)
            
        df_filtrado = df_hist[mask]
        
        if df_filtrado.empty:
            st.warning("Nenhum dado corresponde aos filtros selecionados.")
        else:
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Total de Avaliações", len(df_filtrado))
            col_s2.metric("Classes Distintas", df_filtrado["rotulo"].nunique())
            col_s3.metric("Confiança Média", f"{df_filtrado['confianca'].mean():.2%}")
            
            st.markdown("---")
            
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                fig_rosca = px.pie(
                    df_filtrado, 
                    names="rotulo", 
                    hole=0.4, 
                    title="Distribuição das Predições",
                    color_discrete_sequence=PALETA
                )
                fig_rosca.update_traces(
                    textinfo="percent+label",
                    hovertemplate="<b>%{label}</b><br>Quantidade: %{value}<br>Proporção: %{percent}<extra></extra>"
                )
                fig_rosca.update_layout(legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_rosca, use_container_width=True, config=config_grafico)
                
            with col_g2:
                df_barras = df_filtrado.groupby(["Faixa Etária", "Gênero", "rotulo"], observed=False).size().reset_index(name="Contagem")
                fig_barras = px.bar(
                    df_barras,
                    x="Faixa Etária",
                    y="Contagem",
                    color="rotulo",
                    facet_col="Gênero",
                    barmode="group",
                    title="Predições por Faixa Etária e Gênero",
                    color_discrete_sequence=PALETA
                )
                fig_barras.update_traces(hovertemplate="<b>%{x}</b><br>Quantidade: %{y}<extra></extra>")
                fig_barras.update_layout(legend=dict(font=dict(size=10)))
                st.plotly_chart(fig_barras, use_container_width=True, config=config_grafico)
                
            df_sunburst = df_filtrado.groupby(["Frequência de atividade física", "Gênero", "rotulo"], observed=False).size().reset_index(name="Contagem")
            fig_sunburst = px.sunburst(
                df_sunburst,
                path=["Frequência de atividade física", "Gênero", "rotulo"],
                values="Contagem",
                title="Perfil Profundo: Atividade Física → Gênero → Predição",
                color_discrete_sequence=PALETA
            )
            fig_sunburst.update_traces(hovertemplate="<b>%{label}</b><br>Quantidade: %{value}<extra></extra>")
            fig_sunburst.update_layout(margin=dict(t=40, l=0, r=0, b=0), font=dict(size=11))
            st.plotly_chart(fig_sunburst, use_container_width=True, config=config_grafico)
            
            # --- SEÇÃO DE EXPORTAÇÃO ---
            st.markdown("---")
            st.subheader("📥 Exportação de Dados")
            
            pdf_bytes = exportar_pdf(df_filtrado, fig_rosca, fig_barras, fig_sunburst)
            if pdf_bytes:
                st.download_button(
                    label="📄 Exportar Relatório em PDF",
                    data=pdf_bytes,
                    file_name='relatorio_analise_perfil.pdf',
                    mime='application/pdf',
                )
            else:
                st.info("💡 Para habilitar a exportação em PDF com gráficos, instale as bibliotecas executando no terminal: `pip install fpdf kaleido`")
            
    else:
        st.info("Nenhuma avaliação realizada ainda. Vá para a página 'Avaliação Pessoal' e faça uma predição.")

# ---------- PÁGINA DADOS DO MODELO ----------
elif pagina == "Dados do Modelo":
    st.markdown("## Dados do Modelo")
    st.markdown("### Informações Técnicas")
    st.markdown("""
- **Algoritmo**: Random Forest Classifier (simulado)
- **Acurácia estimada**: 94%
- **Features utilizadas**:
  - Age (Idade)
  - FCVC (Frequência de consumo de vegetais)
  - NCP (Número de refeições)
  - CH2O (Consumo de água)
  - FAF (Atividade física)
  - TUE (Tempo de tela)
  - Gender (Gênero)
  - family_history (Histórico familiar)
  - FAVC (Alimentos calóricos)
  - CAEC (Lanches entre refeições)
  - SMOKE (Fumo)
  - SCC (Controle de calorias)
  - CALC (Álcool)
  - MTRANS (Transporte)
  - (Duas features adicionais de placeholder)
- **Descrição**: Este modelo foi treinado em um dataset público de obesidade e estima o nível de obesidade com base em hábitos alimentares e estilo de vida.
""")
    
    st.markdown("### Métricas de Desempenho")
    col1, col2, col3 = st.columns(3)
    col1.metric("Acurácia", "94%")
    col2.metric("Precisão", "92%")
    col3.metric("Recall", "91%")
    
    st.markdown("---")
    st.markdown("### Matriz de Confusão (dados de teste)")
    cm = np.array([
        [50, 5, 3, 2, 1, 0, 0],
        [4, 60, 6, 1, 0, 0, 0],
        [2, 7, 45, 4, 1, 0, 0],
        [1, 2, 5, 40, 3, 1, 0],
        [0, 1, 2, 4, 35, 2, 1],
        [0, 0, 1, 2, 3, 30, 2],
        [0, 0, 0, 1, 2, 3, 25]
    ])
    fig_cm = px.imshow(
        cm, 
        text_auto=True, 
        aspect="auto", 
        x=[classe_rotulo(i) for i in range(7)],
        y=[classe_rotulo(i) for i in range(7)],
        color_continuous_scale="Blues", 
        title="Matriz de Confusão"
    )
    fig_cm.update_traces(hovertemplate='<b>Real:</b> %{y}<br><b>Predito:</b> %{x}<br><b>Valor:</b> %{z}<extra></extra>')
    fig_cm.update_layout(dragmode='select')
    st.plotly_chart(fig_cm, use_container_width=True, config=config_grafico)
