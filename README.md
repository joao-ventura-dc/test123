# 🏭 POC: Análise e Previsão de Posturas de Trabalho com IA

## 📋 Descrição do Projeto

Este projeto implementa um **Proof of Concept (POC)** completo para análise e previsão de posturas de trabalho utilizando técnicas de Inteligência Artificial e Machine Learning. O sistema processa dados biomecânicos de múltiplas câmeras, identifica padrões, detecta anomalias e cria modelos preditivos para avaliar riscos ergonómicos.

## 🎯 Objetivos do POC

### Técnicos
- ✅ Consolidar dados de múltiplos ficheiros XLSX num dataset unificado
- ✅ Realizar limpeza, normalização e engenharia de features
- ✅ Explorar padrões e agrupamentos nos dados
- ✅ Testar modelos de deteção de anomalias
- ✅ Criar modelo preditivo temporal para scores ergonómicos

### Funcionais
- ✅ Demonstrar insights automáticos sobre posturas de risco
- ✅ Validar previsões de risco ergonómico
- ✅ Criar fundação técnica para evolução futura

## 🚀 Quick Start com Docker

### Pré-requisitos
- Docker
- Docker Compose

### Executar o Pipeline Completo

```bash
# Construir e executar
docker-compose up --build

# Ou em background
docker-compose up -d --build
```

O pipeline irá:
1. Consolidar todos os ficheiros XLSX da pasta `biomechanic scores/`
2. Realizar análise exploratória completa (EDA)
3. Detectar anomalias usando múltiplos algoritmos
4. Treinar modelos preditivos
5. Gerar relatórios e visualizações

### Executar com Jupyter Notebook (Opcional)

```bash
# Iniciar Jupyter Notebook
docker-compose --profile jupyter up jupyter

# Aceder via browser
# http://localhost:8888
```

## 📁 Estrutura do Projeto

```
.
├── biomechanic scores/      # Dados originais (XLSX)
├── src/                     # Código fonte
│   ├── data_consolidation.py
│   ├── exploratory_analysis.py
│   ├── anomaly_detection.py
│   ├── predictive_model.py
│   └── main_pipeline.py
├── data/
│   ├── raw/                 # Dados brutos
│   └── processed/           # Dados processados
├── models/                  # Modelos ML treinados
├── reports/                 # Relatórios e visualizações
│   ├── eda/
│   ├── anomalies/
│   └── predictions/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 📊 Outputs Gerados

### 1. Datasets Consolidados
- `data/processed/consolidated_data.csv` - Dataset unificado
- `data/processed/data_with_features.csv` - Dataset com features engineered
- `data/processed/data_with_anomalies.csv` - Dataset com anomalias detetadas
- `data/processed/detected_anomalies.csv` - Apenas anomalias

### 2. Relatórios
- `reports/eda/eda_report.md` - Relatório de análise exploratória
- `reports/anomalies/anomaly_report.md` - Relatório de anomalias
- `reports/predictions/prediction_report.md` - Relatório de modelos preditivos

### 3. Visualizações
- **EDA**: Distribuições, correlações, boxplots, categorias de risco
- **Anomalias**: PCA visualization, comparação de métodos, scores
- **Previsões**: Actual vs Predicted, feature importance, confusion matrix

### 4. Modelos Treinados
- `models/rf_regressor_*.joblib` - Modelos de regressão para scores
- `models/rf_classifier_risk.joblib` - Modelo de classificação de risco
- `models/scaler_*.joblib` - Scalers para normalização
- `models/label_encoder_risk.joblib` - Label encoder

## 🔧 Instalação Local (sem Docker)

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar pipeline completo
python src/main_pipeline.py

# Ou executar módulos individualmente
python src/data_consolidation.py
python src/exploratory_analysis.py
python src/anomaly_detection.py
python src/predictive_model.py
```

## 🧠 Módulos e Funcionalidades

### 1. Consolidação de Dados (`data_consolidation.py`)
- Lê todos os ficheiros XLSX da pasta `biomechanic scores/`
- Extrai metadados (data, câmera) dos nomes dos ficheiros
- Consolida num único dataset
- Gera metadados e estatísticas

**Estrutura dos dados:**
- `timestamp`, `scoreA`, `scoreB`, `scoreC`
- `neck`, `trunk`, `knee`, `arm`, `forearm`, `hand`
- Metadados: `source_file`, `camera_id`, `recording_date`

### 2. Análise Exploratória (`exploratory_analysis.py`)
- Estatísticas descritivas completas
- Limpeza e normalização de dados
- **Feature Engineering:**
  - `avg_score` - Score médio geral
  - `max_score` - Pior score (maior risco)
  - `joint_std` - Variabilidade postural
  - `risk_category` - Categorização de risco (Baixo/Moderado/Alto/Crítico)
  - Features temporais (hora, dia da semana, fim de semana)
- Visualizações exploratórias
- Análise de padrões e correlações

### 3. Deteção de Anomalias (`anomaly_detection.py`)
Implementa múltiplos algoritmos:
- **Isolation Forest** - Deteta outliers baseado em árvores de decisão
- **DBSCAN** - Clustering espacial (noise = anomalia)
- **Elliptic Envelope** - Assume distribuição Gaussiana
- **Statistical (Z-Score)** - Método estatístico clássico
- **Ensemble** - Combina todos os métodos para maior robustez

### 4. Modelos Preditivos (`predictive_model.py`)
- **Regressão**: Prevê scores ergonómicos (scoreA, scoreB, scoreC)
- **Classificação**: Prevê categoria de risco (Baixo/Moderado/Alto/Crítico)
- Usa Random Forest com otimização de hiperparâmetros
- Feature importance analysis
- Métricas: R², RMSE, MAE, Accuracy, Confusion Matrix

## 📈 Resultados Esperados

O pipeline gera análises completas incluindo:

1. **Insights Automáticos:**
   - Identificação de posturas de alto risco
   - Padrões temporais (horas críticas, dias da semana)
   - Correlações entre articulações e scores

2. **Deteção de Anomalias:**
   - Posturas anómalas que requerem atenção imediata
   - Validação cruzada com múltiplos algoritmos
   - Priorização por severidade

3. **Previsões:**
   - Modelo capaz de prever scores ergonómicos
   - Classificação automática de risco
   - Base para sistema de alertas em tempo real

## 🔮 Próximos Passos

Conforme definido no documento do POC:

1. **Implementação de APIs**
   - REST API para previsões em tempo real
   - Endpoints para ingestão de novos dados
   - Integração com sistemas externos

2. **Dashboard Ergonómico**
   - Interface web interativa
   - Visualizações em tempo real
   - Alertas e notificações

3. **Modelos em Tempo Real**
   - Streaming de dados
   - Previsões instantâneas
   - Sistema de alertas automáticos

4. **Integração com Sensores**
   - Captura de dados em tempo real
   - Processamento contínuo
   - Feedback imediato

## 🛠️ Tecnologias Utilizadas

- **Python 3.11**
- **Pandas** - Manipulação de dados
- **NumPy** - Computação numérica
- **Scikit-learn** - Machine Learning
- **Matplotlib / Seaborn** - Visualizações
- **Docker** - Containerização
- **Jupyter** - Análise interativa (opcional)

## 📝 Como Usar os Modelos Treinados

```python
import joblib
import pandas as pd
import numpy as np

# Carregar modelo de regressão
model = joblib.load('models/rf_regressor_scoreA.joblib')
scaler = joblib.load('models/scaler_scoreA.joblib')

# Preparar novos dados
new_data = pd.DataFrame({
    'neck': [45.0],
    'trunk': [30.0],
    'knee': [90.0],
    'arm': [60.0],
    'forearm': [45.0],
    'hand': [0.0],
    'joint_std': [25.0],
    # ... outras features
})

# Normalizar
new_data_scaled = scaler.transform(new_data)

# Fazer previsão
prediction = model.predict(new_data_scaled)
print(f"Score previsto: {prediction[0]:.2f}")

# Classificação de risco
risk_model = joblib.load('models/rf_classifier_risk.joblib')
risk_scaler = joblib.load('models/scaler_risk.joblib')
label_encoder = joblib.load('models/label_encoder_risk.joblib')

new_data_risk_scaled = risk_scaler.transform(new_data)
risk_pred = risk_model.predict(new_data_risk_scaled)
risk_category = label_encoder.inverse_transform(risk_pred)
print(f"Categoria de risco: {risk_category[0]}")
```

## 📞 Suporte

Para questões ou problemas:
1. Verificar logs do Docker: `docker-compose logs`
2. Verificar ficheiros gerados em `reports/`
3. Executar módulos individualmente para debug

## 📄 Licença

Este é um Proof of Concept (POC) para demonstração de capacidades técnicas.

---

**Desenvolvido para POC de Análise e Previsão de Posturas de Trabalho com IA** 🤖🏭
