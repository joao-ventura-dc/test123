# 🚀 Guia Rápido de Arranque

## Opção 1: Executar com Docker (Recomendado)

### Passo 1: Arrancar o Pipeline

```bash
# Dar permissões ao script (se necessário)
chmod +x run.sh

# Executar
./run.sh
```

**Ou manualmente:**

```bash
docker-compose up --build
```

### Passo 2: Ver os Resultados

Após a execução (demora alguns minutos), os resultados estarão em:

- **📁 data/processed/** - Datasets consolidados e processados
- **📁 models/** - Modelos ML treinados (.joblib)
- **📁 reports/** - Relatórios em Markdown e visualizações (PNG)

## Opção 2: Jupyter Notebook (Análise Interativa)

```bash
# Arrancar Jupyter
./run_jupyter.sh

# Ou manualmente
docker-compose --profile jupyter up jupyter
```

Aceder em: **http://localhost:8888**

## 📊 O que o Pipeline Faz?

1. **Consolidação** - Une todos os ficheiros XLSX num único dataset
2. **EDA** - Análise exploratória com estatísticas e visualizações
3. **Anomalias** - Detecta posturas anómalas com 5 algoritmos diferentes
4. **Previsão** - Treina modelos para prever scores e risco

## 🎯 Principais Outputs

### Datasets
- `consolidated_data.csv` - Todos os dados unidos
- `data_with_features.csv` - Com features engenharia
- `data_with_anomalies.csv` - Com anomalias identificadas

### Modelos
- `rf_regressor_*.joblib` - Previsão de scores
- `rf_classifier_risk.joblib` - Classificação de risco

### Relatórios
- `reports/eda/eda_report.md` - Análise exploratória
- `reports/anomalies/anomaly_report.md` - Anomalias detectadas
- `reports/predictions/prediction_report.md` - Performance dos modelos

### Visualizações (PNG)
- Distribuições de scores
- Matriz de correlação
- Boxplots de articulações
- Distribuição de risco
- Visualização de anomalias (PCA)
- Comparação de métodos
- Previsões (actual vs predicted)
- Feature importance

## 🧹 Limpeza

Para remover outputs e recomeçar:

```bash
./cleanup.sh
```

## ⚡ Execução Rápida (tudo numa linha)

```bash
docker-compose up --build && echo "✅ Concluído! Verifique a pasta reports/"
```

## 🆘 Problemas Comuns

### "Pasta biomechanic scores não encontrada"
- Certifica-te que os ficheiros XLSX estão na pasta `biomechanic scores/`

### "Docker não instalado"
- Instala Docker Desktop: https://www.docker.com/products/docker-desktop

### "Pipeline falhou"
- Verifica logs: `docker-compose logs`
- Verifica se os XLSX têm as colunas corretas

### "Sem outputs"
- Verifica se o container terminou com sucesso
- Os ficheiros são criados dentro do container e copiados para o host

## 📈 Próximos Passos

Após executar o pipeline:

1. Lê os relatórios em `reports/`
2. Vê as visualizações (ficheiros PNG)
3. Usa os modelos treinados em `models/` para fazer previsões
4. Explora os dados processados em `data/processed/`

---

**Tempo estimado de execução:** 2-5 minutos (depende do número de ficheiros)
