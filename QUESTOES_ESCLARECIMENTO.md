# ❓ Questões para Esclarecimento - POC Análise de Posturas

**Data:** 2025-12-18
**Para:** Cliente/Equipa do Projeto
**De:** Equipa de Desenvolvimento (João & Fábio)

---

## 📊 1. Escala Numérica dos Scores e Variáveis

### Contexto
Atualmente, temos no dataset as seguintes variáveis:
- **Scores globais:** `scoreA`, `scoreB`, `scoreC`
- **Variáveis posturais:** `neck`, `trunk`, `knee`, `arm`, `forearm`, `hand`

### Questões:

**1.1.** Todos estes valores usam a **mesma escala numérica** (ex: 1 a 9)?

**1.2.** Existe **diferença na gama de valores** entre:
- Scores globais (A, B, C)
- Classificações das partes do corpo (neck, trunk, etc.)

**1.3.** Qual é o **significado exato de cada escala**?
- Exemplo: 1 = postura ideal, 9 = postura crítica?
- Ou a interpretação é diferente?

**1.4.** Existe documentação sobre:
- Limites mínimos e máximos de cada variável
- Interpretação clínica/ergonómica de cada valor
- Thresholds de risco (ex: score > 7 = alto risco)

---

## 🎯 2. Significado dos Scores A, B e C

### Contexto
O dataset inclui três scores globais (`scoreA`, `scoreB`, `scoreC`), mas o seu significado concreto não está documentado.

### Questões:

**2.1.** O que representa **cada score**?
- Score A = ?
- Score B = ?
- Score C = ?

**2.2.** Estes scores seguem alguma **metodologia específica**?
- REBA (Rapid Entire Body Assessment)?
- RULA (Rapid Upper Limb Assessment)?
- OWAS (Ovako Working Posture Analysis System)?
- Outra metodologia proprietária?

**2.3.** Como são **calculados** estes scores?
- São calculados a partir das variáveis posturais (neck, trunk, etc.)?
- Ou são medidos/calculados independentemente?

**2.4.** Qual a **relação entre os três scores**?
- Complementam-se?
- Representam diferentes perspetivas da mesma postura?
- Devemos usá-los todos ou há um que é mais importante?

---

## 👤 3. Identificação de Pessoas no Dataset

### Contexto
Os ficheiros no dataset têm o formato: `YYYYMMDDHHMMSS_cameraX_computed.xlsx`

Exemplo: `20240201062522_camera3_computed.xlsx`

### Questões:

**3.1.** Cada **câmara representa uma pessoa**?
- camera1 = Pessoa A
- camera2 = Pessoa B
- camera3 = Pessoa C
- Etc.

**3.2.** Ou as câmaras representam **ângulos diferentes** da mesma pessoa?
- Exemplo: camera1 = vista frontal, camera2 = vista lateral, etc.

**3.3.** Podemos usar o **`camera_id` como identificador único** de pessoa?

**3.4.** Se cada câmara NÃO representa uma pessoa:
- Qual é o identificador correto para distinguir pessoas?
- Existe alguma variável que identifique a pessoa no dataset?
- Como devemos agregar os dados de múltiplas câmaras?

**3.5.** Quantas **pessoas diferentes** estão no dataset atual?

**3.6.** As pessoas são sempre as mesmas ao longo dos diferentes dias?
- Ou há variação no número/identidade das pessoas monitorizadas?

---

## 📈 4. Questões Adicionais para Análise

### 4.1. Contexto Temporal
- Qual é a **frequência de amostragem** dos dados?
  - Ex: 1 medição por segundo? Por minuto?
- Os timestamps são sequenciais dentro de cada ficheiro?

### 4.2. Contexto Laboral
- Que **tipo de trabalho** estavam as pessoas a realizar?
- Há informação sobre:
  - Tipo de tarefa
  - Turno de trabalho
  - Pausas/descansos
  - Condições ambientais

### 4.3. Missing Data
- Quando há valores em falta (NaN), o que representam?
  - Sensor não detetou?
  - Pessoa não estava visível?
  - Erro de medição?

---

## 🎯 Impacto no Modelo

### Porque é importante esclarecer:

**Para os Scores (A, B, C):**
- Afeta a **interpretação dos resultados** dos modelos preditivos
- Define os **thresholds de alerta** no sistema
- Determina qual score é mais relevante para priorizar

**Para a Identificação:**
- Afeta o **design do modelo**:
  - Se cada câmara = pessoa → podemos fazer análise por pessoa
  - Se câmaras = ângulos → precisamos agregar antes
- Impacta a **feature engineering**:
  - Podemos criar features de histórico por pessoa
  - Podemos comparar padrões entre pessoas
- Determina a **estratégia de validação** dos modelos:
  - Train/test split por pessoa vs por tempo

**Para as Escalas:**
- Afeta a **normalização** dos dados
- Define os **limites para deteção de anomalias**
- Influencia a **interpretação clínica** dos alertas

---

## 📝 Ações Necessárias

Por favor, fornecer:

1. ✅ **Documentação técnica** sobre:
   - Escalas numéricas de todas as variáveis
   - Metodologia de cálculo dos scores
   - Interpretação ergonómica dos valores

2. ✅ **Schema dos dados** detalhado:
   - Dicionário de dados completo
   - Relação entre câmaras e pessoas
   - Estrutura temporal dos dados

3. ✅ **Contexto do negócio**:
   - Objetivo final do sistema
   - Use cases prioritários
   - Thresholds de risco atuais (se existirem)

---

## 📞 Contacto

Para esclarecimentos, contactar:
- João & Fábio
- Equipa de Desenvolvimento - POC Análise de Posturas

**Data limite para resposta:** [DEFINIR DATA]

---

**Nota:** Estas questões são essenciais para garantir que os modelos de IA estão corretamente calibrados e que as previsões são clinicamente válidas e úteis para o contexto ergonómico pretendido.
