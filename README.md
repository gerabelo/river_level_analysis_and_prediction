# Análise e Previsão do Nível do Rio Negro

Este repositório contém dois scripts principais para análise histórica e previsão do nível do Rio Negro:

- `dump_negro_plot.py`: Gera gráficos históricos de nível do rio por ano e curvas de média.
- `LSTM.py`: Treina e utiliza um modelo LSTM para prever níveis futuros com base em séries temporais.

---

## 📊 Visualização Histórica (`dump_negro_plot.py`)

### Objetivo

Gera um gráfico com os níveis históricos do Rio Negro por ano, destacando uma data específica, além de incluir curvas de média para os últimos 3, 5, 10 e 20 anos.

### Entrada esperada

- Arquivo CSV chamado `plot_input.csv` com ao menos duas colunas:
  - `DATE`: data no formato `dd/mm/yyyy` ou detectável via `pandas`.
  - `LEVEL`: nível do rio (float).

### Execução

```bash
python dump_negro_plot.py
```
### Saída
Gráfico exibido com os anos especificados e curvas de médias.

Fonte dos dados: Porto de Manaus

## 🤖 Previsão com LSTM (LSTM.py)

### Objetivo

Utiliza uma rede neural LSTM para prever os próximos 15 dias de nível do Rio Negro, com base nos dados históricos.

### Entrada esperada

- Arquivo CSV chamado `stm_input.csv` com as colunas:
  - `DATE`: data no formato `dd/mm/yyyy`.
  -  `LEVEL`: nível do rio (float).

### Execução

```bash
python LSTM.py
```

### Saídas
Arquivo de imagem previsao_lstm.png com:
- Dados reais
- Previsões de treino, validação e teste
- Previsão para os próximos dias
- Modelo salvo em modelo_lstm.h5 após treinamento (ou reutilizado se já existir)

## 🧩 Requisitos
Instale as dependências com:

```bash
pip install -r requirements.txt
```

requirements.txt
- pandas
- numpy
- matplotlib
- scikit-learn
- tensorflow

## 📁 Estrutura sugerida do projeto
.
├── dump_negro_plot.py
├── LSTM.py
├── plot_input.csv
├── lstm_input.csv
├── previsao_lstm.png
├── modelo_lstm.h5
├── requirements.txt
└── README.md

## 📌 Notas
Certifique-se de que os arquivos de entrada estejam no mesmo diretório dos scripts.

O script LSTM usa matplotlib.use('Agg'), então não abre o gráfico na tela, apenas salva como imagem.

Os dados devem ser organizados em formato limpo, com valores nulos tratados previamente.

## 📜 Licença
Este projeto está licenciado sob a MIT License, salvo disposição em contrário.