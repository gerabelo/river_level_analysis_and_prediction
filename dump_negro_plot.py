import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def filtro_level(valor):
    if isinstance(valor, (int, float)):
        return valor != 0.0
    elif isinstance(valor, str):
        return valor.isdigit() and int(valor) != 0.0
    else:
        return False

# Caminho do arquivo
arquivo_negro = 'plot_input.csv'

# Leitura do CSV
df = pd.read_csv(arquivo_negro, parse_dates=['DATE'], dayfirst=True)

# Filtrar e preparar os dados
df = df[df['LEVEL'].apply(filtro_level)]
df['day_of_year'] = df['DATE'].dt.strftime('%j').astype(int)
df['YEAR'] = df['DATE'].dt.year
df = df.sort_values(by='DATE')

# Curvas por ano (visuais)
anos_desejados = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
df_filtrado = df[df['YEAR'].isin(anos_desejados)]

# Ponto de destaque
dia_marcador = 138

# Configurar o gráfico
plt.figure(figsize=(10, 6))
plt.title('Rio Negro')
plt.xlabel('dias')
plt.ylabel('nível (metros)')
plt.figtext(0.5, 0.01, 'Fonte: https://www.portodemanaus.com.br', ha='center', va='center', fontsize=10)

# Cores e estilos
cores = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'gray', 'cyan', 'black']
estilos = ['dotted', 'dotted', 'dotted', 'dotted', 'dotted', 'dotted', 'dotted', 'dotted', 'dashdot']
largura = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
#estilos = ['solid', 'dashed', 'dotted', 'dashdot', 'solid', 'dashed', 'dotted', 'dashdot', 'solid']

# Plotar cada ano individual
for i, ano in enumerate(anos_desejados):
    dados_ano = df_filtrado[df_filtrado['YEAR'] == ano]
    plt.plot(dados_ano['day_of_year'], dados_ano['LEVEL'], label=str(ano), color=cores[i], linewidth=largura[i], linestyle=estilos[i])

    level_marcado = dados_ano[dados_ano['day_of_year'] == dia_marcador]['LEVEL'].values
    if len(level_marcado) > 0:
        plt.scatter(dia_marcador, level_marcado, color=cores[i], marker='o', s=100, edgecolor='black', zorder=5)
        plt.text(dia_marcador, level_marcado, f'{level_marcado[0]}   ', fontsize=7, color='black', va='top', ha='right', zorder=10)

# Função para plotar curvas de média
def plot_media(dframe, anos, label, cor, largura, estilo='solid'):
    df_media = dframe[dframe['YEAR'].isin(anos)]
    if not df_media.empty:
        media_por_dia = df_media.groupby('day_of_year')['LEVEL'].mean()
        plt.plot(media_por_dia.index, media_por_dia.values, label=label, color=cor, linewidth=largura, linestyle=estilo)

# ➕ Curvas de média com intervalos fixos
plot_media(df, list(range(2022, 2025)), 'Média últimos 3 anos (2022–2024)', 'darkblue', 1.0)
plot_media(df, list(range(2020, 2025)), 'Média últimos 5 anos (2020–2024)', 'darkorange', 1.0)
plot_media(df, list(range(2015, 2025)), 'Média últimos 10 anos (2015–2024)', 'darkred', 1.0)
plot_media(df, list(range(2005, 2025)), 'Média últimos 20 anos (2005–2024)', 'navy', 1.0)

plt.legend(title='Ano / Médias')
plt.grid(True)
plt.tight_layout()
plt.show()
