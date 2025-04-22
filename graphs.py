# graphs.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
import pandas as pd

# ------------------------------------------
# Função para Regressão Múltiplas
# ------------------------------------------

def regressao_multipla(df, dependente, independentes):
    """
    Realiza uma regressão múltipla usando statsmodels.

    Parâmetros:
    df (pd.DataFrame): O DataFrame contendo os dados.
    dependente (str): O nome da variável dependente.
    independentes (list): Lista com os nomes das variáveis independentes.

    Retorna:
    O resumo do modelo de regressão.
    """
    # Adiciona uma constante ao modelo (intercepto)
    X = sm.add_constant(df[independentes])  # Variáveis independentes
    y = df[dependente]  # Variável dependente

    # Ajusta o modelo de regressão
    modelo = sm.OLS(y, X).fit()

    # Retorna o resumo do modelo
    return modelo

# ------------------------------------------
# Função para Regressão Linear
# ------------------------------------------

def regressao_linear(df=None, x_col=None, y_col=None, n_clusters=None, show_trendline=False, 
                 title="Scatter Plot", xlabel="X Axis", ylabel="Y Axis",
                 trendline_color='red', show_correlation=True):
    
    """
    Cria um gráfico scatter com opções de clustering e linha de tendência.
    
    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    x_col (str): Nome da coluna no DataFrame para o eixo X.
    y_col (str): Nome da coluna no DataFrame para o eixo Y.
    n_clusters (int): Número de clusters para K-Means.
    show_trendline (bool): Se True, mostra a linha de regressão linear.
    title (str): Título do gráfico.
    xlabel (str): Rótulo do eixo X.
    ylabel (str): Rótulo do eixo Y.
    trendline_color (str): Cor da linha de tendência.
    show_correlation (bool): Se True, exibe o coeficiente de correlação no gráfico.
    
    Retorna:
    model: Modelo de regressão linear treinado (se show_trendline=True).
    """
    # Verifica se o DataFrame foi fornecido
    if df is None or x_col is None or y_col is None:
        raise ValueError("Você deve fornecer um DataFrame e os nomes das colunas para x e y.")
    
    # Extrai os dados das colunas especificadas
    x = df[x_col].values
    y = df[y_col].values
    
    # Aplica K-Means se especificado
    if n_clusters is not None:
        data = np.column_stack((x, y))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_
        colors = plt.cm.viridis(labels / (n_clusters - 1)) if n_clusters > 1 else 'blue'
    else:
        colors = 'blue'
        centroids = None
    
    # Calcula o coeficiente de correlação de Pearson
    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]
    
    # Configura o gráfico
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c=colors, alpha=0.6, edgecolors='w', label='Dados')
    
    # Inicializa o modelo de regressão linear
    model = None
    if show_trendline:
        # Treina o modelo de regressão linear
        model = LinearRegression()
        model.fit(x.reshape(-1, 1), y)  # Ajusta o modelo aos dados
        
        # Calcula a linha de tendência
        trendline_x = np.linspace(min(x), max(x), 100).reshape(-1, 1)
        trendline_y = model.predict(trendline_x)
        
        # Calcula o R²
        r_squared = r2_score(y, model.predict(x.reshape(-1, 1)))
        
        # Plota a linha de tendência
        plt.plot(trendline_x, trendline_y, color=trendline_color, linestyle='--', 
                 label=f'Linha de Tendência\n$R^2 = {r_squared:.2f}$')
    
    # Centróides
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red', 
                    s=100, linewidths=1.5, label='Centroids')
    
    # Exibe o coeficiente de correlação DENTRO do gráfico
    if show_correlation:
        # Posiciona o texto no canto superior direito do gráfico
        plt.text(
            0.95, 0.95,  # Posição relativa (95% da largura e altura do gráfico)
            f'Correlação: {correlation_coefficient:.2f}', 
            transform=plt.gca().transAxes,  # Usa coordenadas relativas ao eixo
            fontsize=12, 
            ha='right',  # Alinhamento horizontal à direita
            va='top',    # Alinhamento vertical ao topo
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')  # Caixa de texto
        )
    
    # Configurações finais
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.5)
    if show_trendline or centroids is not None:
        plt.legend()
    plt.show()
    
    # Retorna o modelo treinado (se aplicável)
    return model

# ------------------------------------------
# Função para Bar Plot
# ------------------------------------------
def bar_plot(data, x=None, y=None, hue=None, estimator='sum', 
             color=None, colormap='viridis', title="Bar Plot", 
             xlabel="Categories", ylabel="Values", figsize=(10, 6), 
             annotate=False, rotation=45, stacked=False, legend=True):
    """
    Cria um gráfico de barras a partir de dados em listas, dicionários ou DataFrames.

    Parâmetros:
    data (list/dict/DataFrame): Dados de entrada.
    x (str/list): Nomes das categorias no eixo X (se data for DataFrame).
    y (str/list): Nomes das colunas com valores (se data for DataFrame).
    hue (str): Coluna para agrupamento (barras agrupadas/empilhadas).
    estimator (str): Função de agregação ('sum', 'mean', 'max', etc.).
    error_bars (list/array): Valores para barras de erro.
    color (str/list): Cor(es) das barras.
    colormap (str): Mapa de cores do matplotlib.
    title (str): Título do gráfico.
    xlabel/ylabel (str): Rótulos dos eixos.
    figsize (tuple): Tamanho da figura.
    annotate (bool): Mostrar valores nas barras.
    rotation (int): Rotação dos rótulos do eixo X.
    stacked (bool): Barras empilhadas se True.
    legend (bool): Mostrar legenda.
    """
    # Pré-processamento para DataFrames
    if isinstance(data, pd.DataFrame):
        if x and y:
            if hue:
                df_grouped = data.groupby([x, hue])[y].agg(estimator).unstack()
                categories = df_grouped.index
                values = df_grouped.values.T
                legend_labels = df_grouped.columns
            else:
                categories = data[x].values
                values = data[y].values
        else:
            raise ValueError("Especifique 'x' e 'y' para DataFrames")
    
    # Configurações de cores
    n_bars = values.shape[0] if len(values.shape) > 1 else 1
    colors = (
        plt.cm.get_cmap(colormap)(np.linspace(0, 1, n_bars)) 
        if not color else 
        [color] * n_bars if isinstance(color, str) else color
    )

    # Plotagem
    fig, ax = plt.subplots(figsize=figsize)
    x_pos = np.arange(len(categories))
    bar_width = 0.8 / n_bars if n_bars > 1 and not stacked else 0.8

    for i in range(n_bars):
        current_values = values[i] if len(values.shape) > 1 else values
        offset = bar_width * i if n_bars > 1 and not stacked else 0
        
        ax.bar(
            x_pos + offset,
            current_values,
            width=bar_width,
            color=colors[i],
            label=legend_labels[i] if hue else f'Grupo {i+1}',
            bottom=(values[:i].sum(axis=0) if stacked and i > 0 else None)
        )

    # Personalizações
    ax.set_xticks(x_pos + (bar_width * (n_bars - 1) / 2 if n_bars > 1 else 0))
    ax.set_xticklabels(categories, rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if legend and n_bars > 1:
        ax.legend()
    
    if annotate:
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Módulo de gráficos importado com sucesso!")