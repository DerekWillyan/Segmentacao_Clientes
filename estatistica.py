import pandas as pd
import numpy as np
from scipy.stats import bartlett, levene
from scipy.stats import kstest, shapiro, norm
from scipy.stats import skew, kurtosis
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import gower
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import prince
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import itertools

# Cochran's C test
def cochran_test(*grupos):
    variancias = [np.var(g, ddof=1) for g in grupos]
    G = max(variancias) / sum(variancias)
    return G

# Fmax de Hartley
def fmax_test(*grupos):
    variancias = [np.var(g, ddof=1) for g in grupos]
    fmax = max(variancias) / min(variancias)
    return fmax

# Simulação para p-valor (bootstrap)
def simulate_pvalue(stat_func, grupos, stat_obs, n_iter=10000, random_state=None):
    rng = np.random.default_rng(random_state)
    concat = np.concatenate(grupos)
    tamanhos = [len(g) for g in grupos]
    
    count = 0
    for _ in range(n_iter):
        rng.shuffle(concat)
        amostras = np.split(concat.copy(), np.cumsum(tamanhos)[:-1])
        stat_sim = stat_func(*amostras)
        if stat_sim >= stat_obs:
            count += 1
    return count / n_iter

# Função principal
def teste_homogeneidade(df, colunas, metodo='bartlett', n_iter=10000):
    grupos = [df[col].dropna().values for col in colunas]
    
    if metodo.lower() == 'bartlett':
        stat, p = bartlett(*grupos)
        return {'teste': 'Bartlett (Qui-quadrado)', 'stat': stat, 'pvalue': p}

    elif metodo.lower() == 'levene':
        stat, p = levene(*grupos, center='mean')
        return {'teste': 'Levene (F)', 'stat': stat, 'pvalue': p}

    elif metodo.lower() == 'cochran':
        stat = cochran_test(*grupos)
        p = simulate_pvalue(cochran_test, grupos, stat, n_iter=n_iter)
        return {'teste': 'C de Cochran', 'stat': stat, 'pvalue': p}

    elif metodo.lower() == 'fmax':
        stat = fmax_test(*grupos)
        p = simulate_pvalue(fmax_test, grupos, stat, n_iter=n_iter)
        return {'teste': 'Fmax de Hartley', 'stat': stat, 'pvalue': p}

    else:
        raise ValueError("Método inválido. Escolha: 'bartlett', 'levene', 'cochran', ou 'fmax'.")

# Teste de Kolmogorov-Smirnov (KS) padronizado com Z-score
def teste_normalidade_ks(df, colunas):
    resultados = {}
    for coluna in colunas:
        dados = df[coluna].dropna()
        dados = (dados - dados.mean()) / dados.std()
        _, pvalue = kstest(dados, 'norm')
        resultados[coluna] = pvalue
    return resultados

# Teste de Shapiro-Wilk
def teste_normalidade_shapiro(df, colunas):
    resultados = {}
    for coluna in colunas:
        dados = df[coluna].dropna()
        _, pvalue = shapiro(dados)
        resultados[coluna] = pvalue
    return resultados

# Função unificada
def teste_normalidade(df, colunas, metodo='ks'):
    metodo = metodo.lower()
    
    if metodo == 'ks':
        return teste_normalidade_ks(df, colunas)
    elif metodo == 'shapiro-wilk':
        return teste_normalidade_shapiro(df, colunas)
    else:
        raise ValueError("Método inválido. Escolha entre: 'ks', 'shapiro-wilk'.")

def calcular_assimetria(dados):
    """
    Calcula a assimetria (coeficiente de Fisher de assimetria)
    e imprime a interpretação.
    Aceita um DataFrame ou uma série do pandas.
    """
    if isinstance(dados, pd.DataFrame):
        for coluna in dados.columns:
            assimetria = skew(dados[coluna], bias=False)
            print(f"Coeficiente de Assimetria de Fisher para a coluna '{coluna}': {assimetria:.4f}")
            # Interpretação da assimetria
            if assimetria > 0:
                print(f"Distribuição assimétrica à direita (assimetria positiva) para a coluna '{coluna}'")
            elif assimetria < 0:
                print(f"Distribuição assimétrica à esquerda (assimetria negativa) para a coluna '{coluna}'")
            else:
                print(f"Distribuição simétrica para a coluna '{coluna}'")
    else:
        assimetria = skew(dados, bias=False)
        print(f"Coeficiente de Assimetria de Fisher: {assimetria:.4f}")
        # Interpretação da assimetria
        if assimetria > 0:
            print("Distribuição assimétrica à direita (assimetria positiva)")
        elif assimetria < 0:
            print("Distribuição assimétrica à esquerda (assimetria negativa)")
        else:
            print("Distribuição simétrica")
    
    return assimetria

def calcular_curtose(dados):
    """
    Calcula a curtose (coeficiente de Fisher de curtose)
    e imprime a interpretação.
    Aceita um DataFrame ou uma série do pandas.
    OBS: resultado é subtraído de 3 (normal = 0)
    """
    if isinstance(dados, pd.DataFrame):
        for coluna in dados.columns:
            curtose_fisher = kurtosis(dados[coluna], fisher=True, bias=False)
            print(f"Coeficiente de Curtose de Fisher para a coluna '{coluna}': {curtose_fisher:.4f}")
            # Interpretação da curtose
            if curtose_fisher > 0:
                print(f"Distribuição leptocúrtica (pico mais alto que o normal) para a coluna '{coluna}'")
            elif curtose_fisher < 0:
                print(f"Distribuição platicúrtica (pico mais achatado) para a coluna '{coluna}'")
            else:
                print(f"Distribuição mesocúrtica (curtose normal) para a coluna '{coluna}'")
    else:
        curtose_fisher = kurtosis(dados, fisher=True, bias=False)
        print(f"Coeficiente de Curtose de Fisher: {curtose_fisher:.4f}")
        # Interpretação da curtose
        if curtose_fisher > 0:
            print("Distribuição leptocúrtica (pico mais alto que o normal)")
        elif curtose_fisher < 0:
            print("Distribuição platicúrtica (pico mais achatado)")
        else:
            print("Distribuição mesocúrtica (curtose normal)")
    
    return curtose_fisher

def agrupamento_hierarquico(
    df,
    colunas_alvo,
    metodo='ward',
    distancia='euclidean',
    altura_corte=None,
    num_clusters=None,
    nome_coluna_cluster='cluster',
    atribuir_ao_dataframe=True
):
    """
    Realiza agrupamento hierárquico em colunas selecionadas do DataFrame.
    
    Parâmetros:
        df: pandas.DataFrame
        colunas_alvo: list[str] - Colunas a serem consideradas no agrupamento
        metodo: str - Método de linkage: 'single', 'complete', 'average', 'ward'
        distancia: str - Tipo de distância: 'euclidean', 'sqeuclidean' ou 'gower'
        altura_corte: float - Altura do corte no dendrograma para definir os clusters
        num_clusters: int - Número fixo de clusters (ignora altura_corte)
        nome_coluna_cluster: str - Nome da nova coluna com os rótulos dos clusters
        atribuir_ao_dataframe: bool - Se True, adiciona os clusters ao DataFrame original
    
    Retorna:
        Se atribuir_ao_dataframe=True: retorna o DataFrame com a nova coluna.
        Se atribuir_ao_dataframe=False: retorna um array com os rótulos dos clusters.
    """
    
    for col in colunas_alvo:
        if col not in df.columns:
            raise ValueError(f"A coluna '{col}' não está no DataFrame.")
    
    dados = df[colunas_alvo].dropna().copy()

    # Verifica tipo de distância
    if distancia == 'euclidean':
        matriz_ligacao = linkage(dados, method=metodo, metric='euclidean')
    elif distancia == 'sqeuclidean':
        dist_sq = pdist(dados, metric='sqeuclidean')
        matriz_ligacao = linkage(dist_sq, method=metodo)
    elif distancia == 'gower':
        dist_gower = gower.gower_matrix(dados)
        dist_condensada = squareform(dist_gower, checks=False)
        matriz_ligacao = linkage(dist_condensada, method=metodo)
    else:
        raise ValueError("A distância deve ser 'euclidean', 'sqeuclidean' ou 'gower'.")

    # Plot do dendrograma
    plt.figure(figsize=(12, 6))
    dendrogram(
        matriz_ligacao,
        labels=dados.index,
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=altura_corte
    )
    plt.title('Dendrograma - Agrupamento Hierárquico')
    plt.xlabel('Observações')
    plt.ylabel('Distância')
    if altura_corte:
        plt.axhline(y=altura_corte, c='red', linestyle='--', label=f'Altura de Corte = {altura_corte}')
        plt.legend()
    plt.tight_layout()
    plt.show()

    # Criação dos clusters
    if num_clusters:
        clusters = fcluster(matriz_ligacao, num_clusters, criterion='maxclust')
    elif altura_corte:
        clusters = fcluster(matriz_ligacao, altura_corte, criterion='distance')
    else:
        raise ValueError("Você precisa informar 'altura_corte' ou 'num_clusters'.")

    print(f"Clusters formados: {len(np.unique(clusters))}")

    if atribuir_ao_dataframe:
        df[nome_coluna_cluster] = np.nan
        df.loc[dados.index, nome_coluna_cluster] = clusters
        return df
    else:
        return clusters

def anova_significativa(df, variavel_dependente, fatores, alpha=0.05):
    """
    Realiza ANOVA automaticamente e retorna apenas os efeitos significativos (p < alpha).
    
    Parâmetros:
    - df: DataFrame com os dados
    - variavel_dependente: str, nome da variável dependente
    - fatores: list[str], variáveis independentes (fatores)
    - alpha: nível de significância (default: 0.05)

    Retorna:
    - DataFrame com apenas os efeitos que rejeitam a hipótese nula (p < alpha)
    """
    # Tipo de ANOVA
    num_fatores = len(fatores)
    if num_fatores == 1:
        tipo = "One-Way ANOVA"
    elif num_fatores == 2:
        tipo = "Two-Way ANOVA"
    else:
        tipo = f"{num_fatores}-Way ANOVA"

    print(f"\n📊 Tipo de ANOVA detectado: {tipo}\n")

    # Fórmula com interações
    formula = f"{variavel_dependente} ~ " + " * ".join([f"C({f})" for f in fatores])
    
    # Ajustar modelo
    modelo = ols(formula, data=df).fit()
    anova = sm.stats.anova_lm(modelo, typ=2)

    # Mostrar floats com 4 casas decimais
    pd.options.display.float_format = '{:.3f}'.format

    # Filtrar apenas efeitos com p < alpha
    significativos = anova[anova["PR(>F)"] < alpha]

    return significativos

def regressao_linear_multipla(df, var_dependente, vars_independentes, max_comb=3):
    pd.options.display.float_format = '{:.3f}'.format
    # Dados para regressão
    X = df[vars_independentes]
    y = df[var_dependente]
    
    # Regressão com todas as variáveis
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # Tabela de coeficientes e p-valores
    resultados = pd.DataFrame({
        'Variável': model.params.index,
        'Coeficiente': model.params.values,
        'P-Valor': model.pvalues.values
    })

    # Separar alpha e betas
    alpha = resultados[resultados['Variável'] == 'const']
    betas = resultados[resultados['Variável'] != 'const']

    # Variáveis com p-valor <= 0.05
    variaveis_significativas = betas[betas['P-Valor'] <= 0.05].reset_index(drop=True)

    # Combinações significativas
    combinacoes_significativas = []
    for r in range(1, min(max_comb, len(vars_independentes)) + 1):
        for combo in itertools.combinations(vars_independentes, r):
            X_sub = df[list(combo)]
            X_sub_const = sm.add_constant(X_sub)
            model_sub = sm.OLS(y, X_sub_const).fit()
            pvals_sub = model_sub.pvalues.drop('const', errors='ignore')
            if all(pvals_sub <= 0.05):
                linha = {'Combinação': ' + '.join(combo)}
                for var in combo:
                    linha[var] = round(pvals_sub[var], 5)
                combinacoes_significativas.append(linha)

    combinacoes_df = pd.DataFrame(combinacoes_significativas)

    return alpha.reset_index(drop=True), betas.reset_index(drop=True), variaveis_significativas, combinacoes_df

def analise_fatorial(df, plot_loading=True, limiar_carga=0.4):
    # Padronização
    scaler = StandardScaler()
    dados_padronizados = scaler.fit_transform(df)

    # Autovalores para definir número de fatores
    fa = FactorAnalyzer(rotation=None)
    fa.fit(dados_padronizados)
    eigenvalues, _ = fa.get_eigenvalues()

    tabela_autovalores = pd.DataFrame({
        'Fator': [f'Fator {i+1}' for i in range(len(eigenvalues))],
        'Autovalor': eigenvalues
    })

    n_fatores = sum(eigenvalues > 1)

    # Análise fatorial com rotação
    fa = FactorAnalyzer(n_factors=n_fatores, rotation='varimax')
    fa.fit(dados_padronizados)

    cargas = pd.DataFrame(fa.loadings_,
                          index=df.columns,
                          columns=[f'Fator {i+1}' for i in range(n_fatores)])

    comunalidades = pd.DataFrame(fa.get_communalities(),
                                 index=df.columns,
                                 columns=['Comunalidade'])

    # Mapeamento das variáveis associadas a cada fator
    combinacoes = {}
    for fator in cargas.columns:
        variaveis = cargas.index[np.abs(cargas[fator]) >= limiar_carga].tolist()
        combinacoes[fator] = variaveis
    tabela_combinacoes = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in combinacoes.items()]))

    # Loading plot
    if plot_loading and n_fatores >= 2:
        plt.figure(figsize=(8, 6))
        x = cargas.iloc[:, 0]
        y = cargas.iloc[:, 1]
        plt.scatter(x, y)
        for i, var in enumerate(cargas.index):
            plt.text(x[i], y[i], var, fontsize=9)
        plt.axhline(0, color='grey', lw=1)
        plt.axvline(0, color='grey', lw=1)
        plt.title('Loading Plot (Fator 1 x Fator 2)')
        plt.xlabel('Fator 1')
        plt.ylabel('Fator 2')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return tabela_autovalores, cargas, comunalidades, tabela_combinacoes

def analise_correspondencia(df, colunas, n_componentes=2, plotar=True):
    """
    Realiza Análise de Correspondência Simples ou Múltipla com base na quantidade de colunas fornecidas.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com os dados
    - colunas (list): lista com os nomes das colunas a serem usadas
    - n_componentes (int): número de componentes a extrair
    - plotar (bool): se True, exibe o gráfico (mapa perceptual)

    Retorna:
    - modelo ajustado (prince.CA ou prince.MCA)
    - DataFrame com coordenadas dos componentes
    """
    df_selecionado = df[colunas].copy()

    if len(colunas) == 2:
        tipo = 'simples'
    elif len(colunas) > 2:
        tipo = 'multipla'
    else:
        raise ValueError("É necessário fornecer pelo menos duas colunas.")

    if tipo == 'simples':
        tabela = pd.crosstab(df_selecionado[colunas[0]], df_selecionado[colunas[1]])
        ca = prince.CA(n_components=n_componentes, random_state=42)
        ca = ca.fit(tabela)
        row_coords = ca.row_coordinates(tabela)
        col_coords = ca.column_coordinates(tabela)

        if plotar:
            if row_coords.shape[1] >= 2 and col_coords.shape[1] >= 2:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=row_coords.iloc[:, 0], y=row_coords.iloc[:, 1], s=100, label='Linhas')
                sns.scatterplot(x=col_coords.iloc[:, 0], y=col_coords.iloc[:, 1], s=100, marker='s', label='Colunas')
                for i, txt in enumerate(row_coords.index):
                    plt.annotate(txt, (row_coords.iloc[i, 0], row_coords.iloc[i, 1]), fontsize=9)
                for i, txt in enumerate(col_coords.index):
                    plt.annotate(txt, (col_coords.iloc[i, 0], col_coords.iloc[i, 1]), fontsize=9, color='blue')
                plt.axhline(0, color='gray', linestyle='--')
                plt.axvline(0, color='gray', linestyle='--')
                plt.title('Análise de Correspondência Simples (Mapa Perceptual)')
                plt.xlabel('Componente 1')
                plt.ylabel('Componente 2')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("Apenas 1 componente foi extraído. Mapa perceptual não será plotado.")

        return ca, row_coords.join(col_coords, how='outer')

    else:  # tipo == 'multipla'
        mca = prince.MCA(n_components=n_componentes, random_state=42)
        mca = mca.fit(df_selecionado)
        coords = mca.row_coordinates(df_selecionado)

        if plotar:
            if coords.shape[1] >= 2:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(x=coords.iloc[:, 0], y=coords.iloc[:, 1], s=100)
                for i, txt in enumerate(coords.index):
                    plt.annotate(txt, (coords.iloc[i, 0], coords.iloc[i, 1]), fontsize=9)
                plt.axhline(0, color='gray', linestyle='--')
                plt.axvline(0, color='gray', linestyle='--')
                plt.title('Análise de Correspondência Múltipla (Mapa Perceptual)')
                plt.xlabel('Componente 1')
                plt.ylabel('Componente 2')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("Apenas 1 componente foi extraído. Mapa perceptual não será plotado.")

        return mca, coords

def kmeans_cluster_plot(df, x_col, y_col, n_clusters=3, add_cluster_column=True):
    """
    Aplica KMeans e plota gráfico de dispersão com clusters e centróides.

    Parâmetros:
    - df: DataFrame de entrada
    - x_col: nome da coluna para o eixo x
    - y_col: nome da coluna para o eixo y
    - n_clusters: número de clusters
    - add_cluster_column: se True, adiciona a coluna 'cluster' ao DataFrame

    Retorna:
    - df (modificado com coluna 'cluster', se add_cluster_column=True)
    """

    # Subset dos dados para clustering
    X = df[[x_col, y_col]]

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='cluster', palette='Set2', s=60)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.6, marker='X', label='Centroides')
    plt.title(f'KMeans Clustering com {n_clusters} Clusters')
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
    plt.show()

    # Retornar o DataFrame (com ou sem a coluna 'cluster')
    if add_cluster_column:
        return df
    else:
        return df.drop(columns=['cluster'])
