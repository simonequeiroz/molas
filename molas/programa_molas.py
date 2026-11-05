# ============================================================
# Análise de molas sanfonadas de PLA impressas em 3D
# Estrutura do modelo do professor: entrada -> cálculo por mola
# -> intervalos de confiança -> apresentação -> gráficos
#
# Critério dos ensaios: quatro cargas menores e uma carga maior, até o escoamento.
# A rigidez usa as quatro cargas menores; a maior carga caracteriza o escoamento.
# ============================================================

# Carregamento das bibliotecas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.ticker import FuncFormatter
from scipy import stats

# OBS: caso não rode, verifique se o pacote openpyxl foi instalado

# Pasta do programa (planilhas e resultados ficam aqui)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------
# Entrada de dados
# ------------------------------------------------------------
arquivo_testes = "Planilha_Python_CIDA_2.xlsx"     # testes próprios
arquivo_professor = "experimento.xlsx"             # dados do professor

# --- Testes próprios ---
# tabela1: resumo, uma linha por mola (aba Página1)
tabela1_testes = pd.read_excel(arquivo_testes, sheet_name="Página1", header=0, usecols="A:K")
tabela1_testes = tabela1_testes.rename(columns={"n": "N", "m": "M"})

# tabela2: medições, 5 cargas por mola (aba Página2), colunas de fórmula K e L não são lidas
tabela2_testes = pd.read_excel(arquivo_testes, sheet_name="Página2", header=0, usecols="A:J")
tabela2_testes.columns = ["ID", "C", "h", "t", "L", "N", "Ltot", "M", "F", "delta"]

# A coluna ID só está preenchida em uma linha de cada mola.
# Por isso, cada linha recebe o ID da mola, identificada pela geometria (mesmas 7 medidas).
geo = ["C", "h", "t", "L", "N", "Ltot", "M"]
tabela2_testes["grupo"] = tabela2_testes.groupby(geo, sort=False).ngroup()
tabela2_testes["ID"] = tabela2_testes["grupo"].map(
    tabela2_testes.dropna(subset=["ID"]).groupby("grupo")["ID"].first())

# --- Dados do professor ---
# tabela1: resumo, uma linha por mola (aba tabela1; os dados começam em B2)
tabela1_professor = pd.read_excel(arquivo_professor, sheet_name="tabela1", header=1, usecols="B:K")
tabela1_professor = tabela1_professor.rename(columns={"n": "N", "m": "M"})

# tabela2: medições (aba tabela2); as colunas de fórmula rigidez e dutilidade não são lidas
tabela2_professor = pd.read_excel(arquivo_professor, sheet_name="tabela2", header=1, usecols="B:D")
tabela2_professor.columns = ["ID", "F", "delta"]

# Esta aba não traz a geometria: o comprimento C de cada mola vem da tabela1, pelo ID
tabela2_professor["C"] = tabela2_professor["ID"].map(tabela1_professor.set_index("ID")["C"])

# --- Junção das duas planilhas (só as colunas que existem nas duas) ---
tabela1_testes["origem"] = "Testes próprios"
tabela1_professor["origem"] = "Professor"
tabela2_testes["origem"] = "Testes próprios"
tabela2_professor["origem"] = "Professor"
colunas1 = ["ID", "C", "h", "t", "L", "N", "Ltot", "M", "origem"]
colunas2 = ["ID", "C", "F", "delta", "origem"]
tabela1_original = pd.concat([tabela1_testes[colunas1], tabela1_professor[colunas1]], ignore_index=True)
tabela2_original = pd.concat([tabela2_testes[colunas2], tabela2_professor[colunas2]], ignore_index=True)
tabela1_original["ID"] = tabela1_original["ID"].astype(int)
tabela2_original["ID"] = tabela2_original["ID"].astype(int)

# Colunas calculadas em cada medição
tabela2_original["rigidez"] = tabela2_original["F"] / tabela2_original["delta"]        # K = F/δ (kgf/mm)
tabela2_original["deformacao"] = tabela2_original["delta"] / tabela2_original["C"] * 100  # δ/C × 100 (%)

# A maior carga de cada mola é a medição até o escoamento
maior_carga = tabela2_original.groupby("ID")["F"].transform("max")
tabela2_original["escoamento"] = tabela2_original["F"] == maior_carga

# ------------------------------------------------------------
# Processamento dos dados
# ------------------------------------------------------------
# a) Resultados de cada mola
resultados = []
for id_mola in tabela1_original["ID"]:
    medidas = tabela2_original[tabela2_original["ID"] == id_mola]
    menores = medidas[~medidas["escoamento"]]
    esc = medidas[medidas["escoamento"]].iloc[0]

    # Rigidez K_reg: coeficiente angular da reta F × δ nas quatro cargas menores
    ajuste = stats.linregress(menores["delta"], menores["F"])

    # Mesma reta com as cinco cargas, só para comparar o efeito do ponto de escoamento
    ajuste_5 = stats.linregress(medidas["delta"], medidas["F"])

    # Rigidez secante média das quatro cargas menores e seu intervalo de confiança de 95%
    secante = menores["rigidez"].to_numpy()
    ic_inf, ic_sup = stats.t.interval(0.95, len(secante) - 1, loc=secante.mean(), scale=stats.sem(secante))

    resultados.append({
        "ID": id_mola,
        "K_reg": ajuste.slope,
        "intercepto": ajuste.intercept,
        "R2": ajuste.rvalue ** 2,
        "K_reg_5_cargas": ajuste_5.slope,
        "R2_5_cargas": ajuste_5.rvalue ** 2,
        "K_sec_media": secante.mean(),
        "K_sec_ic_inf": ic_inf,
        "K_sec_ic_sup": ic_sup,
        "F_esc": esc["F"],
        "delta_esc": esc["delta"],
        "def_esc": esc["deformacao"],
        "K_sec_esc": esc["rigidez"],
        "var_K_esc": (esc["rigidez"] / secante.mean() - 1) * 100,   # variação no escoamento (%)
    })
tabela1_df = tabela1_original.merge(pd.DataFrame(resultados), on="ID")
tabela1_dict = tabela1_df.to_dict('list')     # Converte para um dicionário de listas
tabela2_dict = tabela2_original.to_dict('list')

# b) Grupos analisados: todas as molas e cada planilha
grupos = {"Todas as molas": tabela1_df}
for origem in tabela1_df["origem"].unique():
    grupos[origem] = tabela1_df[tabela1_df["origem"] == origem]

# c) Estatística descritiva de K_reg
descritiva = []
for nome, dados in grupos.items():
    k = dados["K_reg"]
    ic_inf, ic_sup = stats.t.interval(0.95, len(k) - 1, loc=k.mean(), scale=stats.sem(k))
    descritiva.append({
        "grupo": nome, "n": len(k), "media": k.mean(), "media_N_mm": k.mean() * 9.80665,
        "desvio_padrao": k.std(), "CV_pct": k.std() / k.mean() * 100,
        "IC95_inf": ic_inf, "IC95_sup": ic_sup, "mediana": k.median(),
        "minimo": k.min(), "maximo": k.max(), "shapiro_p": stats.shapiro(k).pvalue,
    })
descritiva_df = pd.DataFrame(descritiva)

# d) Correlações entre geometria e K_reg (e entre deformação no escoamento e K_reg)
correlacoes = []
for nome, dados in grupos.items():
    for var in geo + ["def_esc"]:
        pearson = stats.pearsonr(dados[var], dados["K_reg"])
        spearman = stats.spearmanr(dados[var], dados["K_reg"])
        log_log = stats.pearsonr(np.log(dados[var]), np.log(dados["K_reg"]))
        correlacoes.append({
            "grupo": nome, "n": len(dados), "variavel": var,
            "pearson_r": pearson[0], "pearson_p": pearson[1],
            "spearman_rho": spearman[0], "spearman_p": spearman[1],
            "log_log_r": log_log[0],
        })
correlacoes_df = pd.DataFrame(correlacoes)

# e) Comparação da rigidez entre as duas origens
comparacao = stats.mannwhitneyu(grupos["Testes próprios"]["K_reg"], grupos["Professor"]["K_reg"])

# Salva as tabelas de resultados (formato que o Excel em português abre direto)
tabela1_df.to_csv("resumo_por_mola.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
correlacoes_df.to_csv("correlacoes.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
descritiva_df.to_csv("estatistica_descritiva.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------
# Apresentação dos resultados
# ------------------------------------------------------------
print("\nApresentação do dicionário lido - tabela1: ")
print(tabela1_dict)
print("\nApresentação do dicionário lido - tabela2: ")
print(tabela2_dict)

print("\nRigidez por mola, nas quatro cargas menores (kgf/mm):")
for _, mola in tabela1_df.iterrows():
    print(f"  ID {int(mola['ID'])} ({mola['origem']}): K_reg = {mola['K_reg']:.4f}  R² = {mola['R2']:.4f}  "
          f"K_sec = {mola['K_sec_media']:.4f}  IC95% [{mola['K_sec_ic_inf']:.4f}; {mola['K_sec_ic_sup']:.4f}]")

print("\nEscoamento por mola:")
for _, mola in tabela1_df.iterrows():
    print(f"  ID {int(mola['ID'])}: F = {mola['F_esc']:.3f} kgf  δ = {mola['delta_esc']:.2f} mm  "
          f"δ/C = {mola['def_esc']:.1f}%  variação de F/δ = {mola['var_K_esc']:+.1f}%")

print("\nEstatística descritiva de K_reg:")
print(descritiva_df.round(4).to_string(index=False))

print("\nCorrelações com K_reg:")
print(correlacoes_df.round(3).to_string(index=False))

print(f"\nK_reg, testes próprios x professor (Mann-Whitney): p = {comparacao.pvalue:.3f}")
print(f"Nível de significância com correção de Bonferroni (7 dimensões): {0.05 / 7:.4f}")

# ------------------------------------------------------------
# Gráficos
# ------------------------------------------------------------
# Aparência comum: as duas origens têm sempre a mesma cor e o mesmo marcador
CORES = {"Testes próprios": "#2a78d6", "Professor": "#eb6834"}
MARCADORES = {"Testes próprios": "o", "Professor": "s"}
CINZA = "#898781"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.titlesize": 11, "axes.titlelocation": "left",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.labelcolor": "#52514e",
    "savefig.dpi": 150, "savefig.bbox": "tight",
})


def virgula(valor, casas=2):
    """Escreve um número com vírgula decimal."""
    return f"{valor:.{casas}f}".replace(".", ",")


formato_virgula = FuncFormatter(lambda valor, _: f"{valor:g}".replace(".", ","))


def eixos_com_virgula(eixo):
    eixo.xaxis.set_major_formatter(formato_virgula)
    eixo.yaxis.set_major_formatter(formato_virgula)


def correlacao_todas(var):
    """Pearson r e p de uma variável, com todas as molas."""
    return stats.pearsonr(tabela1_df[var], tabela1_df["K_reg"])


def dispersao(eixo, var, y="K_reg"):
    for origem, dados in grupos.items():
        if origem != "Todas as molas":
            eixo.scatter(dados[var], dados[y], s=45, color=CORES[origem], marker=MARCADORES[origem],
                         edgecolor="white", linewidth=0.8, label=origem)
    eixos_com_virgula(eixo)


# Dispersão da rigidez em função da altura t (modelo do professor)
plt.figure()
dispersao(plt.gca(), "t")
r, p = correlacao_todas("t")
plt.title(f"Rigidez tende a aumentar com a altura (r = {virgula(r)}; p = {virgula(p, 3)})")
plt.xlabel("Altura da mola t (mm)")
plt.ylabel("Rigidez K_reg (kgf/mm)")
plt.legend(frameon=False)
plt.savefig("grafico_K_x_t.png")

# Dispersão da rigidez em função da espessura h
plt.figure()
dispersao(plt.gca(), "h")
r, p = correlacao_todas("h")
plt.title(f"Espessura não tem correlação significativa com a rigidez (r = {virgula(r)}; p = {virgula(p, 3)})")
plt.xlabel("Espessura h (mm)")
plt.ylabel("Rigidez K_reg (kgf/mm)")
plt.legend(frameon=False)
plt.savefig("grafico_K_x_h.png")

# Dispersão da deformação relativa no escoamento em função do passo L
plt.figure()
dispersao(plt.gca(), "L", "def_esc")
r, p = stats.pearsonr(tabela1_df["L"], tabela1_df["def_esc"])
plt.title(f"Deformação no escoamento e passo (r = {virgula(r)}; p = {virgula(p, 3)})")
plt.xlabel("Passo L (mm)")
plt.ylabel("Deformação relativa no escoamento δ/C (%)")
plt.legend(frameon=False)
plt.savefig("grafico_deformacao_x_L.png")

# Rigidez secante por mola: cargas menores e maior carga (escoamento)
ordem = tabela1_df.sort_values("K_reg", ascending=False)["ID"].tolist()
plt.figure(figsize=(10, 4.2))
eixo = plt.gca()
for posicao, id_mola in enumerate(ordem):
    medidas = tabela2_original[tabela2_original["ID"] == id_mola]
    origem = medidas["origem"].iloc[0]
    pontos = medidas[~medidas["escoamento"]]
    eixo.scatter([posicao] * len(pontos), pontos["rigidez"], s=22, color=CORES[origem],
                 marker=MARCADORES[origem], alpha=0.8, zorder=3)
    eixo.scatter(posicao, medidas[medidas["escoamento"]]["rigidez"], s=48, marker="X", color="#0b0b0b",
                 edgecolor="white", linewidth=0.6, zorder=4)
mola_58 = tabela1_df[tabela1_df["ID"] == 58].iloc[0]
eixo.annotate(f"ID 58: {virgula(mola_58['var_K_esc'], 0)}% acima das cargas menores",
              (ordem.index(58), mola_58["K_sec_esc"]), xytext=(10, 6), textcoords="offset points",
              fontsize=9, color="#0b0b0b")
eixo.set_yscale("log")
eixo.yaxis.set_major_formatter(formato_virgula)
eixo.set_xticks(range(len(ordem)), [str(i) for i in ordem])
eixo.set_title("Na maior carga, F/δ muda pouco na maioria das molas; a ID 58 é a exceção")
eixo.set_xlabel("ID da mola (da maior para a menor rigidez)")
eixo.set_ylabel("Rigidez secante F/δ (kgf/mm, escala log)")
for origem in CORES:
    eixo.scatter([], [], s=22, color=CORES[origem], marker=MARCADORES[origem], label=f"Cargas menores: {origem.lower()}")
eixo.scatter([], [], s=48, marker="X", color="#0b0b0b", label="Maior carga (escoamento)")
eixo.legend(frameon=False, loc="upper right")
plt.savefig("grafico_rigidez_secante.png")

# Deformação relativa no escoamento por mola
plt.figure(figsize=(10, 3.6))
eixo = plt.gca()
ordenado = tabela1_df.sort_values("def_esc", ascending=False)
posicoes = range(len(ordenado))
eixo.bar(posicoes, ordenado["def_esc"], width=0.55, color=[CORES[o] for o in ordenado["origem"]])
eixo.set_xticks(list(posicoes), [str(i) for i in ordenado["ID"]])
eixo.grid(axis="x", visible=False)
eixo.yaxis.set_major_formatter(formato_virgula)
eixo.set_title(f"Deformação no escoamento variou de {virgula(ordenado['def_esc'].min(), 0)}% a "
               f"{virgula(ordenado['def_esc'].max(), 0)}%")
eixo.set_xlabel("ID da mola")
eixo.set_ylabel("δ/C no escoamento (%)")
for origem, cor in CORES.items():
    eixo.bar(0, 0, color=cor, label=origem)
eixo.legend(frameon=False)
plt.savefig("grafico_deformacao_escoamento.png")

# Carga × alongamento: reta das quatro cargas menores e ponto de escoamento, um painel por origem
figura, paineis = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
for eixo, origem in zip(paineis, ["Testes próprios", "Professor"]):
    for _, mola in grupos[origem].iterrows():
        medidas = tabela2_original[tabela2_original["ID"] == mola["ID"]]
        pontos = medidas[~medidas["escoamento"]]
        destaque = mola["ID"] == 58
        cor = "#0b0b0b" if destaque else CORES[origem]
        x_reta = np.array([0, mola["delta_esc"]])
        eixo.plot(x_reta, mola["intercepto"] + mola["K_reg"] * x_reta, color=cor, linewidth=1.1, alpha=0.75)
        eixo.scatter(pontos["delta"], pontos["F"], s=18, color=cor, marker=MARCADORES[origem], zorder=3)
        eixo.scatter(mola["delta_esc"], mola["F_esc"], s=48, marker="X", color=cor, edgecolor="white",
                     linewidth=0.6, zorder=4)
        if destaque or mola["F_esc"] >= 3:
            eixo.annotate(f"ID {int(mola['ID'])}", (mola["delta_esc"], mola["F_esc"]), xytext=(7, 0),
                          textcoords="offset points", va="center", fontsize=9, color="#0b0b0b")
    eixo.set_title(f"{origem} (n = {len(grupos[origem])})")
    eixo.set_xlabel("Alongamento δ (mm)")
    eixos_com_virgula(eixo)
paineis[0].set_ylabel("Carga F (kgf)")
paineis[0].scatter([], [], s=18, color=CINZA, label="Cargas menores")
paineis[0].scatter([], [], s=48, marker="X", color=CINZA, label="Maior carga (escoamento)")
paineis[0].plot([], [], color=CINZA, linewidth=1.1, label="Reta das cargas menores")
paineis[0].legend(frameon=False, loc="upper right")
figura.tight_layout()
figura.savefig("grafico_carga_alongamento.png")

# Rigidez contra quatro dimensões
figura, paineis = plt.subplots(2, 2, figsize=(10, 6.6), sharey=True)
nomes = {"t": "Altura t (mm)", "L": "Passo L (mm)", "N": "Espiras N", "Ltot": "Largura Ltot (mm)"}
for eixo, (var, rotulo) in zip(paineis.flat, nomes.items()):
    dispersao(eixo, var)
    r, p = correlacao_todas(var)
    eixo.set_title(f"r = {virgula(r)}; p = {virgula(p, 3)}")
    eixo.set_xlabel(rotulo)
    if var == "N":
        eixo.set_xticks([3, 4, 5])
paineis[0, 0].set_ylabel("Rigidez K_reg (kgf/mm)")
paineis[1, 0].set_ylabel("Rigidez K_reg (kgf/mm)")
paineis[0, 0].legend(frameon=False)
figura.tight_layout()
figura.savefig("grafico_K_x_geometria.png")

# Matriz de correlação de Pearson
plt.figure(figsize=(7.4, 6.2))
matriz = tabela1_df[geo + ["def_esc", "K_reg"]].corr()
nomes_matriz = geo + ["δ/C esc.", "K_reg"]
anotacoes = matriz.map(lambda valor: virgula(valor))
sb.heatmap(matriz, annot=anotacoes, fmt="", cmap="RdBu_r", vmin=-1, vmax=1, square=True, linewidths=1.5,
           linecolor="white", xticklabels=nomes_matriz, yticklabels=nomes_matriz,
           cbar_kws={"shrink": 0.8, "format": formato_virgula})
plt.grid(False)
plt.title(f"Matriz de correlação de Pearson (n = {len(tabela1_df)} molas)")
plt.savefig("grafico_correlacao.png")

# Matriz de dispersão
grafico = sb.pairplot(tabela1_df, vars=["h", "t", "L", "N", "Ltot", "K_reg"], hue="origem", palette=CORES,
                      markers=MARCADORES, height=1.45, diag_kind="hist", plot_kws={"s": 28, "edgecolor": "white"})
for eixo in grafico.axes.flat:
    eixo.xaxis.set_major_formatter(formato_virgula)
    eixo.yaxis.set_major_formatter(formato_virgula)
    eixo.tick_params(labelsize=8)
    eixo.locator_params(nbins=4)
grafico.savefig("grafico_matriz_dispersao.png")

plt.show()
