# ============================================================
# Análise de molas sanfonadas de PLA impressas em 3D
# Ensaio de tração em máquina, referência ASTM D638
#
# Fluxo (modelo do professor):
#   Leitura (Excel) -> DataFrame -> Converte p/ DICT -> análises -> gráficos
#
# Definições (as mesmas das planilhas do professor e da turma):
#   Rigidez K      = média de F/δ das 5 cargas de cada mola (kgf/mm)
#   Ductilidade D  = δ/C na 5ª carga (a mais alta), em %
# Nenhuma carga e nenhuma mola é removida.
# ============================================================

# Carregamento das bibliotecas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.ticker import FuncFormatter
from scipy import stats

# OBS: para instalar as bibliotecas: pip install -r requirements.txt

# Pasta do programa (planilhas e resultados ficam aqui)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------
# 1. Leitura (Excel) -> DataFrame
# ------------------------------------------------------------
arquivo_turma = "Planilha_Python_CIDA_2.xlsx"      # planilha da turma
arquivo_professor = "experimento.xlsx"             # planilha do professor

# --- Turma ---
# Página1: uma linha por mola; K e D já calculados na planilha (usados só na conferência)
tabela1_turma = pd.read_excel(arquivo_turma, sheet_name="Página1", header=0, usecols="A:L")
tabela1_turma = tabela1_turma.rename(columns={"n": "N", "m": "M", "K": "K_planilha", "D": "D_planilha"})

# Página2: as 5 cargas de cada mola; as colunas de fórmula K e L não são lidas
tabela2_turma = pd.read_excel(arquivo_turma, sheet_name="Página2", header=0, usecols="A:J")
tabela2_turma.columns = ["ID", "C", "h", "t", "L", "N", "Ltot", "M", "F", "delta"]

# A coluna ID só está preenchida em uma linha de cada mola.
# Cada linha recebe o ID da mola, identificada pela geometria (mesmas 7 medidas).
geo = ["C", "h", "t", "L", "N", "Ltot", "M"]
tabela2_turma["grupo"] = tabela2_turma.groupby(geo, sort=False).ngroup()
tabela2_turma["ID"] = tabela2_turma["grupo"].map(
    tabela2_turma.dropna(subset=["ID"]).groupby("grupo")["ID"].first())

# --- Professor ---
# tabela1: uma linha por mola (dados a partir de B2); K e D da planilha só para conferência.
# Na planilha do professor, D está em fração (δ/C): multiplica por 100 para ficar em %.
tabela1_professor = pd.read_excel(arquivo_professor, sheet_name="tabela1", header=1, usecols="B:K")
tabela1_professor = tabela1_professor.rename(columns={"n": "N", "m": "M", "K": "K_planilha", "D": "D_planilha"})
tabela1_professor["D_planilha"] = tabela1_professor["D_planilha"] * 100

# tabela2: as 5 cargas de cada mola; as colunas de fórmula rigidez e dutilidade não são lidas
tabela2_professor = pd.read_excel(arquivo_professor, sheet_name="tabela2", header=1, usecols="B:D")
tabela2_professor.columns = ["ID", "F", "delta"]

# --- Junção das duas planilhas, com a coluna origem ---
tabela1_turma["origem"] = "Turma"
tabela1_professor["origem"] = "Professor"
tabela2_turma["origem"] = "Turma"
tabela2_professor["origem"] = "Professor"
colunas1 = ["ID", "origem"] + geo + ["K_planilha", "D_planilha"]
colunas2 = ["ID", "origem", "F", "delta"]
tabela1_df = pd.concat([tabela1_turma[colunas1], tabela1_professor[colunas1]], ignore_index=True)
tabela2_df = pd.concat([tabela2_turma[colunas2], tabela2_professor[colunas2]], ignore_index=True)
tabela1_df["ID"] = tabela1_df["ID"].astype(int)
tabela2_df["ID"] = tabela2_df["ID"].astype(int)

# Observações registradas sobre molas específicas (nada é removido por causa delas)
OBSERVACOES = {
    58: "5ª carga: ruptura relatada; F/δ desse ponto é 2,5x o das 4 cargas anteriores. "
        "Conferir a força no registro da máquina em δ = 28,6 mm",
}

# ------------------------------------------------------------
# 2. Converte p/ DICT
# ------------------------------------------------------------
tabela1_dict = tabela1_df.to_dict("list")     # uma lista por coluna
tabela2_dict = tabela2_df.to_dict("list")

# ------------------------------------------------------------
# 3. Análises (a partir dos dicionários)
# ------------------------------------------------------------
# a) F/δ de cada medição
tabela2_dict["rigidez"] = [f / d for f, d in zip(tabela2_dict["F"], tabela2_dict["delta"])]

# b) K e D de cada mola, com o IC 95% da média das 5 razões F/δ
calculo = {"ID": [], "K": [], "K_dp": [], "K_IC95_inf": [], "K_IC95_sup": [],
           "F_5": [], "delta_5": [], "D": [], "observacao": []}
for i, id_mola in enumerate(tabela1_dict["ID"]):
    linhas = [j for j, x in enumerate(tabela2_dict["ID"]) if x == id_mola]
    if len(linhas) != 5:
        raise ValueError(f"A mola ID {id_mola} tem {len(linhas)} cargas, e não 5.")
    cargas = sorted((tabela2_dict["F"][j], tabela2_dict["delta"][j]) for j in linhas)  # da menor para a maior
    secantes = np.array([f / d for f, d in cargas])
    k = secantes.mean()
    dp = secantes.std(ddof=1)
    ic_inf, ic_sup = stats.t.interval(0.95, len(secantes) - 1, loc=k, scale=dp / np.sqrt(len(secantes)))
    f5, d5 = cargas[-1]                                   # 5ª carga, a mais alta
    calculo["ID"].append(id_mola)
    calculo["K"].append(k)
    calculo["K_dp"].append(dp)
    calculo["K_IC95_inf"].append(ic_inf)
    calculo["K_IC95_sup"].append(ic_sup)
    calculo["F_5"].append(f5)
    calculo["delta_5"].append(d5)
    calculo["D"].append(d5 / tabela1_dict["C"][i] * 100)
    calculo["observacao"].append(OBSERVACOES.get(id_mola, ""))

# DataFrame de resultados, só para salvar e desenhar (as contas acima usam os dicionários)
resultado = tabela1_df.merge(pd.DataFrame(calculo), on="ID")

# c) Conferência com os valores já calculados nas planilhas, com 4 casas decimais
resultado["K_confere"] = resultado["K"].round(4) == resultado["K_planilha"].round(4)
resultado["D_confere"] = resultado["D"].round(4) == resultado["D_planilha"].round(4)
# O professor arredondou D (em fração) para 3 casas: em %, a precisão dele é de 0,1 ponto
eh_prof = resultado["origem"] == "Professor"
resultado.loc[eh_prof, "D_confere"] = (resultado.loc[eh_prof, "D"] - resultado.loc[eh_prof, "D_planilha"]).abs() <= 0.05
# O professor arredondou K para 5 casas: compara com tolerância de meia unidade da 5ª casa
resultado.loc[eh_prof, "K_confere"] = (resultado.loc[eh_prof, "K"] - resultado.loc[eh_prof, "K_planilha"]).abs() <= 0.5e-5 + 1e-12

# d) Grupos analisados: todas as molas e cada origem
grupos = {"Todas as molas": resultado}
for origem in ["Turma", "Professor"]:
    grupos[origem] = resultado[resultado["origem"] == origem]

# e) Estatística descritiva de K e D
descritiva = []
for nome, dados in grupos.items():
    for var in ["K", "D"]:
        x = dados[var]
        ic_inf, ic_sup = stats.t.interval(0.95, len(x) - 1, loc=x.mean(), scale=stats.sem(x))
        descritiva.append({
            "grupo": nome, "variavel": var, "n": len(x), "media": x.mean(), "desvio_padrao": x.std(),
            "CV_pct": x.std() / x.mean() * 100, "IC95_inf": ic_inf, "IC95_sup": ic_sup,
            "mediana": x.median(), "minimo": x.min(), "maximo": x.max(), "shapiro_p": stats.shapiro(x).pvalue,
        })
descritiva_df = pd.DataFrame(descritiva)


# f) Teste de outlier nas 17 molas: Grubbs (α = 0,05) e critério do boxplot (IQR)
def grubbs_iterativo(valores, ids, alfa=0.05):
    """Grubbs bilateral, repetido: retira o valor detectado e testa de novo, até não detectar.
    A retirada vale só dentro do teste; nenhum dado é apagado da análise."""
    valores, ids = list(valores), list(ids)
    etapas = []
    while len(valores) >= 3:
        x = np.array(valores)
        n = len(x)
        i = int(np.argmax(np.abs(x - x.mean())))
        g = abs(x[i] - x.mean()) / x.std(ddof=1)
        t = stats.t.ppf(1 - alfa / (2 * n), n - 2)
        g_critico = (n - 1) / np.sqrt(n) * np.sqrt(t ** 2 / (n - 2 + t ** 2))
        etapas.append({"etapa": len(etapas) + 1, "n": n, "ID": ids[i], "valor": x[i],
                       "estatistica": g, "limite": g_critico, "outlier": g > g_critico})
        if g <= g_critico:
            break
        valores.pop(i)
        ids.pop(i)
    return etapas


outliers = []
limites_iqr = {}
for var in ["K", "D"]:
    for etapa in grubbs_iterativo(resultado[var], resultado["ID"]):
        outliers.append({"variavel": var, "metodo": "Grubbs (α = 0,05)", **etapa})
    q1, q3 = resultado[var].quantile([0.25, 0.75])
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    limites_iqr[var] = (q1, q3, lim_inf, lim_sup)
    for _, mola in resultado.iterrows():
        if mola[var] < lim_inf or mola[var] > lim_sup:
            outliers.append({"variavel": var, "metodo": "Boxplot (1,5 × IQR)", "etapa": None,
                             "n": len(resultado), "ID": mola["ID"], "valor": mola[var],
                             "estatistica": None, "limite": lim_sup if mola[var] > lim_sup else lim_inf,
                             "outlier": True})
outliers_df = pd.DataFrame(outliers)
ids_outlier = {var: sorted(outliers_df[(outliers_df["variavel"] == var) & outliers_df["outlier"]]["ID"].unique())
               for var in ["K", "D"]}

# g) Correlação de Pearson entre geometria, K e D
correlacoes = []
for alvo in ["K", "D"]:
    for var in geo:
        r, p = stats.pearsonr(resultado[var], resultado[alvo])
        rho, p_s = stats.spearmanr(resultado[var], resultado[alvo])
        correlacoes.append({"alvo": alvo, "variavel": var, "n": len(resultado), "pearson_r": r, "pearson_p": p,
                            "spearman_rho": rho, "spearman_p": p_s})
correlacoes_df = pd.DataFrame(correlacoes)

# Salva as tabelas (separador ; e vírgula decimal, que o Excel em português abre direto)
salvar = dict(sep=";", decimal=",", index=False, encoding="utf-8-sig")
colunas_resumo = ["ID", "origem"] + geo + ["K", "K_dp", "K_IC95_inf", "K_IC95_sup", "F_5", "delta_5", "D",
                                           "K_planilha", "D_planilha", "K_confere", "D_confere", "observacao"]
resultado[colunas_resumo].round({"K": 4, "D": 4}).to_csv("resumo_por_mola.csv", **salvar)
descritiva_df.to_csv("estatistica_descritiva.csv", **salvar)
outliers_df.to_csv("outliers.csv", **salvar)
correlacoes_df.to_csv("correlacoes.csv", **salvar)

# ------------------------------------------------------------
# 4. Apresentação dos resultados
# ------------------------------------------------------------
print("\nApresentação do dicionário lido - tabela1: ")
print(tabela1_dict)
print("\nApresentação do dicionário lido - tabela2: ")
print(tabela2_dict)

print("\nK e D por mola (4 casas decimais) e conferência com as planilhas:")
for _, mola in resultado.iterrows():
    print(f"  ID {mola['ID']:>3} ({mola['origem']:<9}) K = {mola['K']:.4f} kgf/mm "
          f"[planilha {mola['K_planilha']:.4f}: {'confere' if mola['K_confere'] else 'NÃO CONFERE'}]   "
          f"D = {mola['D']:.4f}% [planilha {mola['D_planilha']:.4f}: "
          f"{'confere' if mola['D_confere'] else 'NÃO CONFERE'}]")
    if mola["observacao"]:
        print(f"      Obs.: {mola['observacao']}")
if not (resultado["K_confere"].all() and resultado["D_confere"].all()):
    print("\n  ATENÇÃO: há valores que não conferem com as planilhas.")

print("\nEstatística descritiva:")
print(descritiva_df.round(4).to_string(index=False))

print("\nTeste de outlier (nenhuma mola é removida; só registro):")
print(outliers_df.round(4).to_string(index=False))
for var, (q1, q3, lim_inf, lim_sup) in limites_iqr.items():
    print(f"  {var}: Q1 = {q1:.4f}  Q3 = {q3:.4f}  limites do boxplot = [{lim_inf:.4f}; {lim_sup:.4f}]")
for var, ids in ids_outlier.items():
    print(f"  {var}: molas apontadas como outlier = {ids if ids else 'nenhuma'}")

print("\nCorrelações com K e D:")
print(correlacoes_df.round(4).to_string(index=False))

# ------------------------------------------------------------
# 5. Gráficos (17 molas)
# ------------------------------------------------------------
CORES = {"Turma": "#2a78d6", "Professor": "#eb6834"}
MARCADORES = {"Turma": "o", "Professor": "s"}
PRETO = "#0b0b0b"
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


def dispersao(var_x, var_y, rotulo_x, rotulo_y, arquivo):
    """Dispersão com as duas origens, Pearson no título e os outliers de var_y identificados."""
    plt.figure()
    eixo = plt.gca()
    for origem in ["Turma", "Professor"]:
        dados = grupos[origem]
        eixo.scatter(dados[var_x], dados[var_y], s=45, color=CORES[origem], marker=MARCADORES[origem],
                     edgecolor="white", linewidth=0.8, label=origem)
    for _, mola in resultado[resultado["ID"].isin(ids_outlier[var_y])].iterrows():
        eixo.annotate(f"ID {mola['ID']}", (mola[var_x], mola[var_y]), xytext=(6, 0), textcoords="offset points",
                      va="center", fontsize=9, color=PRETO)
    r, p = stats.pearsonr(resultado[var_x], resultado[var_y])
    eixo.set_title(f"{var_y} × {var_x}, 17 molas (Pearson r = {virgula(r)}; p = {virgula(p, 3)})")
    eixo.set_xlabel(rotulo_x)
    eixo.set_ylabel(rotulo_y)
    eixo.xaxis.set_major_formatter(formato_virgula)
    eixo.yaxis.set_major_formatter(formato_virgula)
    eixo.legend(frameon=False)
    plt.savefig(arquivo)


# Dispersões do modelo do professor
dispersao("h", "K", "Espessura h (mm)", "Rigidez K (kgf/mm)", "grafico_K_x_h.png")
dispersao("t", "K", "Altura t (mm)", "Rigidez K (kgf/mm)", "grafico_K_x_t.png")
dispersao("L", "D", "Passo L (mm)", "Ductilidade D (%)", "grafico_D_x_L.png")

# Boxplot de K por mola: as 5 razões F/δ de cada mola, da mais rígida para a menos rígida
ordem = resultado.sort_values("K", ascending=False)
medidas = pd.DataFrame(tabela2_dict)
plt.figure(figsize=(10, 4.4))
eixo = plt.gca()
caixas = eixo.boxplot([medidas[medidas["ID"] == i]["rigidez"] for i in ordem["ID"]], widths=0.55,
                      patch_artist=True, medianprops={"color": PRETO, "linewidth": 1.2},
                      flierprops={"marker": "o", "markersize": 4, "markerfacecolor": PRETO, "markeredgecolor": PRETO})
for caixa, origem in zip(caixas["boxes"], ordem["origem"]):
    caixa.set_facecolor(CORES[origem])
    caixa.set_alpha(0.75)
    caixa.set_edgecolor(CORES[origem])
eixo.scatter(range(1, len(ordem) + 1), ordem["K"], marker="D", s=16, color="white", edgecolor=PRETO,
             linewidth=0.8, zorder=4, label="K (média das 5 cargas)")
eixo.set_yscale("log")
eixo.yaxis.set_major_formatter(formato_virgula)
eixo.set_xticks(range(1, len(ordem) + 1), [str(i) for i in ordem["ID"]])
eixo.grid(axis="x", visible=False)
eixo.set_title("F/δ das 5 cargas de cada mola")
eixo.set_xlabel("ID da mola (da maior para a menor rigidez)")
eixo.set_ylabel("F/δ (kgf/mm, escala log)")
for origem, cor in CORES.items():
    eixo.bar(0, 0, color=cor, alpha=0.75, label=origem)
eixo.set_xlim(0.4, len(ordem) + 0.6)
eixo.legend(frameon=False, loc="upper right")
plt.savefig("grafico_boxplot_K_por_mola.png")

# Boxplot de K e de D nas 17 molas, com os limites do critério IQR e os outliers identificados
figura, paineis = plt.subplots(1, 2, figsize=(9, 4.4))
for eixo, var, rotulo in zip(paineis, ["K", "D"], ["Rigidez K (kgf/mm)", "Ductilidade D (%)"]):
    eixo.boxplot(resultado[var], widths=0.4, showfliers=False, medianprops={"color": PRETO},
                 boxprops={"color": "#52514e"}, whiskerprops={"color": "#52514e"}, capprops={"color": "#52514e"})
    deslocamento = np.linspace(-0.12, 0.12, len(resultado))
    for (_, mola), dx in zip(resultado.sort_values(var).iterrows(), deslocamento):
        eixo.scatter(1 + dx, mola[var], s=30, color=CORES[mola["origem"]], marker=MARCADORES[mola["origem"]],
                     edgecolor="white", linewidth=0.6, zorder=3)
        if mola["ID"] in ids_outlier[var]:
            eixo.annotate(f"ID {mola['ID']}", (1 + dx, mola[var]), xytext=(8, 0), textcoords="offset points",
                          va="center", fontsize=9, color=PRETO)
    lim_sup = limites_iqr[var][3]
    if lim_sup < resultado[var].max() * 1.05:
        eixo.axhline(lim_sup, color="#898781", linewidth=0.9, linestyle="--")
        eixo.annotate("limite do boxplot (Q3 + 1,5·IQR)", (1.5, lim_sup), xytext=(0, 4), textcoords="offset points",
                      ha="right", fontsize=8, color="#52514e")
    eixo.set_xticks([])
    eixo.grid(axis="x", visible=False)
    eixo.set_ylabel(rotulo)
    eixo.yaxis.set_major_formatter(formato_virgula)
    ids = ids_outlier[var]
    eixo.set_title(f"{var}: outlier em " + ", ".join(f"ID {i}" for i in ids) if ids else f"{var}: nenhum outlier")
for origem in CORES:
    paineis[0].scatter([], [], s=30, color=CORES[origem], marker=MARCADORES[origem], label=origem)
paineis[0].legend(frameon=False, loc="upper left")
figura.tight_layout()
figura.savefig("grafico_boxplot_K_D.png")

# Matriz de correlação de Pearson
plt.figure(figsize=(7.4, 6.2))
matriz = resultado[geo + ["K", "D"]].corr()
sb.heatmap(matriz, annot=matriz.map(virgula), fmt="", cmap="RdBu_r", vmin=-1, vmax=1, square=True,
           linewidths=1.5, linecolor="white", cbar_kws={"shrink": 0.8, "format": formato_virgula})
plt.grid(False)
plt.title(f"Matriz de correlação de Pearson (n = {len(resultado)} molas)")
plt.savefig("grafico_correlacao.png")

# Matriz de dispersão
grafico = sb.pairplot(resultado, vars=["h", "t", "L", "N", "Ltot", "K", "D"], hue="origem", palette=CORES,
                      markers=MARCADORES, height=1.35, diag_kind="hist", plot_kws={"s": 26, "edgecolor": "white"})
for eixo in grafico.axes.flat:
    eixo.xaxis.set_major_formatter(formato_virgula)
    eixo.yaxis.set_major_formatter(formato_virgula)
    eixo.tick_params(labelsize=8)
    eixo.locator_params(nbins=4)
grafico.savefig("grafico_matriz_dispersao.png")

plt.show()
