from scipy.stats import norm

def calcular_sigma(unidades: int, oportunidades_por_unidade: int, defeitos: int) -> float:
    """
    Calcula o nível sigma de um processo com base em unidades, oportunidades e defeitos.
    Fórmula padrão: DPMO -> conversão Sigma (com shift de 1,5).
    """
    oportunidades_totais = unidades * oportunidades_por_unidade
    dpmo = (defeitos / oportunidades_totais) * 1_000_000
    
    yield_rate = 1 - (dpmo / 1_000_000)
    z = norm.ppf(yield_rate)  # valor z da normal
    sigma = z + 1.5           # ajuste Six Sigma tradicional
    

    return round(sigma, 2)
