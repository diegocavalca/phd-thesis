import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

class Janelas:
    def __init__(self, base=None, id_residencia=None, inicio_intervalo=None,
                 fim_intervalo=None, debug=True):
        """
        Args:
            - base (nilmtk.dataset.DataSet): base de dados que será utilizada
                no processaento das janelas;
            - id_residencia (int): indice (1..N) da residencia que terão suas
                medições processadas em janelas (para consultar a lista de
                indices disponíveis basta acessar `BASE.buildings`;
            - inicio_intervalo (str): data / hora inicial da faixa de análise
                dos dados (padrao `dd/mm/yyyy h:m:s`);
            - fim_intervalo (str): data / hora final da faixa de análise dos
                dados (padrao `dd/mm/yyyy h:m:s`);
            - debug (bool): controle de verbosidade.
        """
        self.base = base
        self.id_residencia = id_residencia
        self.inicio_intervalo = inicio_intervalo
        self.fim_intervalo = fim_intervalo
        self.debug = debug

        # Definir o intervalo de dados considerado
        self.base.set_window(
            start=self.inicio_intervalo,
            end=self.fim_intervalo
        )

        # Selecionar a residencia
        self.residencia = self.base.buildings[self.id_residencia]

        # Lista de dicionarios com informacoes de janelas por carga
        self.dados_carga = []

    def preparar(self, taxa_amostral=3, intervalo_medicao=90):

        # Lista de dicionarios com informacoes de janelas por carga
        dados_cargas = []

        #tamanho_janela = int(intervalo_medicao / taxa_amostral) # Desuso: assumia janela em funcao da taxa
        tamanho_janela = intervalo_medicao

        # # Calculando tamanho máximo da série (padding, dependendo tamanho janeka)
        # series = self.residencia.elec.mains() \
            # .power_series_all_data(sample_period=taxa_amostral)
        # limite_serie = int(len(series.values) / tamanho_janela) * tamanho_janela

        # Gerar janelas para cada canal/aparelho
        if self.debug: print("* Gerar janelas para cada canal/aparelho...")

        inicio_intervalo =  datetime.strptime(self.inicio_intervalo, '%Y-%m-%d %H:%M:%S').date()
        fim_intervalo = datetime.strptime(self.fim_intervalo, '%Y-%m-%d %H:%M:%S').date()
        # for e_i in tqdm_notebook(range(1, len(self.residencia.elec.all_meters())+2)):
        for e_i in range(1, len(self.residencia.elec.all_meters())):
        # for e in self.residencia.elec.all_meters():

            # Selecionando canal/aparelho
            e = self.residencia.elec[e_i]

            # Normalizar nome aparelho/canal de medicao
            aparelho = e.label().lower().replace(" ", "_")

            try:

                # # Extraindo medicoes de energia da carga
                # power = np.array(e.power_series_all_data(sample_period=taxa_amostral).values[:limite_serie])
                
                # Extraindo medicoes de energia da carga (toda a serie)
                power = e.power_series_all_data(sample_period=taxa_amostral)

                # Verificar se a medicao da carga esta dentro do range de analise
                if power.index[0].date() >= inicio_intervalo and power.index[-1].date() <= fim_intervalo:
                
                    # Calculando tamanho máximo da série (padding, dependendo tamanho janeka)
                    limite_serie = int(len(power.values) / tamanho_janela) * tamanho_janela

                    # Garantindo limite da serie valido (caber dentro do reshape do tamanho janela)
                    while limite_serie % tamanho_janela != 0:
                        limite_serie -= 1
                    
                    # Encaixando medicao dentro do tamanho de janelas (p/ fazer reshape)
                    power = power.values[:limite_serie]

                    # Gerando máscara de status (ativo ou não), considerando ruido da carga
                    # ou rede na medição (threshod)
                    status = power > e.on_power_threshold()

                    # Dividindo em janelas (tanto energia, quanto estados)
                    windows_series = power.reshape(-1, tamanho_janela)
                    windows_status = status.reshape(-1, tamanho_janela)

                    # Remover nan (por zero)
                    windows_series = np.nan_to_num(windows_series)

                    # Calcular rotulos a partir das janelas
                    # Podendo ser:
                    #   - `estado` (denotando carga ATIVA [1] ou INATIVA [0]);
                    #   - `total`(soma da janela);
                    #   - `media`;
                    rotulos = {
                        "total": np.sum(windows_series, axis=1),
                        "media": np.mean(windows_series, axis=1),
                        "estado": np.where(
                            np.sum(windows_status, axis=1) > 0, 1, 0
                        )  # Estado de cada janela, baseado na pré-avaliação da serie
                        # completa, considerando ruido
                    }

                    # Consolidar objeto da carga
                    dados_cargas.append({
                        "carga": aparelho,
                        "instancia": e.instance(),
                        "janelas": windows_series,
                        "rotulos": rotulos
                    })

                    if self.debug: print(f"{aparelho} -> {windows_series.shape}")

            except Exception as e:
                if self.debug: print(f"{aparelho}: erro ao extrair dados -> {str(e)}")

        self.dados_cargas = dados_cargas

        return self.dados_cargas

    def listar_registros(self):
        return [(d["instancia"], d["carga"]) for d in self.dados_cargas]

    def listar_cargas(self):
        return [(d["instancia"], d["carga"]) for d in self.dados_cargas if d["carga"] not in ['site_meter']]

    def listar_medidores(self):
        return [(d["instancia"], d["carga"]) for d in self.dados_cargas if d["carga"] in ['site_meter']]

    def filtrar(self, filtros=None):
        dados_filtrados = []
        for d in self.dados_cargas:
            for f in filtros:
                if d["instancia"] == f[0] and d["carga"] == f[1]:
                    dados_filtrados.append(d)
                    break
        return dados_filtrados
