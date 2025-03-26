import time
import pandas as pd
import dask.dataframe as dd
from sqlalchemy import create_engine
from helpers import DataPreparation
from dask.diagnostics import ProgressBar

# Lê os dados com pandas (pode ser mais seguro para SQLs complexos)
engine = create_engine("mysql://root:Ana.mysql.18@127.0.0.1/vitaldb_anesthesiaml")
df = pd.read_sql('SELECT * FROM vitaldb_preprocessed', con=engine)

vital_signs = df.columns[2:]

# Função adaptada para Dask (sem alterações internas)
def prepare_case(group, signs, time_window_before, time_window_after, target):
    caseid = group['caseid'].iloc[0]
    preparer = DataPreparation(caseid, signs, time_window_before, time_window_after, target)
    prepared_data = preparer.series_to_supervised(group)
    return prepared_data

# Cria o meta automaticamente usando um grupo de amostra
sample_caseid = df['caseid'].iloc[0]
sample_group = df[df['caseid'] == sample_caseid]
sample_result = prepare_case(sample_group, vital_signs, 5*60, 1, 'insp_sevo')
meta = sample_result.iloc[0:0]

# Converte o DataFrame para Dask com várias partitions
ddf = dd.from_pandas(df, npartitions=20)

# Função wrapper com os mesmos argumentos que prepare_case espera
def prepare_case_dask(group):
    return prepare_case(group, vital_signs, 5*60, 1, 'insp_sevo')

# Executa o processamento com barra de progresso
with ProgressBar():
    final_df = ddf.groupby('caseid').apply(prepare_case_dask, meta=meta).compute()

# (Opcional) Ordenar o resultado final
final_df = final_df.sort_values(by=['caseid', 't']).reset_index(drop=True)

print("Final dataframe shape:", final_df.shape)
