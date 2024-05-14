import pandas as pd
import os

# Definir o diretório dos arquivos
directory = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data_indian/IndianLabelsOriginal/'
file_path_a = os.path.join(directory, 'a. IDRiD_Disease Grading_Training Labels.csv')
file_path_b = os.path.join(directory, 'b. IDRiD_Disease Grading_Testing Labels.csv')

# Listar arquivos no diretório para verificação
files = os.listdir(directory)
print("Arquivos na pasta:", files)

# Verificar se os arquivos existem antes de tentar carregá-los
if not os.path.isfile(file_path_a):
    print(f"Erro: Arquivo {file_path_a} não encontrado.")
    raise FileNotFoundError(f"Arquivo {file_path_a} não encontrado.")
if not os.path.isfile(file_path_b):
    print(f"Erro: Arquivo {file_path_b} não encontrado.")
    raise FileNotFoundError(f"Arquivo {file_path_b} não encontrado.")

# Carregar os datasets
dfLabel_a = pd.read_csv(file_path_a)
dfLabel_b = pd.read_csv(file_path_b)

# Mostrar as colunas existentes para verificação
print("Colunas no dfLabel_a antes da remoção:", dfLabel_a.columns)
print("Colunas no dfLabel_b antes da remoção:", dfLabel_b.columns)

# Remover todas as colunas após "Retinopathy grade" em dfLabel_a
columns_to_keep_a = ['Image name', 'Retinopathy grade']
dfLabel_a = dfLabel_a.loc[:, columns_to_keep_a]

# Remover todas as colunas após "Retinopathy grade" em dfLabel_b
columns_to_keep_b = ['Image name', 'Retinopathy grade']
dfLabel_b = dfLabel_b.loc[:, columns_to_keep_b]

# Mostrar as colunas restantes após a remoção
print("Colunas no dfLabel_a após a remoção:", dfLabel_a.columns)
print("Colunas no dfLabel_b após a remoção:", dfLabel_b.columns)

# Renomear colunas
dfLabel_a = dfLabel_a.rename(columns={
    'Image name': 'id_code',
    'Retinopathy grade': 'diagnosis'
})

dfLabel_b = dfLabel_b.rename(columns={
    'Image name': 'id_code',
    'Retinopathy grade': 'diagnosis'
})

# Salvar os datasets modificados
output_directory = 'C:/Users/Bruno/Desktop/projeto_retinopatia/data'
os.makedirs(output_directory, exist_ok=True)

output_path_a = os.path.join(output_directory, 'modified_a_Train_Labels_Indian.csv')
output_path_b = os.path.join(output_directory, 'modified_b_Test_Labels_Indian.csv')

dfLabel_a.to_csv(output_path_a, index=False)
dfLabel_b.to_csv(output_path_b, index=False)

# Verificar se as colunas foram removidas corretamente após salvar
dfLabel_a_check = pd.read_csv(output_path_a)
dfLabel_b_check = pd.read_csv(output_path_b)
print("Colunas restantes em dfLabel_a após salvar:", dfLabel_a_check.columns)
print("Colunas restantes em dfLabel_b após salvar:", dfLabel_b_check.columns)

print("Dataset modificado salvo em:", output_path_a)
print("Dataset modificado salvo em:", output_path_b)
