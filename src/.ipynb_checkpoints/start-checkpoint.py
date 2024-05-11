import subprocess
import os

def run_script(script_path):
    """Executa um script Python e permite que a saída seja mostrada diretamente no terminal."""
    print(f"Executando {script_path}...")
    try:
        # subprocess.call não captura a saída, permitindo que ela seja exibida no terminal
        subprocess.call(['python', script_path])
        print(f"Sucesso: {script_path} completou sem erros.\n")
    except Exception as e:
        print(f"Erro ao executar {script_path}:\n{e}\n")

def main():
    scripts_dir = 'C:/Users/Bruno/Desktop/projeto_retinopatia/src'
    scripts = ['data_preparation.py', 'train_model.py', 'evaluate_model.py', 'predict.py']
    
    for script in scripts:
        script_path = os.path.join(scripts_dir, script)
        run_script(script_path)

if __name__ == "__main__":
    main()
