import os
import re
import pandas as pd
import argparse


def parse_log_file(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    epoch_data = []
    current_epoch = None
    val_acc, test_acc = None, None
    val_classes, test_classes = {}, {}
    temp_val_classes = {}
    current_block = None

    for line in lines:
        # detecta blocos de classes
        if "Val per-class accuracy:" in line:
            current_block = "val"
            temp_val_classes = {}
            continue

        if "Test per-class accuracy:" in line:
            current_block = "test"
            test_classes = {}
            continue

        match = re.search(r"\s+(\w+)\s*:\s*([\d.]+)%", line)
        if match and current_block:
            label, acc = match.group(1), float(match.group(2))
            if current_block == "val":
                temp_val_classes[label] = acc
            elif current_block == "test":
                test_classes[label] = acc
            continue

        # detecta início de epoch
        if "Epoch" in line and ":" in line:
            match_epoch = re.search(r"Epoch (\d+)", line)
            if match_epoch:
                # se já estávamos em outro epoch, salva o anterior antes de resetar
                if current_epoch is not None and val_acc is not None:
                    epoch_data.append(
                        (current_epoch, val_acc, test_acc, val_classes.copy(), test_classes.copy())
                    )

                # inicia novo epoch
                current_epoch = int(match_epoch.group(1))
                val_acc, test_acc = None, None
                test_classes = {}
                val_classes = temp_val_classes.copy()
                current_block = None
            continue

        # pega métricas globais
        if "val acc:" in line:
            m = re.search(r"val acc:\s+([\d.]+)%", line)
            if m:
                val_acc = float(m.group(1))

        if "test acc:" in line:
            m = re.search(r"test acc:\s+([\d.]+)%", line)
            if m:
                test_acc = float(m.group(1))

    # salva o último epoch se não foi salvo dentro do loop
    if current_epoch is not None and val_acc is not None:
        epoch_data.append(
            (current_epoch, val_acc, test_acc, val_classes.copy(), test_classes.copy())
        )

    if not epoch_data:
        return None

    # escolhe melhor epoch (maior val_acc)
    best = max(epoch_data, key=lambda x: x[1] if x[1] is not None else -1)
    return best

def process_all_logs(base_dir, output_file="resultados.xlsx"):
    all_rows = []

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path) or "_" not in folder:
            continue

        model, training_type = folder.split("_", 1)
        log_path = os.path.join(folder_path, "train.log")
        if not os.path.exists(log_path):
            continue

        result = parse_log_file(log_path)
        if not result:
            print(f"[AVISO] Nenhum epoch válido encontrado em {log_path}")
            continue

        epoch, val_acc, test_acc, val_classes, test_classes = result

        row = {
            "modelo": model,
            "training_type": training_type,
            "epoch": epoch,
            "val_acc": val_acc,
            "test_acc": test_acc if test_acc is not None else "",  # mantém vazio no Excel
        }

        # adiciona todas as classes de validação
        for k, v in val_classes.items():
            row[f"val_{k}"] = v

        # adiciona todas as classes de teste (se houver)
        for k, v in test_classes.items():
            row[f"test_{k}"] = v

        all_rows.append(row)

    if not all_rows:
        print("Nenhum dado encontrado!")
        return

    df = pd.DataFrame(all_rows)
    df.to_excel(output_file, sheet_name="Resultados", index=False)
    print(f"Planilha salva em: {output_file}")



def main():
    parser = argparse.ArgumentParser(description="Converte logs de treino em planilha Excel (1 sheet consolidada)")
    parser.add_argument("base_dir", help="Diretório base com pastas {modelo}_{treinamento}")
    parser.add_argument("-o", "--output", default="resultados.xlsx", help="Nome do arquivo Excel de saída")

    args = parser.parse_args()
    process_all_logs(args.base_dir, args.output)


if __name__ == "__main__":
    main()
