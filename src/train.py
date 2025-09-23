import os
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models import build_model
from ferplus import FERPlusParameters, FERPlusDataset


emotion_table = {
    0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
    4: 'anger', 5: 'disgust', 6: 'fear', 7: 'contempt'
}


train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid']
test_folders  = ['FER2013Test']

def loss_fn(training_mode, logits, targets):
    if training_mode in ('majority','probability','crossentropy'):
        logq = torch.log_softmax(logits, dim=1)
        loss = -(targets * logq).sum(dim=1).mean()
    elif training_mode == 'multi_target':
        q = torch.softmax(logits, dim=1)
        prod = targets * q
        m, _ = prod.max(dim=1)
        loss = -torch.log(m + 1e-12).mean()
    else:
        raise ValueError(f"Unknown training_mode: {training_mode}")
    return loss


def validate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    correct_per_class = torch.zeros(len(emotion_table), dtype=torch.long, device=device)
    total_per_class   = torch.zeros(len(emotion_table), dtype=torch.long, device=device)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1)
            labels = y.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for i in range(len(emotion_table)):
                mask = labels == i
                total_per_class[i]   += mask.sum()
                correct_per_class[i] += (preds[mask] == i).sum()

    acc = correct / max(total, 1)
    class_accs = {
        i: (correct_per_class[i] / total_per_class[i]).item() if total_per_class[i] > 0 else 0.0
        for i in range(len(emotion_table))
    }
    return acc, class_accs


def save_results_to_excel(file_path, row_data):
    import pandas as pd
    try:
        df = pd.read_excel(file_path, sheet_name="resultados")
    except FileNotFoundError:
        df = pd.DataFrame(columns=row_data.keys())

    df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    df.to_excel(file_path, sheet_name="resultados", index=False)
 

def accuracy_from_logits(logits, targets):
    pred = torch.argmax(logits, dim=1)
    true = torch.argmax(targets, dim=1)
    return (pred == true).float().mean().item()

def get_sampler(sampler_type, dataset):
    if sampler_type is None:
        return None

    if sampler_type == "weighted":        
        class_counts = torch.tensor(dataset.per_emotion_count, dtype=torch.float)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        labels = torch.tensor(dataset.labels, dtype=torch.long)
        sample_weights = class_weights[labels]

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  
            replacement=True    
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")
    
def display_class_distribution(type, dataset):
    class_counts = np.bincount(dataset.labels, minlength=len(emotion_table))
    logging.info(f"{type} class distribution:")    
    for idx, count in enumerate(class_counts):
        cname = emotion_table[idx]
        logging.info(f"  {cname:10s}: {count} ({count / len(dataset.labels) * 100:.2f}%)")
    
    logging.info(f"{type} dataset size: {len(dataset.labels)}\n")


def main(base_folder, training_mode='majority', model_name='VGG13', max_epochs=100, batch_size=32, num_workers=0, sampler_type=None, results_file="resultados.xlsx"):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    pin_memory = is_cuda and (num_workers > 0)

    output_model_path = os.path.join(base_folder, 'models')
    output_model_folder = os.path.join(output_model_path, f"{model_name}_{training_mode}")
    os.makedirs(output_model_folder, exist_ok=True)


    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', handlers=[
        logging.FileHandler(os.path.join(output_model_folder, "train.log")),
        logging.StreamHandler()
    ])
    
    writer = SummaryWriter(log_dir=os.path.join(output_model_folder, "tensorboard"))

    logging.info(f"Starting with training mode {training_mode} using {model_name} model and max epochs {max_epochs}.")

    num_classes = len(emotion_table)
    model = build_model(num_classes, model_name).to(device)

    train_params = FERPlusParameters(num_classes, getattr(model, 'input_height', 64), getattr(model, 'input_width', 64), training_mode, False)
    eval_params  = FERPlusParameters(num_classes, getattr(model, 'input_height', 64), getattr(model, 'input_width', 64), "majority", True)

    train_ds = FERPlusDataset(base_folder, train_folders, "label.csv", train_params)
    val_ds   = FERPlusDataset(base_folder, valid_folders, "label.csv", eval_params)
    test_ds  = FERPlusDataset(base_folder, test_folders, "label.csv", eval_params)

    display_class_distribution("Train", train_ds)
    display_class_distribution("Validation", val_ds)
    display_class_distribution("Test", test_ds)

    sampler = get_sampler(sampler_type, train_ds)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  shuffle=(sampler is None), num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    base_lr = getattr(model, 'learning_rate', 1e-3)

    def lr_lambda(epoch):
        if epoch < 20:
            return 1.0
        elif epoch < 40:
            return 0.5
        else:
            return 0.1

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    best_val_acc = 0.0
    best_epoch = 0
    best_test_acc = 0.0
    best_row = None

    for epoch in range(max_epochs):        
        model.train()
        start_time = time.time()
        running_loss, running_acc, n_samples = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device, non_blocking=is_cuda), y.to(device, non_blocking=is_cuda)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(training_mode, logits, y)
            loss.backward()
            optimizer.step()
            
            bs = x.size(0)

            running_loss += loss.detach() * bs  
            preds = logits.argmax(dim=1)
            true  = y.argmax(dim=1)
            running_acc += (preds == true).sum()
            n_samples += bs

            avg_loss = (running_loss / n_samples).item()
            avg_acc  = (running_acc.float() / n_samples).item()
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc*100:.2f}%"})

        scheduler.step()
        train_loss = (running_loss / max(n_samples, 1)).item()
        train_acc  = (running_acc.float() / max(n_samples, 1)).item()        

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        val_acc, val_class_accs = validate(model, val_loader, device)
        writer.add_scalar("Accuracy/val", val_acc, epoch)        

        test_class_accs = {}
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            final_test_acc, test_class_accs = validate(model, test_loader, device)
            writer.add_scalar("Accuracy/test", final_test_acc, epoch)

            if final_test_acc > best_test_acc:
                best_epoch = epoch
                best_test_acc = final_test_acc

                best_row = {
                    "modelo": model_name,
                    "training_type": training_mode,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "test_acc": final_test_acc,
                    **{f"val_{emotion_table[i]}": val_class_accs[i] for i in emotion_table},
                    **{f"test_{emotion_table[i]}": test_class_accs[i] for i in emotion_table},
                }

                torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                            os.path.join(output_model_folder, f"best_model.pt"))

        logging.info(f"Epoch {epoch}: {time.time() - start_time:.2f}s")
        logging.info(f"  train loss:\t{train_loss:.4f}")
        logging.info(f"  train acc:\t{train_acc*100:.2f}%")
        logging.info(f"  val acc:\t{val_acc*100:.2f}%")
        logging.info(f"  val class acc:")
        for idx, acc in val_class_accs.items():
            cname = emotion_table[idx]
            logging.info(f"    {cname:<10s}: {acc*100:.2f}%")
            writer.add_scalar(f"ValClassAcc/{cname}", acc, epoch)

        if epoch == best_epoch:
            logging.info(f"  test acc:\t{best_test_acc*100:.2f}%")
            logging.info(f"  test class acc:")
            for idx, acc in test_class_accs.items():
                cname = emotion_table[idx]
                logging.info(f"    {cname:<10s}: {acc*100:.2f}%")
                writer.add_scalar(f"TestClassAcc/{cname}", acc, epoch)

    writer.close()

    if best_row is not None:
        save_results_to_excel(results_file, best_row)

    logging.info(f"Best val acc: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    logging.info(f"Test acc @ best val: {best_test_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--base_folder", type=str, required=True)
    parser.add_argument("-m","--training_mode", type=str, default='majority')
    parser.add_argument("--model_name", type=str, default='VGG13')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampler", type=str, default=None)
    parser.add_argument("-r", "--results_file", type=str, default="resultados.xlsx")
    args = parser.parse_args()
    main(args.base_folder, args.training_mode, args.model_name, args.epochs, args.batch_size, args.num_workers, args.sampler, args.results_file)



