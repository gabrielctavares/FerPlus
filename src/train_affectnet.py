import os
import time
import argparse
import logging
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models import build_model
from samplers import get_sampler
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision import datasets, transforms 
import kagglehub

from log_util import save_results_to_excel, display_class_distribution, plot_confusion_matrix, display_sampler_distribution


def build_optimizer(model, lr, weight_decay, epochs):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        raise RuntimeError("Nenhum parâmetro treinável encontrado para o otimizador.")

    optimizer = optim.Adam(
        trainable_params,
        lr=lr,                
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    return optimizer, scheduler


def validate(model, dataloader, device, emotion_table):
    model.eval()
    correct, total = 0, 0
    correct_per_class = torch.zeros(len(emotion_table), dtype=torch.long, device=device)
    total_per_class   = torch.zeros(len(emotion_table), dtype=torch.long, device=device)
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total   += y.size(0)
            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            for i in range(len(emotion_table)):
                mask = y == i
                total_per_class[i]   += mask.sum()
                correct_per_class[i] += (preds[mask] == i).sum()
            

    acc = correct / max(total, 1)
    class_accs = {
        i: (correct_per_class[i] / total_per_class[i]).item() if total_per_class[i] > 0 else 0.0
        for i in range(len(emotion_table))
    }    

    cm = confusion_matrix(all_labels, all_preds, labels=list(emotion_table.keys()))
    return acc, class_accs, cm


def main(base_folder, model_name, checkpoint_path=None, max_epochs=100, batch_size=32, num_workers=0, sampler_type=None, results_file="resultados.xlsx"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    pin_memory = is_cuda and (num_workers > 0)
    #kagglehub.login()
    BASE_DATASET_DIR = kagglehub.dataset_download("mstjebashazida/affectnet")

    paths = os.listdir(BASE_DATASET_DIR)
    DATA_DIR = os.path.join(BASE_DATASET_DIR, paths[0])    
    print(os.listdir(DATA_DIR))

    output_model_path = os.path.join(base_folder, 'results')
    output_model_folder = os.path.join(output_model_path, f"{model_name}_affectnet")
    os.makedirs(output_model_folder, exist_ok=True)


    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s', handlers=[
        logging.FileHandler(os.path.join(output_model_folder, "train.log")),
        logging.StreamHandler()
    ])
    
    writer = SummaryWriter(log_dir=os.path.join(output_model_folder, f"tensorboard_{sampler_type or 'none'}"))

    logging.info(f"Starting training {model_name} with max epochs {max_epochs} and sampler {sampler_type if sampler_type else 'none'}.")

    train_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "Train"),
        transform=train_tf
    )

    test_ds = datasets.ImageFolder(
        os.path.join(DATA_DIR, "Test"),
        transform=val_tf
    )

    emotion_table = dict(enumerate(train_ds.classes))

    num_classes = len(train_ds.classes)
    model = build_model(num_classes, model_name, ferplus=False, checkpoint_path=checkpoint_path).to(device)

    display_class_distribution("Train", train_ds, emotion_table)
    display_class_distribution("Test", test_ds, emotion_table)

    sampler = get_sampler(sampler_type, train_ds, verbose=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  shuffle=(sampler is None), num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    #display_sampler_distribution(train_loader)

    base_lr = getattr(model, 'learning_rate', 1e-3)

   
    optimizer, scheduler = build_optimizer(model, base_lr, weight_decay=1e-4, epochs=max_epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    for epoch in range(max_epochs):        
        model.train()
        start_time = time.time()
        running_loss, running_acc, n_samples = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            
            bs = x.size(0)

            running_loss += loss.detach() * bs  
            preds = logits.argmax(dim=1)
            running_acc += (preds == y).sum()
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

        logging.info(f"Epoch {epoch}: {time.time() - start_time:.2f}s")
        logging.info(f"  train loss:\t{train_loss:.4f}")
        logging.info(f"  train acc:\t{train_acc*100:.2f}%")        

    final_test_acc, test_class_accs, test_cm = validate(model, test_loader, device, emotion_table)
    writer.add_scalar("Accuracy/test", final_test_acc, max_epochs)
    logging.info(f"Final test acc:\t{final_test_acc*100:.2f}%")

    
    final_row = {
        "run_id": f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "modelo": model_name,
        "test_acc": final_test_acc,
        **{f"test_{emotion_table[i]}": test_class_accs[i] for i in emotion_table},
        "batch_size": batch_size,
    }   
    save_results_to_excel(results_file, final_row)

    torch.save({'model_state': model.state_dict()},
                os.path.join(output_model_folder, f"model_{batch_size}.pt"))  

    if test_cm is not None:
        labels = list(range(len(emotion_table)))
        fig = plot_confusion_matrix(test_cm, labels, "Matriz de Confusão", os.path.join(output_model_folder, f"confusion_matrix_{batch_size}.pdf"))
        writer.add_figure("ConfusionMatrix/test", fig, max_epochs)        
    writer.close()
    
    logging.info("Training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--base_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, default='VGG16')
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampler", type=str, default=None)
    parser.add_argument("-r", "--results_file", type=str, default=f"resultados_{datetime.now().strftime('%Y%m%d')}.xlsx")
    args = parser.parse_args()
    main(args.base_folder, args.model_name, args.checkpoint_path, args.epochs, args.batch_size, args.num_workers, args.sampler, args.results_file)



