import os
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from ferplus import FERPlusParameters, FERPlusDataset
from definitions import device, emotion_table, emotion_names
from torch.optim.lr_scheduler import LambdaLR


idx2emotion = {v: k for k, v in emotion_table.items()}

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

def accuracy_from_logits(logits, targets):
    pred = torch.argmax(logits, dim=1)
    true = torch.argmax(targets, dim=1)
    return (pred == true).float().mean().item()

def per_class_accuracy(model, dataloader, device):
    model.eval()
    correct = {i: 0 for i in range(len(emotion_table))}
    total   = {i: 0 for i in range(len(emotion_table))}
    
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            labels = torch.argmax(yb, dim=1)

            for p, t in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                total[t] += 1
                if p == t:
                    correct[t] += 1
    
    accs = {}
    for idx in total:
        if total[idx] > 0:
            accs[idx] = correct[idx] / total[idx]
        else:
            accs[idx] = 0.0
    return accs


def main(base_folder, training_mode='majority', model_name='VGG13',
         max_epochs=100, batch_size=32, num_workers=0, device=None):

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    output_model_path = os.path.join(base_folder, 'models')
    output_model_folder = os.path.join(output_model_path, f"{model_name}_{training_mode}")
    os.makedirs(output_model_folder, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(output_model_folder, "train.log"),
        filemode='w', level=logging.INFO
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(f"Starting with training mode {training_mode} using {model_name} model and max epochs {max_epochs}.")

    num_classes = len(emotion_table)
    model = build_model(num_classes, model_name).to(device)

    train_params = FERPlusParameters(num_classes, getattr(model, 'input_height', 64), getattr(model, 'input_width', 64), training_mode, False)
    eval_params  = FERPlusParameters(num_classes, getattr(model, 'input_height', 64), getattr(model, 'input_width', 64), "majority", True)

    train_ds = FERPlusDataset(base_folder, train_folders, "label.csv", train_params)
    val_ds   = FERPlusDataset(base_folder, valid_folders, "label.csv", eval_params)
    test_ds  = FERPlusDataset(base_folder, test_folders, "label.csv", eval_params)

    logging.info("\t\tTrain\tVal\tTest")
    for idx in range(num_classes):
        cname = idx2emotion[idx]
        logging.info(f"{cname:10s}\t{train_ds.per_emotion_count[idx]}\t{val_ds.per_emotion_count[idx]}\t{test_ds.per_emotion_count[idx]}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device=='cuda'))
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=='cuda'))
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=='cuda'))

    base_lr = getattr(model, 'learning_rate', 0.05)


    def lr_lambda(epoch):
        if epoch < 20:
            return 1.0      # 0.05
        elif epoch < 40:
            return 0.5      # 0.025
        else:
            return 0.1      # 0.005

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    best_val_acc = 0.0
    best_epoch = 0
    best_test_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(max_epochs):        
        model.train()
        start_time = time.time()
        running_loss, running_acc, n_samples = 0.0, 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(training_mode, logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy_from_logits(logits.detach(), yb) * bs
            n_samples += bs

            # update tqdm bar with metrics
            avg_loss = running_loss / n_samples
            avg_acc = running_acc / n_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc*100:.2f}%"})

        scheduler.step()
        train_loss = running_loss / max(n_samples, 1)
        train_acc = running_acc / max(n_samples, 1)        

        model.eval()
        with torch.no_grad():
            val_correct, val_count = 0.0, 0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_correct += accuracy_from_logits(logits, yb) * xb.size(0)
                val_count += xb.size(0)
            val_acc = val_correct / max(val_count, 1)

            val_class_accs = per_class_accuracy(model, val_loader, device)
            logging.info("  Val per-class accuracy:")
            for idx, acc in val_class_accs.items():
                cname = idx2emotion[idx]
                logging.info(f"    {cname:10s}: {acc*100:.2f}%")

        test_run = False
        if val_acc > best_val_acc:
            best_val_acc, best_epoch, test_run = val_acc, epoch, True
            torch.save({'epoch': epoch, 'model_state': model.state_dict()},
                       os.path.join(output_model_folder, f"model_{best_epoch}.pt"))
            with torch.no_grad():
                test_correct, test_count = 0.0, 0
                for xb, yb in test_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    test_correct += accuracy_from_logits(logits, yb) * xb.size(0)
                    test_count += xb.size(0)
                final_test_acc = test_correct / max(test_count, 1)
                best_test_acc = max(best_test_acc, final_test_acc)

        logging.info(f"Epoch {epoch}: {time.time() - start_time:.2f}s")
        logging.info(f"  train loss:\t{train_loss:.4f}")
        logging.info(f"  train acc:\t{train_acc*100:.2f}%")
        logging.info(f"  val acc:\t{val_acc*100:.2f}%")
        if test_run:
            logging.info(f"  test acc:\t{final_test_acc*100:.2f}%")
            test_class_accs = per_class_accuracy(model, test_loader, device)
            logging.info("  Test per-class accuracy:")
            for idx, acc in test_class_accs.items():
                cname = idx2emotion[idx]
                logging.info(f"    {cname:10s}: {acc*100:.2f}%")


    logging.info(f"Best val acc: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    logging.info(f"Test acc @ best val: {final_test_acc*100:.2f}%")
    logging.info(f"Best test acc: {best_test_acc*100:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--base_folder", type=str, required=True)
    parser.add_argument("-m","--training_mode", type=str, default='majority')
    parser.add_argument("--model_name", type=str, default='VGG13')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    main(args.base_folder, args.training_mode, args.model_name, args.epochs, args.batch_size, args.num_workers)


