import torch
import os
import time
import logging
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse

from ferplus import FERPlusDataset
from models import build_model  

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
logging.info(f"Cuda available: {torch.cuda.is_available()}")

emotion_table = {
    'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
    'anger': 4, 'disgust': 5, 'fear': 6, 'contempt': 7
}

def cost_func(mode, outputs, targets):
    if torch.any(torch.isnan(outputs)):
        logging.warning("⚠️ Outputs contém NaN!")
    if torch.any(torch.isnan(targets)):
        logging.warning("⚠️ Targets contém NaN!")

    if mode == 'multi_target':
        targets_smoothed = targets + 1e-9
        probs = torch.softmax(outputs, dim=1)
        weighted_probs = probs * targets_smoothed
        max_vals, _ = torch.max(weighted_probs, dim=1)
        perda = -torch.log(max_vals).mean()
        logging.debug(f"[Multi-target] Loss: {perda.item():.4f}")
        return perda
    else:
        idx = targets.argmax(dim=1)
        perda = torch.nn.CrossEntropyLoss()(outputs, idx)
        logging.debug(f"[CrossEntropy] Loss: {perda.item():.4f}")
        return perda


def main(base_folder, mode='majority', model_name='VGG13', epochs=3, bs=64):
    paths = {
        'train': os.path.join(base_folder, 'FER2013Train'),
        'valid':   os.path.join(base_folder, 'FER2013Valid'),
        'test':  os.path.join(base_folder, 'FER2013Test')
    }
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{mode}")

    ds = {
        k: FERPlusDataset(
            folder_paths=[paths[k]],  # mantém lista se o dataset espera assim
            label_file_name='label.csv',
            num_classes=len(emotion_table),
            width=64,
            height=64,
            mode=mode if k == 'train' else 'majority',
            shuffle=(k == 'train'),
            deterministic=(k != 'train')
        )
        for k in paths
    }


        # Log de tamanho e distribuição
    for split, dataset in ds.items():
        logging.info(f"{split} size: {len(dataset)} samples")
        if split == 'train':
            dataset.plot_class_distribution(writer, step=0)

   # DataLoaders
    dl = {
        split: DataLoader(dataset, batch_size=bs, shuffle=(split=='train'),
                          num_workers=4, pin_memory=True)
        for split, dataset in ds.items()
    }

    # Modelo, otimizador e scheduler
    model = build_model(len(emotion_table), model_name).to(device)
    lr = getattr(model, "learning_rate", 0.01)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)

    best_val, best_ep = 0.0, 0
    best_model_path = f"{model_name}_{mode}.pth"

    global_step = 0
    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        logging.info(f"--- Epoch {ep}/{epochs} ---")
        for x, y in tqdm(dl['train'], desc="Train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = cost_func(mode, out, y)
            loss.backward()
            opt.step()

            preds = out.argmax(dim=1)
            trues = y.argmax(dim=1) if mode!='multi_target' else y.argmax(dim=1)

            running_loss   += loss.item() * x.size(0)
            running_correct+= (preds==trues).sum().item()
            running_total  += x.size(0)
            
            # TensorBoard scalars (a cada batch)
            writer.add_scalar("Loss/Train_batch", loss.item(), global_step)
            global_step += 1

        # Scheduler por época
        scheduler.step()

        # Métricas de época
        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total
        writer.add_scalar("Loss/Train_epoch", train_loss, ep)
        writer.add_scalar("Accuracy/Train", train_acc, ep)

        # Validação
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for x, y in dl['valid']:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1)==y.argmax(1)).sum().item()
        val_acc = val_correct / len(ds['valid'])
        writer.add_scalar("Accuracy/Valid", val_acc, ep)
        logging.info(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | Valid acc: {val_acc:.4f} | time: {time.time()-t0:.1f}s")

        # Salva melhor
        if val_acc > best_val:
            best_val, best_ep = val_acc, ep
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"🚀 New best model at epoch {ep} (val_acc={val_acc:.4f})")

    # Avaliação final no teste
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in dl['test']:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_correct += (out.argmax(1)==y.argmax(1)).sum().item()
    test_acc = test_correct / len(ds['test'])
    logging.info(f"🏁 Test acc (best epoch {best_ep}): {test_acc:.4f}")

    writer.add_text("Info", f"Best valid epoch: {best_ep}, test_acc: {test_acc:.4f}")
    writer.close()
    logging.info(f"Model saved to {best_model_path}")
    logging.info("Training complete!")


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-d","--base_folder", required=True)
    p.add_argument("-m","--training_mode", default="majority")
    p.add_argument("-e","--epochs", type=int, default=3)
    p.add_argument("-b","--batch_size", type=int, default=64)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(
        base_folder=args.base_folder,
        mode=args.training_mode,
        epochs=args.epochs,
        bs=args.batch_size
    )