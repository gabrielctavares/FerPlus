import os, time, argparse, logging
import torch, numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_model
from ferplus import FERPlusDataset

# dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Cuda available: {torch.cuda.is_available()}")

emotion_table = {
    'neutral':0,'happiness':1,'surprise':2,'sadness':3,
    'anger':4,'disgust':5,'fear':6,'contempt':7
}


def cost_func(mode, outputs, targets):
    if mode == 'multi_target':
        targets_smoothed = targets + 1e-9
        probs = torch.softmax(outputs, dim=1)
        
        # Calcula a máxima probabilidade ponderada
        weighted_probs = probs * targets_smoothed
        max_vals, _ = torch.max(weighted_probs, dim=1)
        
        # Perda como negativo do log
        return -torch.log(max_vals).mean()
    else:
        # CrossEntropy para outros modos
        idx = targets.argmax(dim=1)
        return torch.nn.CrossEntropyLoss()(outputs, idx)


def main(base_folder, mode='majority', model_name='VGG13', epochs=3, bs=64):
    # define paths das pastas de dados
    paths = {
        'train': [os.path.join(base_folder,'FER2013Train')],
        'val':   [os.path.join(base_folder,'FER2013Valid')],
        'test':  [os.path.join(base_folder,'FER2013Test')]
    }
 
    ds = {
        k: FERPlusDataset(
            folder_paths=paths[k],
            label_file_name='label.csv',
            num_classes=len(emotion_table),
            width=64,
            height=64,
            mode=mode if k == 'train' else 'majority',
            shuffle=(k == 'train'),
            deterministic=(k != 'train')  # Apenas augment no treino
        )
        for k in paths
    }
    dl = {
        k: DataLoader(ds[k], batch_size=bs, shuffle=(k=='train'), num_workers=4, pin_memory=True)
        for k in paths
    }

    # modelo e otimizador
    model = build_model(len(emotion_table), model_name).to(device)
    opt = optim.SGD(model.parameters(), lr=model.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
    
    best_val, best_test, best_ep = 0, 0, 0

    for ep in range(epochs):
        t0 = time.time()
        model.train()
        loss_sum, correct, total = 0.0, 0, 0

        for x, y in tqdm(dl['train'], desc=f"Epoch {ep+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = cost_func(mode, out, y)
            loss.backward()
            opt.step()
            scheduler.step()
            loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            trues = y.argmax(dim=1)
            correct += (preds == trues).sum().item()
            total += x.size(0)

        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            val_correct = 0
            for x, y in dl['val']:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_correct += (out.argmax(1) == y.argmax(1)).sum().item()
        val_acc = val_correct / len(ds['val'])

        if val_acc > best_val:
            best_val, best_ep = val_acc, ep
            torch.save(model.state_dict(), f"{model_name}_{mode}.pth")
            test_corr = 0
            with torch.no_grad():
                for x, y in dl['test']:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    test_corr += (out.argmax(1) == y.argmax(1)).sum().item()
            best_test = test_corr / len(ds['test'])

        logging.info(
            f"Epoch {ep}: train_acc={train_acc*100:.2f}% "
            f"val_acc={val_acc*100:.2f}% time={(time.time()-t0):.1f}s"
        )

    print(
        f"Melhor val_acc={best_val*100:.2f}% (epoch {best_ep}), "
        f"test_acc={best_test*100:.2f}%"
    )


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
