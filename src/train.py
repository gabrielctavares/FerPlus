import torch
import os
import time
import logging
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse

from ferplus import FERPlusDataset, FERPlusParameters
from models import build_model  
from torchvision import transforms
import numpy as np

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
logging.info(f"Cuda available: {torch.cuda.is_available()}")

emotion_table = {
    'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
    'anger': 4, 'disgust': 5, 'fear': 6, 'contempt': 7
}


def cost_func(training_mode, prediction_logits, target):
    if training_mode in ['majority', 'probability', 'crossentropy']:
        labels = torch.argmax(target, dim=1)
        return F.cross_entropy(prediction_logits, labels)  # melhor usar F.* do que nn.*() aqui
    elif training_mode == 'multi_target':
        pred_probs = F.softmax(prediction_logits, dim=1)  # mais idiomático que nn.Softmax()
        prod = pred_probs * target
        loss = -torch.log(torch.max(prod, dim=1).values + 1e-8)
        return loss.mean()
    else:
        raise ValueError(f"Modo de treinamento inválido: {training_mode}")


  

def main(base_folder, mode='majority', model_name='VGG13', epochs=3, bs=64):
    paths = {
        'train': 'FER2013Train',
        'valid':   'FER2013Valid',
        'test':  'FER2013Test'
    }
    
    params_train = FERPlusParameters(target_size=len(emotion_table), width=64, height=64, training_mode=mode, deterministic=False, shuffle=True)
    params_val_test = FERPlusParameters(target_size=len(emotion_table), width=64, height=64, training_mode=mode, deterministic=True, shuffle=False)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(
            (params_train.height, params_train.width), 
            scale=(1.0 / params_train.max_scale, params_train.max_scale), # Mapeia max_scale para scale
            ratio=(1.0 - params_train.max_skew, 1.0 + params_train.max_skew) # Mapeia max_skew para ratio
        ),
        transforms.RandomAffine(
            degrees=params_train.max_angle,
            translate=(params_train.max_shift, params_train.max_shift),
            shear=params_train.max_skew * 180 / np.pi if params_train.max_skew else 0 # Shear em graus
        ) if not params_train.deterministic else transforms.Identity(),
        transforms.RandomHorizontalFlip(p=0.5) if params_train.do_flip else transforms.Identity(),
        transforms.ToTensor(), # Converte PIL Image para FloatTensor e normaliza [0.0, 1.0]
        # Adicione normalização com mean/std se seu modelo foi treinado dessa forma (ex: ImageNet)
        # transforms.Normalize(mean=[0.485], std=[0.229]) # Exemplo para imagens em escala de cinza, se aplicável
    ])

    # Para os conjuntos de validação e teste (sem aumentos, apenas redimensionamento e ToTensor)
    val_test_transforms = transforms.Compose([
        transforms.Resize((params_val_test.height, params_val_test.width)), # Garante o tamanho correto
        transforms.ToTensor(), # Converte PIL Image para FloatTensor e normaliza [0.0, 1.0]
        # transforms.Normalize(mean=[0.485], std=[0.229]) # Mesma normalização do treino
    ])

    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{model_name}_{mode}")

    ds = {
        k: FERPlusDataset(
            base_folder=base_folder,
            sub_folders=[paths[k]],
            label_file_name="label.csv",
            parameters=params_train if k == 'train' else params_val_test,
            transform=train_transforms if k == 'train' else val_test_transforms
        )
        for k in paths
    }


        # Log de tamanho e distribuição
    # for split, dataset in ds.items():
    #     logging.info(f"{split} size: {len(dataset)} samples")
    #     if split == 'train':
    #         dataset.plot_class_distribution(writer, step=0)

   # DataLoaders
    num_workers = min(2, os.cpu_count() // 2)

    dl = {
        split: DataLoader(dataset, batch_size=bs, shuffle=(split=='train'),
                          num_workers=num_workers)
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
            logging.info(f"Batch loss: {loss.item():.4f}")            
            loss.backward()
            opt.step()

            preds = out.argmax(dim=1)
            trues = y.argmax(dim=1)

            #logging.info(f"Batch preds: {preds.tolist()}")
            #logging.info(f"Batch trues: {trues.tolist()}")

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
                val_y = y.argmax(dim=1) if y.ndim > 1 else y
                val_correct += (out.argmax(1)==val_y).sum().item()
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
            val_y = y.argmax(dim=1) if y.ndim > 1 else y
            test_correct += (out.argmax(1)==val_y).sum().item()
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