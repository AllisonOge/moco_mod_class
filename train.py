""""
Script to train a model (either from scratch or with a pretrained model)

Dataset is the RML22 pickle file for modulation classification with 10 classes and 9 snr levels
"""
import numpy as np
import torch
import torch.nn.functional as F
from mobilenetv3_1d import mobilenetv3
from dataset import RML22_Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
import yaml
import argparse
import os
from utils import Tracker, CSVLogger, train_epoch, validate_epoch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


parser = argparse.ArgumentParser(description='Train a model on RML22 dataset')
parser.add_argument("--config", "-c", type=str,
                    help="Path to the config file", required=True)


def main():
    args = parser.parse_args()

    # load the config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # create an experiment directory if it does not exist only if not resuming training
    os.makedirs("experiments", exist_ok=True)
    experiment_name = os.path.join(
        "experiments", config.get("experiment_name"))
    i = 0
    while not config.get("checkpoint") and os.path.exists(experiment_name):
        i += 1
        experiment_name = os.path.join(
            "experiments", f"{config.get('experiment_name')}_{i}")
        print(f"Experiment name already exists. Changing to {experiment_name}")
    os.makedirs(experiment_name, exist_ok=True)

    # save the configuration file
    with open(f"{experiment_name}/config.yaml", "w") as f:
        yaml.dump(config, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_txfm = Compose([
        Lambda(lambda x: torch.tensor(x)),
    ])

    # load the datasets
    train_ds = RML22_Dataset(config.get(
        "dataset").get("train"), data_txfm=data_txfm)
    val_ds = RML22_Dataset(config.get(
        "dataset").get("val"), data_txfm=data_txfm)

    train_loader = DataLoader(train_ds, batch_size=config.get("batch_size"), shuffle=True,
                              num_workers=config.get("num_workers"), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.get("batch_size"), shuffle=False,
                            num_workers=config.get("num_workers"), pin_memory=True)

    model = mobilenetv3(num_classes=config.get(
        "nclasses"), in_chans=2).to(device)
    model_parameters = model.parameters()
    # load weight of pretrained model if not none
    if config.get("pretrained"):
        weights = torch.load(config.get("pretrained"),
                             weights_only=True, map_location=device)
        model.load_state_dict(
            {k.replace("module.base_encoder.", ""): v for k, v in weights["state_dict"].items()}, strict=False)
        print("Pretrained model loaded")
        if config.get("freeze"):
            # freeze the backbone
            for name, param in model.named_parameters():
                if "classifier" not in name:
                    param.requires_grad = False
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        else:
            # finetune with small learning rate
            feature_params = [
                p for n, p in model.named_parameters() if 'classifier' not in n]
            model_parameters = \
                [
                    {'params': feature_param,
                    'lr': config.get('lr') / 10}  # 1/10th of the learning rate
                    for feature_param in feature_params
                ]
            model_parameters += [
                {'params': [p for n, p in model.named_parameters()
                            if 'classifier' in n]}
            ]
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if not p.requires_grad):,} frozen parameters")
    print(
        f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(
        f"Model has {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    criterion = CrossEntropyLoss()
    # TODO: add more optimizers and schedulers
    optim = AdamW(model_parameters, lr=config.get(
        "lr"), weight_decay=config.get("weight_decay"))
    lr_scheduler = ReduceLROnPlateau(optim)

    EPOCHS = config.get("epochs")
    start_epoch = 1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    logger = CSVLogger(filename=f"{experiment_name}/results.csv", append=True)
    tracker = Tracker("val_loss")
    if config.get("checkpoint"):
        checkpoint = torch.load(config.get("checkpoint"), weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optim.load_state_dict(checkpoint["opt_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming training from epoch {start_epoch}") if start_epoch <= EPOCHS else None

        old_history = np.vstack(logger.read())
        for i in range(old_history.shape[1]):
            field = str(old_history[0, i])
            if field != "epoch":
                history[field] = list(map(float, old_history[1:, i].tolist()))
            else:
                history[field] = list(map(int, old_history[1:, i].tolist()))


        print(history)
        tracker.best = np.min(history["val_loss"])

    for epoch in range(start_epoch, EPOCHS + 1):
        avg_tloss = train_epoch(model, train_loader,
                                criterion, optim, device)
        avg_vloss, avg_acc = validate_epoch(
            model, val_loader, criterion, device)
        lr_scheduler.step(avg_vloss)
        lr = lr_scheduler.optimizer.param_groups[0]["lr"]
        # save the model if the validation loss is the best
        print(f"Epoch[{epoch}/{EPOCHS}] Train Loss: {avg_tloss:.2f}, Val Loss: {avg_vloss:.2f}, Val Acc: {avg_acc:.2f}, LR: {lr:.2e}")
        if avg_vloss < tracker.best:
            tracker.best = avg_vloss
            torch.save(model.state_dict(), f"{experiment_name}/weights.pth")
            print(f"Best model saved at {experiment_name}/weights.pth")
        history["train_loss"].append(avg_tloss)
        history["val_loss"].append(avg_vloss)
        history["val_acc"].append(avg_acc)
        history["lr"].append(lr)
        logger.save([epoch, avg_tloss, avg_vloss, avg_acc, lr])

        # save the last checkpoint
        state_dict = {
            "model_state_dict": model.state_dict(),
            "opt_state_dict": optim.state_dict(),
            "lr_state_dict": lr_scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(state_dict, f"{experiment_name}/last_checkpoint.pt")
        print(f"Checkpoint saved at {experiment_name}/last_checkpoint.pt")

    logger.close()
    fig = plt.figure(figsize=(15, 4))
    ax = fig.add_subplot(131)
    ax.plot(history["epoch"], history["train_loss"], label="train loss")
    ax.plot(history["epoch"], history["val_loss"], label="val losss")
    ax.set_xlabel("epochs")
    ax.legend()

    ax = fig.add_subplot(132)
    ax.plot(history["epoch"], history["lr"])
    ax.set_xlabel("epochs")

    ax = fig.add_subplot(133)
    ax.plot(history["epoch"], history["val_acc"])
    ax.set_xlabel("epochs")
    fig.savefig(f"{experiment_name}/training_val_loss.png", dpi=300)
    plt.close()

    # anaylse the model performance with test data if provided else use the validation data
    # plot the confusion matrix and save some predictions to experiment directory
    # load the best model
    print(model.load_state_dict(torch.load(f"{experiment_name}/weights.pth", weights_only=True)))
    model.eval()

    true_labels = []
    pred_labels = []
    # class_acc = {
    #     k: 0 for k in config.get("classes").keys()
    # }
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_hat = model(xb)
            yb_hat = torch.argmax(F.softmax(yb_hat, dim=-1), dim=1)
            true_labels.extend(yb[:, 0].cpu().numpy())
            pred_labels.extend(yb_hat.cpu().numpy())

    # TODO: save the confusion matrix
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    print(classification_report(true_labels, pred_labels,
          target_names=config.get("classes").values()))
    cm = confusion_matrix(true_labels, pred_labels, normalize='true')

    acc = accuracy_score(true_labels, pred_labels, normalize=True)
    prec = precision_score(true_labels, pred_labels, average="macro")
    rec = recall_score(true_labels, pred_labels, average="macro")

    threshold = 0.1
    cm_display = np.where(cm > threshold, cm, np.nan)

    plt.figure(figsize=(15, 10), dpi=300)

    plt.subplot()
    ax = sns.heatmap(cm_display,
                     cmap='Blues',
                     annot=True,
                     fmt=".2f",
                     xticklabels=config.get("classes").values(),
                     yticklabels=config.get("classes").values(),
                     vmin=0,
                     vmax=1,
                     annot_kws={"size": 22, "weight": "bold"},
                     cbar_kws={
                         "shrink": 0.8,       # Adjust the size of the color bar
                         "aspect": 10,        # Set the aspect ratio of the color bar
                         "pad": 0.02,         # Space between the color bar and the heatmap
                     })
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    ax.set_xlabel('Predicted', fontsize=22)
    ax.set_ylabel('True', fontsize=22)
    plt.xticks(fontsize=22, rotation=45)
    plt.yticks(fontsize=22, rotation=0)
    ax.set_title(
        f'Accuracy: {acc * 100:.2f}%, Precision: {prec * 100: .2f}%, Recall: {rec * 100: .2f}%', fontsize=30)
    plt.savefig(f"{experiment_name}/confusion_matrix.png")


if __name__ == "__main__":
    main()
