from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from tennisiq.common import get_device
from tennisiq.cv.court.dataset import CourtDataset
from tennisiq.cv.court.model import CourtKeypointNet
from tennisiq.cv.court.postprocess import postprocess
from tennisiq.cv.court.utils import is_point_in_image


def train_one_epoch(model, train_loader, optimizer, criterion, device: str, epoch: int, max_iters: int = 1000):
    model.train()
    losses = []
    max_iters = min(max_iters, len(train_loader))

    for iter_id, batch in enumerate(train_loader):
        out = model(batch[0].float().to(device))
        gt_hm = batch[1].float().to(device)
        loss = criterion(torch.sigmoid(out), gt_hm)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
        print(f"train | epoch={epoch} iter={iter_id}/{max_iters} loss={loss.item():.6f}")
        if iter_id >= max_iters - 1:
            break

    return float(np.mean(losses)) if losses else 0.0


def validate(model, val_loader, criterion, device: str, epoch: int) -> Tuple[float, int, int, int, int, float, float]:
    model.eval()
    losses = []
    tp, fp, fn, tn = 0, 0, 0, 0
    max_dist = 7

    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            batch_size = batch[0].shape[0]
            out = model(batch[0].float().to(device))
            kps = batch[2]
            gt_hm = batch[1].float().to(device)
            loss = criterion(torch.sigmoid(out), gt_hm)
            pred = torch.sigmoid(out).detach().cpu().numpy()

            for bs in range(batch_size):
                for kps_num in range(14):
                    heatmap = (pred[bs][kps_num] * 255).astype(np.uint8)
                    x_pred, y_pred = postprocess(heatmap)
                    x_gt = float(kps[bs][kps_num][0])
                    y_gt = float(kps[bs][kps_num][1])

                    pred_ok = is_point_in_image(x_pred, y_pred)
                    gt_ok = is_point_in_image(x_gt, y_gt)
                    if pred_ok and gt_ok:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < max_dist:
                            tp += 1
                        else:
                            fp += 1
                    elif pred_ok and not gt_ok:
                        fp += 1
                    elif not pred_ok and gt_ok:
                        fn += 1
                    else:
                        tn += 1

            eps = 1e-15
            precision = round(tp / (tp + fp + eps), 5)
            accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
            print(
                f"val | epoch={epoch} iter={iter_id}/{len(val_loader)} loss={loss.item():.6f} "
                f"tp={tp} fp={fp} fn={fn} tn={tn} precision={precision} accuracy={accuracy}"
            )
            losses.append(loss.item())

    eps = 1e-15
    precision = round(tp / (tp + fp + eps), 5)
    accuracy = round((tp + tn) / (tp + tn + fp + fn + eps), 5)
    return float(np.mean(losses)) if losses else 0.0, tp, fp, fn, tn, precision, accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/datasets/court_identification")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--exp-id", type=str, default="default")
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--val-intervals", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    train_dataset = CourtDataset("train", data_root=args.data_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = CourtDataset("val", data_root=args.data_root)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    model = CourtKeypointNet(out_channels=15).to(device)

    exps_path = os.path.join("checkpoints", "court", args.exp_id)
    tb_path = os.path.join(exps_path, "plots")
    os.makedirs(tb_path, exist_ok=True)

    log_writer = SummaryWriter(tb_path)
    model_last_path = os.path.join(exps_path, "last.pt")
    model_best_path = os.path.join(exps_path, "best.pt")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), weight_decay=0)

    val_best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar("Train/loss", train_loss, epoch)

        if epoch > 0 and epoch % args.val_intervals == 0:
            val_loss, tp, fp, fn, tn, precision, accuracy = validate(model, val_loader, criterion, device, epoch)
            log_writer.add_scalar("Val/loss", val_loss, epoch)
            log_writer.add_scalar("Val/tp", tp, epoch)
            log_writer.add_scalar("Val/fp", fp, epoch)
            log_writer.add_scalar("Val/fn", fn, epoch)
            log_writer.add_scalar("Val/tn", tn, epoch)
            log_writer.add_scalar("Val/precision", precision, epoch)
            log_writer.add_scalar("Val/accuracy", accuracy, epoch)
            if accuracy > val_best_accuracy:
                val_best_accuracy = accuracy
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)


if __name__ == "__main__":
    main()
