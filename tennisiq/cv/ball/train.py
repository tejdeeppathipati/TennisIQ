from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial import distance
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from tennisiq.common import get_device
from tennisiq.cv.ball.dataset import TrackNetDataset, prepare_tracknet_dataset
from tennisiq.cv.ball.model import BallTrackerNet
from tennisiq.cv.ball.postprocess import postprocess


def train_one_epoch(model, train_loader, optimizer, device: str, epoch: int, max_iters: int = 200):
    start_time = time.time()
    losses = []
    criterion = nn.CrossEntropyLoss()
    model.train()
    max_iters = min(max_iters, len(train_loader))

    for iter_id, batch in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(batch[0].float().to(device))
        gt = torch.tensor(batch[1], dtype=torch.long, device=device)
        loss = criterion(out, gt)

        loss.backward()
        optimizer.step()

        duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f"train | epoch={epoch} iter=[{iter_id}|{max_iters}] loss={loss.item():.6f} time={duration}")
        losses.append(loss.item())

        if iter_id >= max_iters - 1:
            break

    return float(np.mean(losses)) if losses else 0.0


def validate(model, val_loader, device: str, epoch: int, min_dist: int = 5) -> Tuple[float, float, float, float]:
    losses = []
    tp = [0, 0, 0, 0]
    fp = [0, 0, 0, 0]
    tn = [0, 0, 0, 0]
    fn = [0, 0, 0, 0]
    criterion = nn.CrossEntropyLoss()
    model.eval()

    for iter_id, batch in enumerate(val_loader):
        with torch.no_grad():
            out = model(batch[0].float().to(device))
            gt = torch.tensor(batch[1], dtype=torch.long, device=device)
            loss = criterion(out, gt)
            losses.append(loss.item())

            output = out.argmax(dim=1).detach().cpu().numpy()
            for i in range(len(output)):
                x_pred, y_pred = postprocess(output[i])
                x_gt = float(batch[2][i])
                y_gt = float(batch[3][i])
                vis = int(batch[4][i])

                pred_ok = x_pred is not None and y_pred is not None
                if pred_ok:
                    if vis != 0:
                        dst = distance.euclidean((x_pred, y_pred), (x_gt, y_gt))
                        if dst < min_dist:
                            tp[vis] += 1
                        else:
                            fp[vis] += 1
                    else:
                        fp[vis] += 1
                else:
                    if vis != 0:
                        fn[vis] += 1
                    else:
                        tn[vis] += 1

            print(
                f"val | epoch={epoch} iter=[{iter_id}|{len(val_loader)}] loss={np.mean(losses):.6f} "
                f"tp={sum(tp)} tn={sum(tn)} fp={sum(fp)} fn={sum(fn)}"
            )

    eps = 1e-15
    precision = sum(tp) / (sum(tp) + sum(fp) + eps)
    vc1 = tp[1] + fp[1] + tn[1] + fn[1]
    vc2 = tp[2] + fp[2] + tn[2] + fn[2]
    vc3 = tp[3] + fp[3] + tn[3] + fn[3]
    recall = sum(tp) / (vc1 + vc2 + vc3 + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return float(np.mean(losses)) if losses else 0.0, float(precision), float(recall), float(f1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=str, default="data/datasets/balltracking")
    parser.add_argument("--raw-data-dir", type=str, default="data/datasets/balltracking/images")
    parser.add_argument("--prepare-if-missing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--exp-id", type=str, default="default")
    parser.add_argument("--num-epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--val-intervals", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prepare_if_missing and not os.path.exists(os.path.join(args.dataset_root, "labels_train.csv")):
        prepare_tracknet_dataset(args.raw_data_dir, args.dataset_root)

    train_dataset = TrackNetDataset("train", args.dataset_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    val_dataset = TrackNetDataset("val", args.dataset_root)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True)

    device = get_device(args.device)
    model = BallTrackerNet().to(device)

    exps_path = os.path.join("checkpoints", "ball", args.exp_id)
    tb_path = os.path.join(exps_path, "plots")
    os.makedirs(tb_path, exist_ok=True)
    log_writer = SummaryWriter(tb_path)

    model_last_path = os.path.join(exps_path, "last.pt")
    model_best_path = os.path.join(exps_path, "best.pt")

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    val_best_metric = 0.0

    for epoch in range(args.num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, args.steps_per_epoch)
        log_writer.add_scalar("Train/loss", train_loss, epoch)
        log_writer.add_scalar("Train/lr", optimizer.param_groups[0]["lr"], epoch)

        if epoch > 0 and epoch % args.val_intervals == 0:
            val_loss, precision, recall, f1 = validate(model, val_loader, device, epoch)
            log_writer.add_scalar("Val/loss", val_loss, epoch)
            log_writer.add_scalar("Val/precision", precision, epoch)
            log_writer.add_scalar("Val/recall", recall, epoch)
            log_writer.add_scalar("Val/f1", f1, epoch)
            if f1 > val_best_metric:
                val_best_metric = f1
                torch.save(model.state_dict(), model_best_path)
            torch.save(model.state_dict(), model_last_path)


if __name__ == "__main__":
    main()
