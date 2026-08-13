"""
Fine-tune Stage 1 (MiT-B4 UNet) on local StructGAN shear-wall pairs.

Input  : train/data/{L1_7,L2_7,L1L2_8}/train_A/*.png  (architectural image)
Target : shear-wall binary mask extracted from matching train_B/*.png
         (shear wall pixels = red, BGR [0,0,255], tolerance 15)

Usage:
    python finetune_stage1/finetune.py
    python finetune_stage1/finetune.py --epochs 50 --batch-size 4
"""

from __future__ import annotations
import argparse, json, logging, os, time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

log = logging.getLogger(__name__)

# ── shear-wall colour (StructGAN B-images, BGR) ─────────────────────────────
_SW_BGR = np.array([0, 0, 255], dtype=np.uint8)
_TOL    = 15


def extract_sw_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return uint8 mask (0/255) where shear-wall colour is detected."""
    diff = np.abs(img_bgr.astype(np.int16) - _SW_BGR.astype(np.int16))
    return ((diff < _TOL).all(axis=2) * 255).astype(np.uint8)


# ── dataset ──────────────────────────────────────────────────────────────────

class ShearWallDataset(Dataset):
    GROUPS = ["L1_7", "L2_7", "L1L2_8"]

    def __init__(self, data_dir: str, split: str, transform=None):
        self.transform = transform
        self.pairs: list[tuple[str, str]] = []
        root = Path(data_dir)

        if split in ("train", "test"):
            for g in self.GROUPS:
                a_dir = root / g / f"{split}_A"
                b_dir = root / g / f"{split}_B"
                if not a_dir.is_dir():
                    log.warning("Missing: %s", a_dir)
                    continue
                for fp in sorted(a_dir.glob("*.png")):
                    bp = b_dir / fp.name
                    if bp.exists():
                        self.pairs.append((str(fp), str(bp)))
                    else:
                        log.warning("No B-pair for %s", fp.name)

        log.info("Dataset[%s]: %d pairs", split, len(self.pairs))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        a_path, b_path = self.pairs[idx]
        img_a = cv2.imread(a_path)
        img_b = cv2.imread(b_path)
        if img_a is None or img_b is None:
            raise RuntimeError(f"Cannot read pair: {a_path}")

        # Some StructGAN images are wide double-panels — use left half only
        h, w = img_a.shape[:2]
        if w > 1.5 * h:
            img_a = img_a[:, : w // 2]
            img_b = img_b[:, : w // 2]

        image = cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB)   # H×W×3 uint8
        mask  = extract_sw_mask(img_b)                   # H×W uint8 0/255

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out["image"]
            mask  = out["mask"].float() / 255.0           # [0,1]

        return image, mask.unsqueeze(0)                   # (3,H,W), (1,H,W)


# ── transforms ───────────────────────────────────────────────────────────────
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def train_transform() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.4),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def val_transform() -> A.Compose:
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


# ── model ────────────────────────────────────────────────────────────────────

def build_model(weights_path: str, device: str) -> smp.Unet:
    aux = smp.Segformer(encoder_name="mit_b4", encoder_weights=None)
    model = smp.Unet(
        encoder_name="mit_b4",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )
    model.encoder = aux.encoder
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.to(device)


# ── loss ─────────────────────────────────────────────────────────────────────

def tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1.0) -> torch.Tensor:
    p  = torch.sigmoid(logits).view(-1)
    t  = targets.view(-1)
    tp = (p * t).sum()
    fp = ((1 - t) * p).sum()
    fn = (t * (1 - p)).sum()
    return 1 - (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return tversky_loss(logits, targets) + 0.5 * F.binary_cross_entropy_with_logits(logits, targets)


# ── metrics ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def dice_iou(logits: torch.Tensor, targets: torch.Tensor,
             threshold: float = 0.5) -> tuple[float, float]:
    pred = (torch.sigmoid(logits) > threshold).float()
    t    = (targets > 0.5).float()
    tp   = (pred * t).sum().item()
    fp   = (pred * (1 - t)).sum().item()
    fn   = ((1 - pred) * t).sum().item()
    dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
    iou  = (tp + 1)     / (tp + fp + fn + 1)
    return dice, iou


# ── train / val loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = combined_loss(model(imgs), masks)
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, device) -> tuple[float, float, float]:
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        total_loss += combined_loss(logits, masks).item() * imgs.size(0)
        d, i = dice_iou(logits, masks)
        total_dice += d * imgs.size(0)
        total_iou  += i * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n, total_iou / n


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fine-tune Stage 1 shear-wall segmentation")
    p.add_argument("--data-dir",      default="train/data")
    p.add_argument("--weights",       default="stages/stage_1/mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth")
    p.add_argument("--out-dir",       default="finetune_stage1/checkpoints")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch-size",    type=int,   default=2)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--freeze-epochs", type=int,   default=10,
                   help="Keep encoder frozen for this many epochs before full fine-tune")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",        default=None,
                   help="Path to checkpoint to resume from (skips freeze phase, uses lr*0.1)")
    p.add_argument("--start-epoch",   type=int,   default=1,
                   help="Epoch number to start from when resuming (for display only)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── data ──────────────────────────────────────────────────────────────────
    train_ds = ShearWallDataset(args.data_dir, "train", train_transform())
    val_ds   = ShearWallDataset(args.data_dir, "test",  val_transform())

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, num_workers=0, pin_memory=False)

    resuming = args.resume is not None
    total_epochs = args.start_epoch - 1 + args.epochs

    print(f"\n{'='*60}")
    print(f"  Train : {len(train_ds)} pairs")
    print(f"  Val   : {len(val_ds)} pairs")
    print(f"  Device: {args.device}   Epochs: {args.epochs}   LR: {args.lr}")
    if resuming:
        print(f"  Resuming from: {args.resume}  (ep {args.start_epoch} -> {total_epochs})")
    else:
        print(f"  Freeze encoder for first {args.freeze_epochs} epochs")
    print(f"{'='*60}\n")

    # ── model ─────────────────────────────────────────────────────────────────
    if resuming:
        # Load architecture then override weights from checkpoint
        model = build_model(args.weights, args.device)
        state = torch.load(args.resume, map_location=args.device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(args.device)
        # Full fine-tune at reduced LR
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    else:
        model = build_model(args.weights, args.device)
        # Phase 1: decoder-only, encoder frozen
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(
            filter(lambda param: param.requires_grad, model.parameters()),
            lr=args.lr,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.freeze_epochs)

    best_iou = 0.0
    history  = []

    for i in range(1, args.epochs + 1):
        epoch = args.start_epoch - 1 + i

        # Phase 2: unfreeze encoder with lower LR (only in non-resume mode)
        if not resuming and i == args.freeze_epochs + 1:
            print("\n  [Phase 2] Unfreezing encoder -- LR -> 1e-5\n")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.freeze_epochs)

        t0 = time.time()
        tr_loss            = train_epoch(model, train_dl, optimizer, args.device)
        vl_loss, dice, iou = validate(model, val_dl, args.device)
        scheduler.step()
        elapsed = time.time() - t0

        history.append(dict(epoch=epoch, tr_loss=tr_loss, vl_loss=vl_loss,
                            dice=dice, iou=iou))

        msg = (f"Ep {epoch:>3}/{total_epochs}  "
               f"tr={tr_loss:.4f}  vl={vl_loss:.4f}  "
               f"dice={dice:.4f}  iou={iou:.4f}  "
               f"[{elapsed:.0f}s]")
        print(msg)
        log.info(msg)

        # Save best
        if iou > best_iou:
            best_iou  = iou
            best_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"    -> Best IoU {best_iou:.4f}  saved: {best_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            ck = os.path.join(args.out_dir, f"ckpt_ep{epoch:03d}.pth")
            torch.save(model.state_dict(), ck)

    # Final model + history
    torch.save(model.state_dict(), os.path.join(args.out_dir, "final_model.pth"))
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Training complete.  Best IoU: {best_iou:.4f}")
    print(f"  Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
