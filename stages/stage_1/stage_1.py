import os
import argparse
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── config ───────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH  = "stages/stage_1/mitunet_finetune_a6_mit_b4_tversky_8864_28E.pth"
IMAGE_PATH    = "images/plan_05.png"

PROB_THRESHOLD     = 0.4    # lower than default 0.5 → more detections before filtering
MIN_HALF_THICK_PX  = 2      # used only in preprocessing to identify thin walls
MIN_COMPONENT_AREA = 2000   # keep only components >= this many pixels (removes
                             # isolated noise and very short partition-wall segments)


# ── Option 2: preprocessing — blank out thin walls ────────────────────────────
# Thin walls (those whose half-thickness < MIN_HALF_THICK_PX in the source image)
# are painted pure white so the model ignores them entirely.

def _thin_wall_mask(image_bgr):
    """Return mask of pixels that belong to thin walls (binary, 255/0)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, all_walls = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(all_walls, cv2.DIST_L2, 5)
    # thick seed: pixels far enough from wall edge
    thick_seed = (dist >= MIN_HALF_THICK_PX).astype(np.uint8) * 255
    if cv2.countNonZero(thick_seed) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thick_restored = cv2.dilate(thick_seed, k)
    else:
        thick_restored = np.zeros_like(all_walls)
    return cv2.bitwise_and(all_walls, cv2.bitwise_not(thick_restored))


def preprocess_suppress_thin_walls(image_bgr):
    """
    Option 2: blank thin walls to white so only thick walls reach the model.
    Returns modified BGR image.
    """
    thin = _thin_wall_mask(image_bgr)
    enhanced = image_bgr.copy()
    enhanced[thin > 0] = [255, 255, 255]
    return enhanced


# ── Option 1: post-processing — component area filter ────────────────────────
# The model over-detects all walls.  Large connected components are structural
# (shear) walls; tiny fragments are noise or short partition walls.

def postprocess(mask_uint8):
    """
    Remove connected components below MIN_COMPONENT_AREA.
    Returns cleaned binary mask (uint8, 0/255).
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8)
    clean = np.zeros_like(mask_uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= MIN_COMPONENT_AREA:
            clean[labels == i] = 255
    return clean


# ── model ────────────────────────────────────────────────────────────────────

aux_segformer = smp.Segformer(encoder_name="mit_b4", encoder_weights=None)
model = smp.Unet(
    encoder_name="mit_b4",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    decoder_attention_type="scse",
)
model.encoder = aux_segformer.encoder
state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def run_model(img_bgr, orig_w, orig_h):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aug     = transform(image=img_rgb)
    tensor  = aug["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(tensor))
        mask  = (probs > PROB_THRESHOLD).float()
    result = mask.squeeze().cpu().numpy()
    return (cv2.resize(result, (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)


# ── pipeline ─────────────────────────────────────────────────────────────────

def run(image_path=IMAGE_PATH, out_dir="."):
    """Segment one floor-plan image into wall masks written to out_dir."""
    os.makedirs(out_dir, exist_ok=True)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    orig_h, orig_w = image_bgr.shape[:2]

    # Option 2: blank thin walls in the input
    image_enhanced = preprocess_suppress_thin_walls(image_bgr)
    cv2.imwrite(os.path.join(out_dir, "output_preprocessed.png"), image_enhanced)

    # Run model on original and enhanced inputs
    mask_raw      = run_model(image_bgr,      orig_w, orig_h)
    mask_enhanced = run_model(image_enhanced, orig_w, orig_h)

    # Option 1: component-area filter on both
    mask_raw_filtered      = postprocess(mask_raw)
    mask_enhanced_filtered = postprocess(mask_enhanced)

    # Save all variants for comparison
    cv2.imwrite(os.path.join(out_dir, "output_mask.png"),               mask_raw)
    cv2.imwrite(os.path.join(out_dir, "output_mask_raw_filtered.png"),      mask_raw_filtered)
    cv2.imwrite(os.path.join(out_dir, "output_mask_enhanced_filtered.png"), mask_enhanced_filtered)
    cv2.imwrite(os.path.join(out_dir, "output_mask_.png"),                   mask_raw_filtered)

    # Summary
    def count_components(m):
        n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        return n - 1, sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, n))

    rc, rp = count_components(mask_raw)
    rfc, rfp = count_components(mask_raw_filtered)
    efc, efp = count_components(mask_enhanced_filtered)

    print(f"\nImage: {orig_w}x{orig_h}  ({os.path.basename(image_path)})")
    print(f"  Raw model output          : {rc:2d} components, {rp:6d} px")
    print(f"  Raw + area filter         : {rfc:2d} components, {rfp:6d} px")
    print(f"  Enhanced + area filter    : {efc:2d} components, {efp:6d} px")
    print(f"  -> {os.path.join(out_dir, 'output_mask.png')}")
    return os.path.join(out_dir, "output_mask.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Stage 1 - wall segmentation")
    ap.add_argument("-i", "--input", default=IMAGE_PATH,
                    help="Input floor-plan image")
    ap.add_argument("--out-dir", default=".",
                    help="Directory to write the output masks")
    args = ap.parse_args()
    run(args.input, args.out_dir)
