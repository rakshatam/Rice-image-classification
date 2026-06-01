# rice_convnextv2_robust.py
#
# Enhancements:
# 1. Stronger Augmentations (Rotation, Affine) in train_transform to make the task harder.
# 2. Increased Regularization (higher WEIGHT_DECAY and DROP_PATH_RATE) to fight fast overfitting.
# 3. Added Learning Rate Warmup (5 epochs) for more stable convergence (a common best practice).
# 4. Renamed model to 'Nano' to reflect the smaller depths being used vs. 'Tiny'.
# 5. Increased BATCH_SIZE to 128.
# 6. Added from-scratch batched Score-CAM implementation for XAI.
# 7. Added visual plotting for confusion matrix (seaborn) and metrics (matplotlib).

import os
import random
from pathlib import Path
from tqdm import tqdm
import time
import copy

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont # For plotting heatmaps
import matplotlib.pyplot as plt
import seaborn as sns # For plotting confusion matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR # Added for warmup

# ----------------------------
#  User settings / hyperparams
# ----------------------------
DATA_ROOT = "/kaggle/input/rice-image-dataset/Rice_Image_Dataset"  # user-provided
OUTPUT_DIR = "/kaggle/working/rice_convnextv2_robust"
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 128 
NUM_WORKERS = 4
EPOCHS = 30
WARMUP_EPOCHS = 5 # Added warmup
LR = 3e-4
WEIGHT_DECAY = 5e-5 # Increased from 1e-5
DROP_PATH_RATE = 0.2 # Increased from 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_SIZE = 0.10   # 10% holdout test
VAL_SIZE = 0.18   
PRINT_EVERY = 1

# XAI Score-CAM settings
SCORE_CAM_TARGET_LAYER_NAME = 'stages.3' # The last stage: model.stages[3]
NUM_XAI_IMAGES = 5 # Number of test images to generate heatmaps for
XAI_BATCH_SIZE = 64 # Batch size for running Score-CAM forward passes

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
#  Deterministic seeds
# ----------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Make CUDA deterministic (can make training slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# ----------------------------
#  CBAM implementation 
# ----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=True)
        )
    def forward(self, x):  # x: (B, C, H, W)
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)  # (B, C)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        scale = torch.sigmoid(out).view(b, c, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
    def forward(self, x):
        # x: (B, C, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg, maxv], dim=1)  # (B,2,H,W)
        attn = torch.sigmoid(self.conv(cat))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

# ----------------------------
#  Global Response Normalization (GRN) 
# ----------------------------
class GRN(nn.Module):
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
    def forward(self, x):
        # x: (B, C, H, W)
        gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)  # (B, C, 1, 1)
        gx_mean = gx.mean(dim=1, keepdim=True)  # (B,1,1,1)
        nx = gx / (gx_mean + self.eps)  # normalized global response (B,C,1,1)
        return self.gamma * (x * nx) + self.beta

# ----------------------------
#  ConvNeXtV2-like block 
# ----------------------------
class LayerNormChannelLast(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(channels, eps=eps)
    def forward(self, x):
        # x: (B, C, H, W) -> convert to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim, drop_path=0.0, expansion=4, kernel_size=7):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)  # depthwise
        self.norm = LayerNormChannelLast(dim)
        self.pw1 = nn.Conv2d(dim, dim * expansion, kernel_size=1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim * expansion, dim, kernel_size=1)
        self.grn = GRN(dim)
        self.gamma = nn.Parameter(torch.ones(dim))  # residual scaling, like ConvNeXtV2 uses small learnable scale
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.grn(x)
        # apply scale per-channel
        x = x * self.gamma.view(1, -1, 1, 1)
        x = self.drop_path(x) + residual
        return x

# ----------------------------
#  DropPath (stochastic depth)
# ----------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x.div(keep_prob) * random_tensor
        return output

# ----------------------------
#  Model Definition
#  Renamed to 'Nano' to reflect the depths: [2,2,6,2]
# ----------------------------
class ConvNeXtV2NanoCBAM(nn.Module):
    def __init__(self, in_chans=3, num_classes=5, depths=[2,2,6,2], dims=[48,96,192,384], drop_path_rate=0.0):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNormChannelLast(dims[0])
        )
        # Stages
        total_blocks = sum(depths)
        cur = 0
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        for i in range(len(depths)):
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtV2Block(dim=dims[i], drop_path=dp_rates[cur + j]))
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))
            # downsample between stages (except after last)
            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    LayerNormChannelLast(dims[i]),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                ))
            else:
                self.downsamples.append(nn.Identity())

        self.cbams = nn.ModuleList([CBAM(ch) for ch in dims])
        # head
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        # simple weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B,3,H,W)
        x = self.stem(x)  # downsample by 4
        for i, stage in enumerate(self.stages):
            x = stage(x)  # stage processing
            x = self.cbams[i](x) # Apply CBAM after each stage
            if i < len(self.stages) - 1:
                x = self.downsamples[i](x)
        # global pooling
        x = x.mean([-2, -1])  # (B, C)
        x = self.norm(x)
        x = self.head(x)
        return x

# ----------------------------
#  XAI: Score-CAM Implementation
# ----------------------------

# Simple hook class to store the output of a layer
class Hook:
    def __init__(self):
        self.output = None
        self.handle = None

    def hook_fn(self, module, input, output):
        self.output = output

    def clear(self):
        self.output = None

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None

# Function to un-normalize an image tensor for plotting
def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# Min-max normalization for a batch of feature maps
def normalize_cam_maps(maps):
    # maps shape: (C, H, W)
    C, H, W = maps.shape
    maps_flat = maps.view(C, -1)
    min_vals = maps_flat.min(dim=1, keepdim=True)[0]
    max_vals = maps_flat.max(dim=1, keepdim=True)[0]
    range_vals = max_vals - min_vals + 1e-6
    
    maps_norm = (maps_flat - min_vals) / range_vals
    return maps_norm.view(C, H, W)

def generate_score_cam_batched(model, input_tensor, target_class, hook):
    """
    Generates Score-CAM heatmap using a batched approach for efficiency.
    
    Args:
        model: The trained model (in eval mode).
        input_tensor: Single image tensor (C, H, W), normalized.
        target_class: The class index (int) to generate the map for.
        hook: The Hook object registered on the target layer.
    """
    model.eval()
    hook.clear()
    
    # 1. Get feature maps (activations) from the target layer
    with torch.no_grad():
        # Do a single forward pass to trigger the hook
        _ = model(input_tensor.unsqueeze(0).to(DEVICE))
    
    if hook.output is None:
        raise ValueError("Hook did not capture any output. Check layer name.")
        
    # Squeeze batch dim, move to device: (C, H_feat, W_feat)
    activations = hook.output.squeeze(0).to(DEVICE)
    C, H_feat, W_feat = activations.shape

    # 2. Upsample all feature maps to image size
    # (C, H_feat, W_feat) -> (1, C, H_feat, W_feat) -> (1, C, H_img, W_img) -> (C, H_img, W_img)
    with torch.no_grad():
        activations_upsampled = F.interpolate(
            activations.unsqueeze(0),
            size=(IMG_SIZE, IMG_SIZE),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # 3. Normalize upsampled maps to [0, 1] to use as masks
    # (C, H_img, W_img)
    activations_norm = normalize_cam_maps(activations_upsampled)
    
    # 4. Create masked image batch
    # input_tensor: (3, H_img, W_img)
    # activations_norm: (C, H_img, W_img)
    # We want (C, 3, H_img, W_img)
    # (1, 3, H, W) * (C, 1, H, W) -> (C, 3, H, W) via broadcasting
    masked_batch = input_tensor.unsqueeze(0).to(DEVICE) * activations_norm.unsqueeze(1)
    
    # 5. Run forward pass on the masked batch to get scores
    all_outputs = []
    with torch.no_grad():
        # Split into mini-batches to avoid OOM on T4 if C is large
        for batch in torch.split(masked_batch, XAI_BATCH_SIZE):
            all_outputs.append(model(batch))
    
    # outputs: (C, num_classes)
    outputs = torch.cat(all_outputs)
    
    # 6. Get softmax scores for the target class
    # scores: (C,)
    scores = F.softmax(outputs, dim=1)[:, target_class]
    
    # 7. Compute weighted sum of *original* (non-upsampled) feature maps
    # weights (scores): (C,)
    # activations: (C, H_feat, W_feat)
    # We want: (H_feat, W_feat)
    # 'c,chw->hw' means: sum over 'c' (channels)
    with torch.no_grad():
        cam = torch.einsum('c,chw->hw', scores, activations)
    
    # 8. Apply ReLU
    cam = F.relu(cam)
    
    # 9. Normalize final heatmap to [0, 1] for visualization
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)
    
    # Upsample final heatmap to image size
    cam_final = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(IMG_SIZE, IMG_SIZE),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    return cam_final

# ----------------------------
#  Plotting Functions
# ----------------------------
def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Test Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved confusion matrix plot to {save_path}")

def plot_classification_report(report_df, save_path):
    plt.figure(figsize=(10, 6))
    # Plot precision, recall, f1-score for each class (exclude support and averages)
    report_df_plot = report_df.iloc[:-3][['precision', 'recall', 'f1-score']]
    report_df_plot.plot(kind='bar', rot=0, figsize=(12, 8))
    plt.title('Classification Report (Test Set)')
    plt.ylabel('Score')
    plt.grid(axis='y', linestyle='--')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved classification report plot to {save_path}")
    
def plot_xai_heatmaps(original_img_pil, heatmaps_dict, class_names, save_path):
    """Plots original image + 5 heatmaps (one for each class) in a grid."""
    num_classes = len(class_names)
    fig_height = (num_classes + 1) * 4  # 4 inches per row
    
    # a 2-column grid: (Original, Heatmap) and stack them
    # (num_classes+1) rows, 2 columns
    fig, axes = plt.subplots(num_classes + 1, 2, figsize=(10, fig_height), gridspec_kw={'width_ratios': [1, 1]})

    # --- Row 0: Original Image ---
    axes[0, 0].imshow(original_img_pil)
    axes[0, 0].set_title(f"Original Image (Idx {os.path.basename(save_path).split('_')[3]})")
    axes[0, 0].axis('off')
    # Hide the second column in the first row
    axes[0, 1].axis('off') 

    # --- Rows 1 to num_classes+1: Heatmaps ---
    for i, class_name in enumerate(class_names):
        row_idx = i + 1
        heatmap_np = heatmaps_dict[class_name]
        
        # Plot heatmap
        axes[row_idx, 0].imshow(heatmap_np, cmap='jet')
        axes[row_idx, 0].set_title(f"Heatmap for: {class_name}")
        axes[row_idx, 0].axis('off')
        
        # Plot overlay
        axes[row_idx, 1].imshow(original_img_pil)
        axes[row_idx, 1].imshow(heatmap_np, cmap='jet', alpha=0.5)
        axes[row_idx, 1].set_title(f"Overlay for: {class_name}")
        axes[row_idx, 1].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Score-CAM plots to {save_path}")


# ----------------------------
#  Data transforms and dataset split 
# ----------------------------

#  Stronger augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), # Added rotation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10), # Added affine transforms
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), # Slightly stronger jitter
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.14)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load dataset 
base_dataset = datasets.ImageFolder(root=DATA_ROOT)
class_to_idx = base_dataset.class_to_idx
idx_to_class = {v:k for k,v in class_to_idx.items()}
num_classes = len(class_to_idx)
class_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())] # Get ordered list of names
print("Classes:", idx_to_class)

# Create deterministic stratified splits 
targets = [s[1] for s in base_dataset.samples]
indices = np.arange(len(base_dataset))

sss_test = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
trainval_idx, test_idx = next(sss_test.split(indices, targets))
remaining_targets = [targets[i] for i in trainval_idx]
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=SEED)
train_idx_rel, val_idx_rel = next(sss_val.split(trainval_idx, remaining_targets))
train_idx = trainval_idx[train_idx_rel]
val_idx = trainval_idx[val_idx_rel]
assert len(set(train_idx) & set(val_idx)) == 0
assert len(set(train_idx) & set(test_idx)) == 0
assert len(set(val_idx) & set(test_idx)) == 0

# Create Subsets and wrap transforms 
class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.dataset.samples[real_idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

train_ds = TransformedSubset(base_dataset, train_idx, train_transform)
val_ds = TransformedSubset(base_dataset, val_idx, val_transform)
test_ds = TransformedSubset(base_dataset, test_idx, val_transform)

# DataLoaders 
g = torch.Generator()
g.manual_seed(SEED)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, generator=g)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Train/Val/Test sizes: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

# ----------------------------
#  Model, criterion, optimizer, scheduler
# ----------------------------
model = ConvNeXtV2NanoCBAM(
    in_chans=3,
    num_classes=num_classes,
    depths=[2,2,6,2], 
    dims=[48,96,192,384], 
    drop_path_rate=DROP_PATH_RATE 
)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
# Using new, higher WEIGHT_DECAY
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

#  warmup scheduler 
# Main scheduler: Cosine Annealing for the main part of training
main_scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS)
# Warmup scheduler: Linear warmup for the first few epochs
warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_EPOCHS)
# Sequential scheduler: Run warmup, then switch to main scheduler
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])
print(f"Using {WARMUP_EPOCHS} epochs of linear LR warmup.")

# ----------------------------
#  Training and validation loops 
# ----------------------------
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=running_loss / (len(all_targets) + 1e-12))
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val/Test", leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            pbar.set_postfix(loss=running_loss / (len(all_targets) + 1e-12))
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds

# ----------------------------
#  Main training loop 
# ----------------------------
best_val_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'lr':[]}

print("Starting training...")
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    val_loss, val_acc, val_targs, val_preds = eval_model(model, val_loader, criterion, DEVICE)
    
    # Get current LR and step the scheduler
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    
    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save({'model_state_dict': best_model_wts, 'idx_to_class': idx_to_class}, os.path.join(OUTPUT_DIR, "best_model.pth"))
        
    if epoch % PRINT_EVERY == 0:
        print(f"Epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}  LR={current_lr:.1e}  time={(time.time()-t0):.1f}s")

print("Training finished.")

# Save final training curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend(); plt.title('Loss')
plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend(); plt.title('Accuracy')
plt.suptitle('Training and Validation Metrics')
plt.savefig(os.path.join(OUTPUT_DIR, "loss_acc_curves.png"))
plt.close()

# Save LR curve
plt.figure()
plt.plot(history['lr'], label='Learning Rate')
plt.legend(); plt.title('Learning Rate Schedule')
plt.savefig(os.path.join(OUTPUT_DIR, "lr_curve.png"))
plt.close()

# ----------------------------
#  Final test evaluation 
# ----------------------------
print("Loading best model for final test evaluation...")
model.load_state_dict(best_model_wts)
test_loss, test_acc, test_targets, test_preds = eval_model(model, test_loader, criterion, DEVICE)
print(f"Final Test loss: {test_loss:.4f}  Test acc: {test_acc:.4f}")

# Save classification report & confusion matrix 
report = classification_report(test_targets, test_preds, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(OUTPUT_DIR, "test_classification_report.csv"))
cm = confusion_matrix(test_targets, test_preds)
np.save(os.path.join(OUTPUT_DIR, "test_confusion_matrix.npy"), cm)

#  Save visual plots for metrics
plot_confusion_matrix(cm, class_names, os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"))
plot_classification_report(report_df, os.path.join(OUTPUT_DIR, "test_metrics_barchart.png"))

# print textual report
print("Classification report (test):")
print(classification_report(test_targets, test_preds, target_names=class_names))
print("Confusion matrix:")
print(cm)


# ----------------------------
#  XAI: Score-CAM Generation
# ----------------------------
print(f"\nStarting Score-CAM generation for {NUM_XAI_IMAGES} test images...")

# Set up the hook
model.eval()
try:
    target_layer = dict(model.named_modules())[SCORE_CAM_TARGET_LAYER_NAME]
    hook = Hook()
    hook.handle = target_layer.register_forward_hook(hook.hook_fn)
    print(f"Hook registered on layer: {SCORE_CAM_TARGET_LAYER_NAME}")

    # Get a batch of test images
    xai_images, xai_labels = next(iter(test_loader))
    xai_images = xai_images[:NUM_XAI_IMAGES]
    xai_labels = xai_labels[:NUM_XAI_IMAGES]
    
    # Denormalize transform for plotting
    inv_normalize = transforms.Normalize(
       mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
       std=[1/0.229, 1/0.224, 1/0.225]
    )

    for i in range(NUM_XAI_IMAGES):
        img_tensor = xai_images[i] # (3, H, W)
        true_label_idx = xai_labels[i].item()
        true_label_name = idx_to_class[true_label_idx]
        
        print(f"  Generating maps for image {i} (True Label: {true_label_name})...")
        
        heatmaps = {}
        for c in range(num_classes):
            class_name = idx_to_class[c]
            # Generate the heatmap for this image and this class
            heatmap_np = generate_score_cam_batched(model, img_tensor, c, hook)
            heatmaps[class_name] = heatmap_np
            
        # Denormalize original image for plotting
        original_img_pil = transforms.ToPILImage()(inv_normalize(img_tensor.cpu()))
        
        # Plot and save all 5 heatmaps for this one image
        save_name = f"score_cam_img_{i}_true_{true_label_name}.png"
        plot_xai_heatmaps(
            original_img_pil,
            heatmaps,
            class_names,
            os.path.join(OUTPUT_DIR, save_name)
        )

    # Clean up the hook
    hook.remove()
    print("Hook removed.")

except Exception as e:
    print(f"Could not generate Score-CAM heatmaps. Error: {e}")
    if 'hook' in locals() and hook.handle:
        hook.remove()
        print("Hook removed after error.")


print("All done. Outputs saved in:", OUTPUT_DIR)