# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')

# --- CELL ---

!pip install timm albumentations imagehash grad-cam lime

# --- CELL ---

import os, cv2, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns; from tqdm import tqdm; from collections import defaultdict
sns.set_style("whitegrid")
DATASET_PATHS = {"Dataset_1": "/kaggle/input/datasets/alihussainkhan24/red-rot-sugarcane-disease-leaf-dataset", "Dataset_2": "/kaggle/input/datasets/nirmalsankalana/sugarcane-leaf-disease-dataset", "Dataset_3": "/kaggle/input/datasets/kamlesh11v/sugarcane-leaf-image-dataset", "Dataset_4": "/kaggle/input/datasets/shifatearman/sugarcaneld-bd-dataset"}
CLASS_MAPPING = {"healthy": "Healthy", "healthy leaves": "Healthy", "redrot": "RedRot", "red rot": "RedRot", "unhealthy": "RedRot", "mosaic": "Mosaic", "rust": "Rust", "brownrust": "Rust", "brown rust": "Rust", "yellow": "YellowLeaf", "yellow leaf": "YellowLeaf", "banded chlorosis": "BandedChlorosis", "brown spot": "BrownSpot", "grassy shoot": "GrassyShoot", "pokkah boeng": "PokkahBoeng", "sett rot": "SettRot", "smut": "Smut", "dried leaves": "DriedLeaves", "eyespot": "EyeSpot", "eye spot": "EyeSpot", "redleafspot": "RedLeafSpot", "red leaf spot": "RedLeafSpot", "ringspot": "RingSpot", "ring spot": "RingSpot"}
VALID_EXTENSIONS, IGNORED_FOLDERS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"), {"diseases", "viral disease"}

def normalize_class_name(class_name): return CLASS_MAPPING.get(class_name.lower().strip().replace("_", " ").replace("-", " "), class_name.title())
def add_bar_labels(ax):
    for p in ax.patches: ax.annotate(f"{int(p.get_width())}", (p.get_width(), p.get_y() + p.get_height() / 2), xytext=(5, 0), textcoords="offset points", va="center", fontsize=10, fontweight="bold")

all_image_paths, all_labels, dataset_class_counter, before_merge_counter, after_merge_counter = [], [], [], defaultdict(int), defaultdict(int)
for dataset_name, dataset_root in DATASET_PATHS.items():
    print(f"\nScanning {dataset_name}")
    for root, dirs, files in os.walk(dataset_root):
        if not (image_files := [f for f in files if f.lower().endswith(VALID_EXTENSIONS)]): continue
        if (folder_name := os.path.basename(root).lower().strip()) in IGNORED_FOLDERS: print(f"IGNORED FOLDER → Dataset: {dataset_name} | Folder: {folder_name}"); continue
        merged_class, image_count = normalize_class_name(folder_name), 0
        for file in tqdm(image_files, desc=folder_name, leave=False):
            if cv2.imread(image_path := os.path.join(root, file)) is None: continue
            all_image_paths.append(image_path); all_labels.append(merged_class); before_merge_counter[folder_name] += 1; after_merge_counter[merged_class] += 1; image_count += 1
        dataset_class_counter.append([dataset_name, folder_name, merged_class, image_count])

df = pd.DataFrame({"path": all_image_paths, "label": all_labels})
dataset_df = pd.DataFrame(dataset_class_counter, columns=["Dataset", "Original Class", "Merged Class", "Image Count"])
before_df = pd.DataFrame(before_merge_counter.items(), columns=["Original Class", "Image Count"]).sort_values(by="Image Count", ascending=False)
after_df = pd.DataFrame(after_merge_counter.items(), columns=["Merged Class", "Image Count"]).sort_values(by="Image Count", ascending=False)
total_images, total_classes = len(df), df["label"].nunique()
summary_df = pd.DataFrame({"Metric": ["Total Images", "Total Classes"], "Count": [total_images, total_classes]})

plt.figure(figsize=(18, 10)); ax = sns.barplot(data=dataset_df, x="Image Count", y="Original Class", hue="Dataset"); add_bar_labels(ax); plt.title("Dataset-wise Folder Image Count", fontsize=18, fontweight="bold"); plt.xlabel("Number of Images"); plt.ylabel("Folder Name"); plt.legend(title="Dataset"); plt.show()
fig, ax = plt.subplots(1, 2, figsize=(22, 10))
sns.barplot(data=before_df, x="Image Count", y="Original Class", ax=ax[0]); add_bar_labels(ax[0]); ax[0].set_title("Before Class Merge", fontsize=16, fontweight="bold")
sns.barplot(data=after_df, x="Image Count", y="Merged Class", ax=ax[1]); add_bar_labels(ax[1]); ax[1].set_title("After Class Merge", fontsize=16, fontweight="bold")
plt.tight_layout(); plt.show()
plt.figure(figsize=(12, 12)); plt.pie(after_df["Image Count"], labels=after_df["Merged Class"], autopct="%1.1f%%"); plt.title("Final Dataset Class Distribution", fontsize=18, fontweight="bold"); plt.show()
plt.figure(figsize=(8, 5)); ax = sns.barplot(data=summary_df, x="Metric", y="Count")
for p in ax.patches: ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.title("Dataset Summary", fontsize=18, fontweight="bold"); plt.show()
print(f"FINAL DATASET REPORT\nTotal Images  : {total_images:,}\nTotal Classes : {total_classes}\n\nFinal Classes:\n{chr(10).join(sorted(df['label'].unique()))}")

# --- CELL ---

import random; ANALYSIS_SAMPLE_SIZE = 3000
analysis_df = df.sample(min(ANALYSIS_SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
image_widths, image_heights, aspect_ratios, brightness_scores, r_pixels, g_pixels, b_pixels, corrupted_images = [], [], [], [], [], [], [], []
print(f"Images Used For Analysis : {len(analysis_df):,}")

for image_path in tqdm(analysis_df["path"], desc="Analyzing Images"):
    try:
        if (image := cv2.imread(image_path)) is None: corrupted_images.append(image_path); continue
        h, w = image.shape[:2]; image_heights.append(h); image_widths.append(w); aspect_ratios.append(round(w / h, 2))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB); sample_size = min(500, image_rgb.shape[0] * image_rgb.shape[1])
        r_pixels.extend(np.random.choice(image_rgb[:, :, 0].flatten(), sample_size, replace=False))
        g_pixels.extend(np.random.choice(image_rgb[:, :, 1].flatten(), sample_size, replace=False))
        b_pixels.extend(np.random.choice(image_rgb[:, :, 2].flatten(), sample_size, replace=False))
        brightness_scores.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    except Exception: corrupted_images.append(image_path)

summary_df = pd.DataFrame({"Metric": ["Total Images", "Images Used For Analysis", "Corrupted Images", "Average Width", "Average Height", "Average Brightness"], "Value": [len(df), len(analysis_df), len(corrupted_images), round(np.mean(image_widths), 2), round(np.mean(image_heights), 2), round(np.mean(brightness_scores), 2)]})
print("DATASET HEALTH REPORT"); display(summary_df)

unique_classes = sorted(df["label"].unique()); cols = 4; rows = int(np.ceil(len(unique_classes) / cols))
plt.figure(figsize=(20, rows * 5))
for idx, class_name in enumerate(unique_classes):
    if not (class_paths := df[df["label"] == class_name]["path"].tolist()): continue
    image = cv2.cvtColor(cv2.imread(random.choice(class_paths)), cv2.COLOR_BGR2RGB)
    plt.subplot(rows, cols, idx + 1); plt.imshow(image); plt.title(class_name, fontsize=13, fontweight="bold"); plt.axis("off")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(18, 6))
sns.histplot(image_widths, bins=30, kde=True, ax=ax[0]); ax[0].set_title("Image Width Distribution"); ax[0].set_xlabel("Width")
sns.histplot(image_heights, bins=30, kde=True, ax=ax[1]); ax[1].set_title("Image Height Distribution"); ax[1].set_xlabel("Height"); plt.show()
plt.figure(figsize=(14, 6)); sns.kdeplot(r_pixels, label="Red"); sns.kdeplot(g_pixels, label="Green"); sns.kdeplot(b_pixels, label="Blue"); plt.title("RGB Channel Distribution"); plt.xlabel("Pixel Intensity"); plt.ylabel("Density"); plt.legend(); plt.show()
plt.figure(figsize=(12, 6)); sns.histplot(brightness_scores, bins=30, kde=True); plt.title("Brightness Distribution"); plt.xlabel("Brightness"); plt.show()

# --- CELL ---

import hashlib, imagehash; from PIL import Image
raw_df = df.copy().reset_index(drop=True); PERCEPTUAL_HASH_THRESHOLD, phash_cache, duplicate_exact, duplicate_perceptual, conflicts, corrupted = 3, {}, [], [], [], []

def get_exact_hash(path):
    try: return hashlib.md5(open(path, "rb").read()).hexdigest()
    except Exception: return None

def get_phash(path):
    if path in phash_cache: return phash_cache[path]
    try: h = imagehash.phash(Image.open(path).convert("RGB")); phash_cache[path] = h; return h
    except Exception: phash_cache[path] = None; return None

valid_idx = []
for idx, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Verifying Images"):
    try: Image.open(row["path"]).verify(); Image.open(row["path"]).convert("RGB"); valid_idx.append(idx)
    except Exception: corrupted.append(row["path"])

raw_df = raw_df.iloc[valid_idx].reset_index(drop=True); original_valid = len(raw_df); exact_map, exact_keep = defaultdict(list), []
for idx, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Hashing"):
    if (eh := get_exact_hash(row["path"])) and (ph := get_phash(row["path"])): exact_map[eh].append({"index": idx, "path": row["path"], "label": row["label"], "phash": ph})

for eh, recs in tqdm(exact_map.items(), desc="Exact Duplicates"):
    if len(set(r["label"] for r in recs)) > 1: conflicts.extend([{"path": r["path"], "label": r["label"], "hash": eh, "type": "Exact Conflict"} for r in recs]); continue
    exact_keep.append(recs[0]["index"])
    if len(recs) > 1: duplicate_exact.extend([{"path": r["path"], "label": r["label"], "hash": eh} for r in recs[1:]])

clean_df = raw_df.iloc[sorted(exact_keep)].reset_index(drop=True); phash_map, cross_paths = defaultdict(list), set()
for idx, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Global pHash"):
    if p := phash_cache.get(row["path"]): phash_map[str(p)].append({"index": idx, "path": row["path"], "label": row["label"]})

for ph_str, recs in tqdm(phash_map.items(), desc="Cross-Label Conflicts"):
    if len(recs) > 1 and len(set(r["label"] for r in recs)) > 1:
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i]["label"] != recs[j]["label"]: conflicts.append({"path": recs[j]["path"], "label": recs[j]["label"], "conflicting_path": recs[i]["path"], "conflicting_label": recs[i]["label"], "type": "Cross-label Conflict"}); cross_paths.update([recs[i]["path"], recs[j]["path"]])

clean_df = clean_df[~clean_df["path"].isin(cross_paths)].reset_index(drop=True); p_groups, final_keep = defaultdict(list), []
for idx, row in tqdm(clean_df.iterrows(), total=len(clean_df), desc="Intra-Class Grouping"):
    if p := phash_cache.get(row["path"]): p_groups[row["label"]].append({"index": idx, "path": row["path"], "label": row["label"], "phash": p})

for label, recs in tqdm(p_groups.items(), desc="Intra-Class Duplicates"):
    keep_mask = [True] * len(recs)
    for i in range(len(recs)):
        if not keep_mask[i]: continue
        for j in range(i + 1, len(recs)):
            if keep_mask[j] and recs[i]["phash"] - recs[j]["phash"] <= PERCEPTUAL_HASH_THRESHOLD: duplicate_perceptual.append({"kept": recs[i]["path"], "removed": recs[j]["path"], "label": recs[j]["label"], "sim": recs[i]["phash"] - recs[j]["phash"]}); keep_mask[j] = False
    final_keep.extend(recs[k]["index"] for k, keep in enumerate(keep_mask) if keep)

df = clean_df.iloc[sorted(final_keep)].reset_index(drop=True)
print(f"\nCLEANING REPORT\nOriginal Valid Images: {original_valid:,}\nCorrupted Removed: {len(corrupted):,}\nExact Duplicates Removed: {len(duplicate_exact):,}\nCross-Label Conflicts Removed: {len(cross_paths):,}\nIntra-Class Duplicates Removed: {len(duplicate_perceptual):,}\nFinal Clean Images: {len(df):,}")

# --- CELL ---

print("VISUAL PROOF : CROSS-LABEL CONFLICTS")
if cross_conflicts := [c for c in conflicts if c.get("type") == "Cross-label Conflict"]:
    samples = random.sample(cross_conflicts, min(5, len(cross_conflicts)))
    fig, axes = plt.subplots(len(samples), 2, figsize=(12, len(samples) * 4))
    if len(samples) == 1: axes = [axes]
    for i, ex in enumerate(samples):
        img_a, img_b = cv2.cvtColor(cv2.imread(ex["conflicting_path"]), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(ex["path"]), cv2.COLOR_BGR2RGB)
        axes[i][0].imshow(img_a); axes[i][0].axis("off"); axes[i][0].set_title(f"Folder: {ex['conflicting_label']}\n{os.path.basename(ex['conflicting_path'])}", color="darkred", fontweight="bold")
        axes[i][1].imshow(img_b); axes[i][1].axis("off"); axes[i][1].set_title(f"Folder: {ex['label']}\n{os.path.basename(ex['path'])}", color="darkred", fontweight="bold")
    plt.suptitle("Cross-Label Conflicts (Same Image, Different Classes)", fontsize=14, fontweight="bold", y=1.01); plt.tight_layout(); plt.show()
else: print("No Cross-Label Conflicts found to display.")

print("\nVISUAL PROOF : INTRA-CLASS PERCEPTUAL DUPLICATES")
if duplicate_perceptual:
    samples = random.sample(duplicate_perceptual, min(5, len(duplicate_perceptual)))
    fig, axes = plt.subplots(len(samples), 2, figsize=(12, len(samples) * 4))
    if len(samples) == 1: axes = [axes]
    for i, ex in enumerate(samples):
        img_kept, img_rem = cv2.cvtColor(cv2.imread(ex["kept"]), cv2.COLOR_BGR2RGB), cv2.cvtColor(cv2.imread(ex["removed"]), cv2.COLOR_BGR2RGB)
        axes[i][0].imshow(img_kept); axes[i][0].axis("off"); axes[i][0].set_title(f"Kept Image\n{os.path.basename(ex['kept'])}", fontweight="bold")
        axes[i][1].imshow(img_rem); axes[i][1].axis("off"); axes[i][1].set_title(f"Removed Duplicate (Distance: {ex['sim']})\n{os.path.basename(ex['removed'])}", color="darkorange", fontweight="bold")
    plt.suptitle("Same-Label Perceptual Duplicates", fontsize=14, fontweight="bold", y=1.01); plt.tight_layout(); plt.show()
else: print("No Intra-Class Duplicates found to display.")

# --- CELL ---

import torch, pickle; from sklearn.model_selection import train_test_split; from sklearn.preprocessing import LabelEncoder
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

df["label_encoded"] = (le := LabelEncoder()).fit_transform(df["label"]); class_names, NUM_CLASSES = list(le.classes_), len(le.classes_)
with open("label_encoder.pkl", "wb") as f: pickle.dump(le, f)

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_encoded"], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_encoded"], random_state=SEED)
train_df, val_df, test_df = train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

if any([o_tv := len(set(train_df["path"]) & set(val_df["path"])), o_tt := len(set(train_df["path"]) & set(test_df["path"])), o_vt := len(set(val_df["path"]) & set(test_df["path"]))]): raise ValueError("Leakage Detected!")
print(f"Split & Leakage Check Passed\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}\nTotal Classes: {NUM_CLASSES}")

# --- CELL ---

import albumentations as A; from albumentations.pytorch import ToTensorV2; from torch.utils.data import Dataset, DataLoader
IMG_SIZE, BATCH_SIZE = 224, 16

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.Affine(scale=(0.90, 1.10), translate_percent=(-0.04, 0.04), rotate=(-15, 15), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=0.08, p=0.5), A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()
])
val_transform = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])

class SugarcaneDataset(Dataset):
    def __init__(self, df, transform, is_train=False): self.df, self.transform, self.is_train = df, transform, is_train
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.df.iloc[idx]["path"]), cv2.COLOR_BGR2RGB); h, w = img.shape[:2]; c = min(h, w); y, x = (h - c) // 2, (w - c) // 2
        tensor = self.transform(image=cv2.resize(img[y:y+c, x:x+c], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC))["image"]
        if self.is_train: tensor = tensor + torch.randn_like(tensor) * 0.005
        return tensor, torch.tensor(self.df.iloc[idx]["label_encoded"], dtype=torch.long)

use_pin = torch.cuda.is_available()
train_loader = DataLoader(SugarcaneDataset(train_df, train_transform, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=use_pin)
val_loader = DataLoader(SugarcaneDataset(val_df, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=use_pin)
test_loader = DataLoader(SugarcaneDataset(test_df, val_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=use_pin)

def mixup_cutmix(x, y, alpha=0.2):
    if np.random.rand() > 0.5: return x, y, y, 1.0
    lam, idx = np.random.beta(alpha, alpha), torch.randperm(x.size(0)).to(x.device)
    if np.random.rand() > 0.5:
        cx, cy, w, h = np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE), int(IMG_SIZE * np.sqrt(1 - lam)), int(IMG_SIZE * np.sqrt(1 - lam))
        x1, y1, x2, y2 = max(cx - w // 2, 0), max(cy - h // 2, 0), min(cx + w // 2, IMG_SIZE), min(cy + h // 2, IMG_SIZE)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]; lam = 1.0 - ((x2 - x1) * (y2 - y1) / (IMG_SIZE * IMG_SIZE))
    else: x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], lam

# --- CELL ---

sample_imgs, sample_lbls = next(iter(train_loader))
plt.figure(figsize=(12, 12))
for i in range(9):
    img = np.clip(sample_imgs[i].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    plt.subplot(3, 3, i + 1); plt.imshow(img); plt.title(class_names[sample_lbls[i].item()], fontweight="bold"); plt.axis("off")
plt.suptitle("Sample Augmented Training Batch", fontsize=16, fontweight="bold", y=1.02); plt.tight_layout(); plt.show()

mixed_imgs, labels_a, labels_b, lam = mixup_cutmix(sample_imgs, sample_lbls, alpha=0.4) 
plt.figure(figsize=(12, 12))
for i in range(9):
    img = np.clip(mixed_imgs[i].permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    plt.subplot(3, 3, i + 1); plt.imshow(img); plt.axis("off")
    title = f"{class_names[labels_a[i]]} ({lam:.2f})\n+ {class_names[labels_b[i]]} ({1-lam:.2f})" if lam < 1.0 else f"{class_names[labels_a[i]]} (No Mix)"
    plt.title(title, fontweight="bold", fontsize=10)
plt.suptitle("MixUp & CutMix Applied on Training Batch", fontsize=16, fontweight="bold", y=1.02); plt.tight_layout(); plt.show()

# --- CELL ---

import timm, torch, torch.nn as nn, os, numpy as np; from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_df["label_encoded"]), y=train_df["label_encoded"])
class_weights_dict = {k: min(v, 3.0) for k, v in dict(enumerate(class_weights)).items()}
class_weights_tensor = torch.tensor(list(class_weights_dict.values()), dtype=torch.float32).to(device)

print("CLASS WEIGHT COMPUTATION")
for k, v in class_weights_dict.items(): print(f"Class {k} : {v:.4f}")

class HybridSEFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin = timm.create_model("swin_base_patch4_window7_224", pretrained=True, num_classes=0, global_pool="avg")
        self.densenet = timm.create_model("densenet121", pretrained=True, num_classes=0, global_pool="avg")
        for p in self.swin.parameters(): p.requires_grad = False
        for p in self.densenet.parameters(): p.requires_grad = False
        self.ln_swin, self.ln_dense = nn.LayerNorm(1024), nn.LayerNorm(1024)
        self.se = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Linear(128, 2048), nn.Sigmoid())
        self.head = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Dropout(0.40), nn.Linear(1024, 512), nn.GELU(), nn.Dropout(0.30), nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.20), nn.Linear(256, num_classes))
    
    def forward(self, x):
        f = torch.cat([self.ln_swin(self.swin(x)), self.ln_dense(self.densenet(x))], dim=1)
        return self.head(f * self.se(f))

model = HybridSEFusion(NUM_CLASSES).to(device)
os.makedirs("saved_models", exist_ok=True)
print(f"\nModel Ready\nDevice : {device}\nArchitecture : Swin+DenseNet121 w/ SE-Attention Fusion\nTrainable Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# --- CELL ---

import copy, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
EPOCHS_PHASE1, INITIAL_LR, WEIGHT_DECAY = 25, 1e-3, 1e-4

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
optimizer = optim.AdamW(model.head.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7)

best_val_acc, patience_counter, best_wts = 0.0, 0, copy.deepcopy(model.state_dict())
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

print("PHASE 1 : TRAINING START")
for epoch in range(EPOCHS_PHASE1):
    model.train(); run_loss, run_correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels_a, labels_b, lam = mixup_cutmix(inputs, labels, alpha=0.2)
        optimizer.zero_grad(); outputs = model(inputs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        run_loss += loss.item() * inputs.size(0); run_correct += (outputs.argmax(1) == labels_a).sum().item(); total += inputs.size(0)
    t_loss, t_acc = run_loss / total, run_correct / total

    model.eval(); v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            v_loss += loss.item() * inputs.size(0); v_correct += (outputs.argmax(1) == labels).sum().item(); v_total += inputs.size(0)
    v_loss, v_acc = v_loss / v_total, v_correct / v_total; scheduler.step(v_acc)
    history["train_acc"].append(t_acc); history["val_acc"].append(v_acc); history["train_loss"].append(t_loss); history["val_loss"].append(v_loss)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
    if v_acc > best_val_acc: best_val_acc, patience_counter, best_wts = v_acc, 0, copy.deepcopy(model.state_dict())
    else: patience_counter += 1
    if patience_counter >= 8: print("Early Stopping triggered."); break

model.load_state_dict(best_wts); torch.save(model.state_dict(), "saved_models/best_model_hybrid_swin_dense_phase1.pth")
print(f"\nPHASE 1 SUMMARY\nBest Val Acc: {max(history['val_acc']):.4f} | Best Val Loss: {min(history['val_loss']):.4f}\nModel Saved: saved_models/best_model_hybrid_swin_dense_phase1.pth\nPHASE 1 COMPLETE")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
epochs_range = range(1, len(history["train_acc"]) + 1)
ax[0].plot(epochs_range, history["train_acc"], label="Train Acc", color="royalblue", linewidth=2); ax[0].plot(epochs_range, history["val_acc"], label="Val Acc", color="darkorange", linewidth=2)
ax[0].set_title("Phase 1 : Accuracy", fontweight="bold"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_ylim(0, 1)
ax[1].plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", linewidth=2); ax[1].plot(epochs_range, history["val_loss"], label="Val Loss", color="darkorange", linewidth=2)
ax[1].set_title("Phase 1 : Loss", fontweight="bold"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.suptitle("Phase 1 : Hybrid Model Transfer Learning Curves", fontsize=16, fontweight="bold", y=1.02); plt.tight_layout(); plt.show()

# --- CELL ---

import copy, torch, torch.nn as nn, torch.optim as optim, matplotlib.pyplot as plt
FINE_TUNE_EPOCHS, FINE_TUNE_LR, UNFREEZE_PERCENTAGE = 20, 1e-5, 0.35

model.load_state_dict(torch.load("saved_models/best_model_hybrid_swin_dense_phase1.pth", map_location=device))
for base in [model.swin, model.densenet]:
    params = list(base.parameters())
    for idx, p in enumerate(params):
        if idx >= int(len(params) * (1 - UNFREEZE_PERCENTAGE)): p.requires_grad = True

print(f"PHASE 2 : FINE TUNING START\nTotal Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=FINE_TUNE_LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7)

best_val_acc, patience_counter, best_wts = 0.0, 0, copy.deepcopy(model.state_dict())
history_ft = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

for epoch in range(FINE_TUNE_EPOCHS):
    model.train(); run_loss, run_correct, total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs, labels_a, labels_b, lam = mixup_cutmix(inputs, labels, alpha=0.2)
        optimizer.zero_grad(); outputs = model(inputs)
        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        run_loss += loss.item() * inputs.size(0); run_correct += (outputs.argmax(1) == labels_a).sum().item(); total += inputs.size(0)
    t_loss, t_acc = run_loss / total, run_correct / total

    model.eval(); v_loss, v_correct, v_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs); loss = criterion(outputs, labels)
            v_loss += loss.item() * inputs.size(0); v_correct += (outputs.argmax(1) == labels).sum().item(); v_total += inputs.size(0)
    v_loss, v_acc = v_loss / v_total, v_correct / v_total; scheduler.step(v_acc)
    history_ft["train_acc"].append(t_acc); history_ft["val_acc"].append(v_acc); history_ft["train_loss"].append(t_loss); history_ft["val_loss"].append(v_loss)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
    if v_acc > best_val_acc: best_val_acc, patience_counter, best_wts = v_acc, 0, copy.deepcopy(model.state_dict())
    else: patience_counter += 1
    if patience_counter >= 7: print("Early Stopping triggered."); break

model.load_state_dict(best_wts); torch.save(model.state_dict(), "saved_models/fine_tuned_hybrid_swin_dense.pth")
print(f"\nPHASE 2 SUMMARY\nBest Val Acc: {max(history_ft['val_acc']):.4f} | Best Val Loss: {min(history_ft['val_loss']):.4f}\nModel Saved: saved_models/fine_tuned_hybrid_swin_dense.pth\nPHASE 2 COMPLETE")

fig, ax = plt.subplots(1, 2, figsize=(16, 5))
epochs_range = range(1, len(history_ft["train_acc"]) + 1)
ax[0].plot(epochs_range, history_ft["train_acc"], label="Train Acc", color="seagreen", linewidth=2); ax[0].plot(epochs_range, history_ft["val_acc"], label="Val Acc", color="crimson", linewidth=2)
ax[0].set_title("Phase 2 : Accuracy", fontweight="bold"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_ylim(0, 1)
ax[1].plot(epochs_range, history_ft["train_loss"], label="Train Loss", color="seagreen", linewidth=2); ax[1].plot(epochs_range, history_ft["val_loss"], label="Val Loss", color="crimson", linewidth=2)
ax[1].set_title("Phase 2 : Loss", fontweight="bold"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.suptitle("Phase 2 : Fine-Tuning Curves", fontsize=16, fontweight="bold", y=1.02); plt.tight_layout(); plt.show()

# --- CELL ---

import torch, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

model.load_state_dict(torch.load("saved_models/fine_tuned_hybrid_swin_dense.pth", map_location=device))
model.eval(); all_preds, all_labels = [], []

print("EVALUATING ON UNSEEN TEST SET...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        all_preds.extend(model(inputs).argmax(1).cpu().numpy()); all_labels.extend(labels.cpu().numpy())

print("\nCLASSIFICATION REPORT")
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, annot_kws={"fontsize": 11, "fontweight": "bold"}, cbar_kws={"shrink": 0.8})
plt.title("Test Set Confusion Matrix", fontsize=15, fontweight="bold", pad=15)
plt.xlabel("Predicted Class", fontsize=12, fontweight="bold"); plt.ylabel("True Class", fontsize=12, fontweight="bold")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0); plt.tight_layout(); plt.show()

# --- CELL ---

import cv2, torch, numpy as np, matplotlib.pyplot as plt, torch.nn.functional as F
from pytorch_grad_cam import GradCAMPlusPlus; from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image; from skimage.segmentation import mark_boundaries

random_classes = np.random.choice(class_names, 5, replace=False)
sample_paths = [test_df[test_df["label"] == cls].sample(1)["path"].values[0] for cls in random_classes]
sample_titles = [cls for cls in random_classes]
original_images, grad_cam_images, lime_images = [], [], []

model.eval()
cam = GradCAMPlusPlus(model=model, target_layers=[model.densenet.features])
explainer = lime_image.LimeImageExplainer()

def predict_fn(images):
    batch = torch.stack([val_transform(image=img)["image"] for img in images], dim=0).to(device)
    with torch.no_grad(): return F.softmax(model(batch), dim=1).cpu().numpy()

print("PROCESSING XAI MAPS FOR PAPER (Consolidated Grid)...")
for idx, (cls_name, img_path) in enumerate(zip(random_classes, sample_paths)):
    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB); h, w = img_rgb.shape[:2]; c = min(h, w)
    img_cropped = cv2.resize(img_rgb[(h-c)//2:(h-c)//2+c, (w-c)//2:(w-c)//2+c], (IMG_SIZE, IMG_SIZE))
    original_images.append(img_cropped)
    
    input_tensor = val_transform(image=img_cropped)["image"].unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    grad_cam_images.append(show_cam_on_image(img_cropped.astype(np.float32) / 255, grayscale_cam, use_rgb=True))
    
    explanation = explainer.explain_instance(img_cropped, predict_fn, top_labels=1, hide_color=0, num_samples=5000, random_seed=42)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_images.append(mark_boundaries(temp / 255.0, mask))

fig, axes = plt.subplots(3, 5, figsize=(18, 12))
fig.suptitle("Comparative Explainability Analysis on Plant Disease Classes", fontsize=20, fontweight="bold", y=1.03)

for i in range(5):
    axes[0, i].set_title(sample_titles[i], fontsize=14, fontweight="bold")
    axes[0, i].imshow(original_images[i]); axes[0, i].axis("off")
    axes[1, i].imshow(grad_cam_images[i]); axes[1, i].axis("off")
    axes[2, i].imshow(lime_images[i]); axes[2, i].axis("off")

fig.text(0.01, 0.83, 'Row 1: Original Images', va='center', rotation='vertical', fontsize=12, fontweight="bold")
fig.text(0.01, 0.50, 'Row 2: Grad-CAM++ Attention', va='center', rotation='vertical', fontsize=12, fontweight="bold")
fig.text(0.01, 0.17, 'Row 3: LIME Explanations', va='center', rotation='vertical', fontsize=12, fontweight="bold")
plt.tight_layout(pad=3.0); plt.show()

# --- CELL ---

import seaborn as sns; from sklearn.manifold import TSNE

features, labels = [], []
activation = {}
def get_activation(name):
    def hook(module, input, output): activation[name] = output.detach()
    return hook

model.head[-3].register_forward_hook(get_activation("features"))

print("EXTRACTING FEATURES FOR t-SNE...")
model.eval()
with torch.no_grad():
    for inputs, lbls in test_loader:
        _ = model(inputs.to(device))
        features.extend(activation["features"].cpu().numpy()); labels.extend(lbls.numpy())

tsne_results = TSNE(n_components=2, perplexity=30, learning_rate="auto", random_state=42).fit_transform(np.array(features))

plt.figure(figsize=(14, 10))
sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=[class_names[l] for l in labels], palette="tab20", s=60, alpha=0.9, edgecolor="k")
plt.title("t-SNE Visualization", fontsize=16, fontweight="bold")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left"); plt.tight_layout(); plt.show()