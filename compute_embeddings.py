import torch
import numpy as np
import argparse
import time
from pathlib import Path
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
import os
import yaml
from PIL import Image, ImageFile

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_data_yaml(yaml_path: Path) -> dict:
    """Load and parse data.yaml. Returns {} on error or empty file."""
    try:
        with open(yaml_path) as f:
            out = yaml.safe_load(f)
            return out if isinstance(out, dict) else {}
    except Exception:
        return {}


def _find_yolo_split_dirs(dataset_root: Path, data: dict) -> dict[str, Path]:
    """Resolve split dirs from data.yaml or infer from folder structure."""
    if data:
        splits = {}
        for key in ("train", "val", "valid", "test"):
            if key not in data:
                continue
            raw = data[key]
            raw_path = raw[0] if isinstance(raw, list) else raw
            resolved = (dataset_root / raw_path).resolve()
            split_name = "valid" if key == "val" else key
            if resolved.exists():
                split_dir = resolved.parent if resolved.name == "images" else resolved
                splits[split_name] = split_dir
            else:
                fallback = dataset_root / split_name
                if fallback.exists():
                    splits[split_name] = fallback
        if splits:
            return splits
    # Infer: train/, valid/, val/, test/
    splits = {}
    for name in ("train", "valid", "val", "test"):
        d = dataset_root / name
        if d.exists():
            splits["valid" if name == "val" else name] = d
    return splits


def _collect_images_from_split(split_dir: Path) -> list[tuple[str, str]]:
    """
    Collect (image_path, image_id) from a split dir.
    Supports DET: split_dir/images/; CLS: split_dir/classname/ or split_dir/classname/images/.
    """
    out = []
    images_dir = split_dir / "images"
    if images_dir.exists():
        # DET: flat images in images/
        search_dirs = [images_dir]
    else:
        # CLS: subdirs per class, or flat in split_dir
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if subdirs:
            search_dirs = []
            for sd in subdirs:
                img_sub = sd / "images"
                search_dirs.append(img_sub if img_sub.exists() else sd)
        else:
            search_dirs = [split_dir]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for entry in os.scandir(search_dir):
            if not entry.is_file():
                continue
            if not any(entry.name.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                continue
            stem = Path(entry.path).stem
            out.append((str(Path(entry.path).resolve()), stem))
    return out


def _load_yolo_image_paths(images_dir: str, split: str) -> dict[str, list[tuple[str, str]]]:
    """
    Load YOLO-style dataset: returns {split_name: [(path, image_id), ...]}.
    """
    root = Path(images_dir).resolve()
    data_yaml = root / "data.yaml"
    data = _load_data_yaml(data_yaml) if data_yaml.exists() else {}
    if not data and (root / "train").exists():
        pass  # _find_yolo_split_dirs will infer
    split_dirs = _find_yolo_split_dirs(root, data)
    if not split_dirs:
        # Single dir: treat root as one split
        split_dirs = {"train": root}
    split_name = "valid" if split == "val" else split
    if split != "all":
        if split_name not in split_dirs:
            raise ValueError(f"Split '{split}' not found. Available: {list(split_dirs.keys())}")
        split_dirs = {split_name: split_dirs[split_name]}
    result = {}
    for name, sdir in split_dirs.items():
        pairs = _collect_images_from_split(sdir)
        if pairs:
            result[name] = pairs
    return result


class YOLOImageDataset(TorchDataset):
    """PyTorch Dataset for YOLO image paths; returns dict with image, dataset_idx, image_id."""

    def __init__(self, image_pairs: list[tuple[str, str]]):
        self.image_pairs = image_pairs

    def __len__(self) -> int:
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> dict:
        path, image_id = self.image_pairs[idx]
        img = Image.open(path).convert("RGB")
        return {"image": img, "dataset_idx": idx, "image_id": image_id}


class ImageCollator:
    """Collator class for processing image batches with transforms."""
    
    def __init__(self, transform, deduplicate=False, id_column='image_id'):
        self.transform = transform
        self.deduplicate = deduplicate
        self.id_column = id_column
        self.seen_ids = set() if deduplicate else None
    
    def _is_valid_image(self, img):
        """Check if the object is a valid image that can be processed."""
        if img is None:
            return False
        
        try:
            # Try to actually load/verify the image to catch truncated images early
            img.load()
            # Test if we can convert it to RGB
            _ = img.convert('RGB')
            return True
        except (OSError, IOError, Image.DecompressionBombError, Exception) as e:
            print(f"Warning: Invalid or corrupted image detected: {e}")
            return False
    
    def __call__(self, batch):
        images = []
        indices = []
        for item in batch:
            try:
                # Get the explicit dataset index we added
                dataset_idx = item.get('dataset_idx', None)
                if dataset_idx is None:
                    print(f"Warning: No dataset_idx found in item")
                    continue
                
                # Check for deduplication
                if self.deduplicate and self.id_column in item:
                    image_id = item[self.id_column]
                    if image_id in self.seen_ids:
                        continue
                    self.seen_ids.add(image_id)
                
                # Handle different image key patterns
                image_data = None
                collected_images = []
                
                if 'image' in item:
                    image_data = item['image']
                    if isinstance(image_data, list):
                        # Handle list of images
                        for img in image_data:
                            if self._is_valid_image(img):
                                collected_images.append(img)
                    else:
                        # Handle single image
                        if self._is_valid_image(image_data):
                            collected_images.append(image_data)
                elif 'images' in item:
                    image_data = item['images']
                    if isinstance(image_data, list):
                        for img in image_data:
                            if self._is_valid_image(img):
                                collected_images.append(img)
                    else:
                        if self._is_valid_image(image_data):
                            collected_images.append(image_data)
                elif any(key.startswith('image_') and key[6:].isdigit() for key in item.keys()):
                    # Handle numbered image keys like 'image_0', 'image_1', etc.
                    numbered_image_keys = [key for key in item.keys() if key.startswith('image_') and key[6:].isdigit()]
                    # Sort keys by number to maintain order
                    numbered_image_keys.sort(key=lambda x: int(x.split('_')[1]))
                    for key in numbered_image_keys:
                        img_data = item[key]
                        if self._is_valid_image(img_data):
                            collected_images.append(img_data)
                else:
                    print(f"Warning: No image keys found in item. Available keys: {list(item.keys())}")
                    continue
                
                # Process all collected images
                for img in collected_images:
                    if self._is_valid_image(img):
                        try:
                            processed_img = img.convert('RGB')
                            images.append(self.transform(processed_img))
                            indices.append(dataset_idx)
                        except Exception as img_error:
                            print(f"Error processing individual image: {img_error}")
                
            except Exception as e:
                print(f"Error processing item: {e}")
                
        if images:
            return torch.stack(images), indices
        return None, []


def setup_device():
    """Setup and return the appropriate device (GPU/CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def load_model(model_path="models/sscd_disc_mixup.torchscript.pt", device=None):
    """Load and setup the model."""
    model = torch.jit.load(model_path)
    model.eval()
    if device:
        model = model.to(device)
    return model


def create_transforms():
    """Create and return image transforms."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
    )
    return transforms.Compose([
        transforms.Resize([320, 320]),
        transforms.ToTensor(),
        normalize,
    ])


def compute_batch_embeddings(model, dataloader, device):
    """Compute embeddings for all batches and return results with timing info."""
    embeddings_list = []
    image_ids = []
    model_inference_time = 0.0
    
    with torch.no_grad():
        for batch_tensor, batch_indices in tqdm(dataloader, desc="Computing embeddings"):
            if batch_tensor is not None:
                batch_tensor = batch_tensor.to(device)
                start_model_time = time.time()
                embeddings = model(batch_tensor)
                end_model_time = time.time()
                model_inference_time += end_model_time - start_model_time
                embeddings_list.append(embeddings.cpu().numpy())
                image_ids.extend(batch_indices)
    
    return embeddings_list, image_ids, model_inference_time


def save_results(embeddings, image_ids, dataset_name, split, output_dir, name=None):
    """Save embeddings and image IDs to files."""
    os.makedirs(output_dir, exist_ok=True)
    sanitized_dataset_name = dataset_name.replace('/', '-')
    
    # Include name in filename if provided
    if name:
        sanitized_name = name.replace('/', '-')
        filename_base = f'{sanitized_dataset_name}_{sanitized_name}_{split}'
    else:
        filename_base = f'{sanitized_dataset_name}_{split}'
    
    np.save(os.path.join(output_dir, f'{filename_base}_embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, f'{filename_base}_image_ids.npy'), np.array(image_ids))


def print_results(embeddings, total_time, model_inference_time, output_dir):
    """Print timing and result statistics."""
    num_embeddings = len(embeddings)
    time_per_sample = total_time / num_embeddings if num_embeddings > 0 else 0
    model_time_per_sample = model_inference_time / num_embeddings if num_embeddings > 0 else 0
    
    print(f"Saved {num_embeddings} embeddings to {output_dir}/")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Total time: {total_time:.5f} seconds")
    print(f"Time per sample: {time_per_sample:.5f} seconds")
    print(f"Model inference time: {model_inference_time:.5f} seconds")
    print(f"Model time per sample: {model_time_per_sample:.5f} seconds")


def compute_embeddings(dataset_name, name=None, split='test', output_dir='embeddings-lmms', batch_size=32, deduplicate=False, id_column='image_id', images_dir=None):
    """Compute embeddings for HuggingFace dataset or YOLO-style image directory."""
    function_start_time = time.time()

    # Setup components
    device = setup_device()
    model = load_model(device=device)
    transform = create_transforms()
    collator = ImageCollator(transform, deduplicate=deduplicate, id_column=id_column)

    if images_dir is not None:
        # YOLO-style directory
        split_data = _load_yolo_image_paths(images_dir, split)
        if not split_data:
            raise ValueError(f"No images found in {images_dir} for split '{split}'")
        display_name = str(Path(images_dir).name) or "yolo"
        for split_key, pairs in split_data.items():
            ds = YOLOImageDataset(pairs)
            dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collator, num_workers=8)
            start_time = time.time()
            embeddings_list, indices, model_inference_time = compute_batch_embeddings(model, dataloader, device)
            all_embeddings = np.vstack(embeddings_list)
            total_time = time.time() - start_time
            save_results(all_embeddings, indices, display_name, split_key, output_dir, name)
            print(f"\n[{split_key}] ", end="")
            print_results(all_embeddings, total_time, model_inference_time, output_dir)
        print(f"Total function time: {time.time() - function_start_time:.5f} seconds")
        return

    # Load HuggingFace dataset
    # Use load_from_disk for local paths (saved with save_to_disk); load_dataset for HF hub names
    if os.path.isdir(dataset_name):
        ds_dict = load_from_disk(dataset_name)
        splits_to_process = list(ds_dict.keys()) if split == "all" else ["valid" if split == "val" else split]
        if split != "all" and splits_to_process[0] not in ds_dict:
            raise ValueError(f"Split '{split}' (resolved: '{splits_to_process[0]}') not found. Available: {list(ds_dict.keys())}")
    else:
        if split == "all":
            ds = load_dataset(dataset_name, name=name)
            splits_to_process = list(ds.keys())
            ds_dict = ds
        else:
            dataset = load_dataset(dataset_name, name=name, split=split)
            dataset = dataset.add_column("dataset_idx", list(range(len(dataset))))
            dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=8)
            start_time = time.time()
            embeddings_list, image_ids, model_inference_time = compute_batch_embeddings(model, dataloader, device)
            all_embeddings = np.vstack(embeddings_list)
            save_results(all_embeddings, image_ids, dataset_name, split, output_dir, name)
            print_results(all_embeddings, time.time() - start_time, model_inference_time, output_dir)
            return
    
    for split_key in splits_to_process:
        dataset = ds_dict[split_key]
        dataset = dataset.add_column("dataset_idx", list(range(len(dataset))))
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=8)
        start_time = time.time()
        embeddings_list, image_ids, model_inference_time = compute_batch_embeddings(model, dataloader, device)
        all_embeddings = np.vstack(embeddings_list)
        total_time = time.time() - start_time
        save_results(all_embeddings, image_ids, dataset_name, split_key, output_dir, name)
        print(f"\n[{split_key}] ", end="")
        print_results(all_embeddings, total_time, model_inference_time, output_dir)
    
    function_end_time = time.time()
    print(f"Total function time: {function_end_time - function_start_time:.5f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings for HuggingFace dataset or YOLO-style image directory")
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name (required if not using --images_dir)")
    parser.add_argument("--images_dir", type=str, default=None, help="YOLO-style directory (train/, valid/, etc. or data.yaml); alternative to --dataset")
    parser.add_argument("--split", type=str, default="val", help="Split to process (val, train, test, or 'all' for all splits)")
    parser.add_argument("--name", type=str, default=None, help="Dataset (subset) name")
    parser.add_argument("--output_dir", type=str, default="embeddings", help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--deduplicate", action="store_true", help="Enable deduplication based on ID column")
    parser.add_argument("--id_column", type=str, default="image_id", help="Column name for image ID (default: image_id)")

    args = parser.parse_args()
    if args.images_dir:
        if not os.path.isdir(args.images_dir):
            raise SystemExit(f"Error: --images_dir path does not exist: {args.images_dir}")
        compute_embeddings(args.dataset or args.images_dir, args.name, args.split, args.output_dir, args.batch_size, args.deduplicate, args.id_column, images_dir=args.images_dir)
    elif args.dataset:
        compute_embeddings(args.dataset, args.name, args.split, args.output_dir, args.batch_size, args.deduplicate, args.id_column)
    else:
        raise SystemExit("Error: either --dataset or --images_dir is required") 