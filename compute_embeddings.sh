#!/bin/bash
# Compute embeddings for HuggingFace dataset or YOLO-style image directory using SSCD model.
# Usage: ./compute_embeddings.sh [options]
#   --dataset      HuggingFace dataset path or name (use with HF datasets)
#   --images_dir   YOLO-style directory (train/, valid/, etc. or data.yaml); alternative to --dataset
#   --output_dir   Output directory for embeddings
#   --batch_size   Batch size (default: 128)
#   --split        Split to process: val, train, test, or all (default: all)
#   --name         Dataset subset name
#   --deduplicate  Enable deduplication by image_id
#   --id_column    Column for image ID (default: image_id)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (YOLO dataset root; use --dataset for HuggingFace datasets)
IMAGES_DIR="${IMAGES_DIR:-/mnt/data/datasets/roboflow/organized_by_class/remapped/normal_abnormal/yolo-dataset-cls}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/datasets/roboflow/organized_by_class/remapped/normal_abnormal/yolo-dataset-cls/embeddings}"
BATCH_SIZE="${BATCH_SIZE:-128}"
SPLIT="${SPLIT:-all}"

# Parse optional overrides from arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      IMAGES_DIR=""  # dataset takes precedence
      shift 2
      ;;
    --images_dir)
      IMAGES_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --name)
      NAME="$2"
      shift 2
      ;;
    --deduplicate)
      DEDUPLICATE=1
      shift
      ;;
    --id_column)
      ID_COLUMN="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

ARGS=(--output_dir "$OUTPUT_DIR" --batch_size "$BATCH_SIZE" --split "$SPLIT")
if [[ -n "${IMAGES_DIR:-}" ]]; then
  ARGS+=(--images_dir "$IMAGES_DIR")
else
  ARGS+=(--dataset "$DATASET")
fi
[[ -n "${NAME:-}" ]] && ARGS+=(--name "$NAME")
[[ -n "${DEDUPLICATE:-}" ]] && ARGS+=(--deduplicate)
[[ -n "${ID_COLUMN:-}" ]] && ARGS+=(--id_column "$ID_COLUMN")
python3 "${SCRIPT_DIR}/compute_embeddings.py" "${ARGS[@]}" "$@"
