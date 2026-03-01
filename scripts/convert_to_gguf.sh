#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="merged_model"
OUTTYPE="f16"
OUTFILE="gguf/model.f16.gguf"
CONVERTER="../llama.cpp/convert_hf_to_gguf.py"

usage() {
  cat <<'EOF'
Usage: scripts/convert_to_gguf.sh [options]

Options:
  --input-dir PATH     HF merged model directory (default: merged_model)
  --outtype TYPE       GGUF outtype, e.g. f16/f32/bf16 (default: f16)
  --outfile PATH       Output GGUF file (default: gguf/model.f16.gguf)
  --converter PATH     Path to convert_hf_to_gguf.py (default: ../llama.cpp/convert_hf_to_gguf.py)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="$2"
      shift 2
      ;;
    --outtype)
      OUTTYPE="$2"
      shift 2
      ;;
    --outfile)
      OUTFILE="$2"
      shift 2
      ;;
    --converter)
      CONVERTER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "$(dirname "$OUTFILE")"
python "$CONVERTER" "$INPUT_DIR" \
  --outtype "$OUTTYPE" \
  --outfile "$OUTFILE"
