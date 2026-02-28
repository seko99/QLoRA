#!/usr/bin/env bash
set -euo pipefail

INPUT_GGUF="gguf/model.f16.gguf"
OUTPUT_GGUF="gguf/model.Q4_K_M.gguf"
QUANT="Q4_K_M"
QUANTIZE_BIN="../llama.cpp/build/bin/llama-quantize"

usage() {
  cat <<'EOF'
Usage: scripts/quantize.sh [options]

Options:
  --input PATH         Source GGUF file (default: gguf/model.f16.gguf)
  --output PATH        Quantized GGUF file (default: gguf/model.Q4_K_M.gguf)
  --quant TYPE         Quantization type (default: Q4_K_M)
  --quantize-bin PATH  Path to llama-quantize binary (default: ./llama.cpp/build/bin/llama-quantize)
  -h, --help           Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input)
      INPUT_GGUF="$2"
      shift 2
      ;;
    --output)
      OUTPUT_GGUF="$2"
      shift 2
      ;;
    --quant)
      QUANT="$2"
      shift 2
      ;;
    --quantize-bin)
      QUANTIZE_BIN="$2"
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

mkdir -p "$(dirname "$OUTPUT_GGUF")"
"$QUANTIZE_BIN" "$INPUT_GGUF" "$OUTPUT_GGUF" "$QUANT"
