#!/usr/bin/env bash
set -euo pipefail

# Colab TPU v5e setup helper for w2v-bert-tpu.
# Preserves TorchTPU + libtpu 0.0.50 compatibility by pinning jax/jaxlib 0.11.2
# when libtpu < 0.0.53 is present, and installs w2v-bert-tpu in editable mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== Freeing stale TPU holders if present ==="
pkill -9 -f '[c]olab_kernel_launcher' || true
sleep 1

echo "=== Checking libtpu / JAX compatibility ==="
LIBTPU_VER="$(python3 -c 'import importlib.metadata as m; print(m.version("libtpu"))' 2>/dev/null || echo 'none')"
JAX_VER="$(python3 -c 'import importlib.metadata as m; print(m.version("jax"))' 2>/dev/null || echo 'none')"
echo "libtpu=${LIBTPU_VER}, jax=${JAX_VER}"

if [[ "${LIBTPU_VER}" == "0.0.50" && "${JAX_VER}" != "0.11.2" ]]; then
  echo "Restoring jax==0.11.2 jaxlib==0.11.2 for libtpu 0.0.50 Pallas TPU lowering compatibility..."
  python3 -m pip install -q "jax==0.11.2" "jaxlib==0.11.2"
fi

echo "=== Installing w2v-bert-tpu (editable) ==="
python3 -m pip install -q -e "${REPO_ROOT}" --no-build-isolation --no-deps

echo "=== Running environment check ==="
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
  JAX_PLATFORMS=cpu \
  python3 "${SCRIPT_DIR}/check_environment.py"
