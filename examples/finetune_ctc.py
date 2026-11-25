"""Fine-tune a w2v-BERT 2.0 CTC speech recognizer on TPU (or CPU for a quick check).

Data are JSON Lines manifests, one utterance per line::

    {"audio": "clips/0001.wav", "text": "тема про яку не люблять говорити"}

Relative audio paths are resolved against the manifest's folder. Text is tokenized with a
SentencePiece model (by default ``tokenizer.model`` from the ``--model`` repo); id 0 is the CTC
blank, as in ``speech-uk/w2v-bert-v3``.

Start from a fine-tuned CTC checkpoint (``--model speech-uk/w2v-bert-v3``) or from the base
encoder (``--model facebook/w2v-bert-2.0 --tokenizer my.model``), which gets a new CTC head
sized to the tokenizer.

Example (TPU)::

    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 JAX_PLATFORMS=cpu \\
        python examples/finetune_ctc.py --train train.jsonl --eval dev.jsonl \\
        --output-dir runs/uk --batch-size 8 --max-seconds 20 --steps 20000

Notes:

* Attention runs in the Pallas kernels, forward and backward, for long inputs, and in plain
  PyTorch for short ones (the ``backend="auto"`` rule). Gradients match Hugging Face's.
* Layer drop and Hugging Face's SpecAugment are switched off by ``optimize()``: they can't be
  compiled (random Python branches, numpy masks). This script applies SpecAugment to the input
  features on the device instead (``--time-masks``, ``--freq-masks``).
* Weights are FP32 by default (AdamW updates in full precision; TPU matmuls on FP32 inputs run
  at the MXU's default precision). ``--precision autocast`` runs the forward pass under BF16
  autocast; that mode is untested with TorchTPU.
* Batches are padded to fixed frame buckets and a fixed batch size, so the number of compiled
  shapes stays small (two programs per bucket: training and evaluation).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
import torch.nn.functional as F

import w2vbert_tpu

FRAMES_PER_SECOND = 50  # encoder frames (input_features rows) per second of audio


@dataclass
class Utterance:
    audio: Path
    text: str
    seconds: float


def read_manifest(path: str, max_seconds: float) -> list[Utterance]:
    """Read a JSONL manifest and drop clips longer than ``max_seconds``."""
    import soundfile as sf

    base = Path(path).parent
    items = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio = Path(row["audio"])
        audio = audio if audio.is_absolute() else base / audio
        seconds = float(row.get("duration") or sf.info(str(audio)).duration)
        if seconds <= max_seconds:
            items.append(Utterance(audio, str(row["text"]), seconds))
    return items


def length_batches(items: list[Utterance], batch_size: int, shuffle: bool) -> list[list[int]]:
    """Batches of similar length (less padding), in random order when ``shuffle``."""
    order = sorted(range(len(items)), key=lambda i: items[i].seconds)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def load_clip(path: Path) -> torch.Tensor:
    audio, _ = w2vbert_tpu.load_audio(path)  # mono, 16 kHz
    audio = audio - audio.mean()  # peak normalization, as used for speech-uk/w2v-bert-v3
    return audio / audio.abs().max().clamp_min(1e-8)


def make_batch(
    items: list[Utterance],
    indices: list[int],
    tokenizer: Any,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Features padded to a frame bucket and to ``batch_size`` rows, plus padded targets."""
    waves = [load_clip(items[i].audio) for i in indices]
    inputs = w2vbert_tpu.extract_features(waves, sampling_rate=16000)
    padded = w2vbert_tpu.pad_to_bucket(
        inputs["input_features"], inputs["attention_mask"], batch_buckets=(batch_size,)
    )
    tokens = [tokenizer.encode(items[i].text) for i in indices]
    max_tokens = max(1, max(len(t) for t in tokens))
    targets = torch.zeros(batch_size, max_tokens, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    for row, ids in enumerate(tokens):
        targets[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        target_lengths[row] = len(ids)
    real_rows = torch.zeros(batch_size)
    real_rows[: len(indices)] = 1.0
    return {
        "input_features": padded.input_features.to(device),
        "attention_mask": padded.attention_mask.to(device),
        "targets": targets.to(device),
        "target_lengths": target_lengths.to(device),
        "real_rows": real_rows.to(device),
        "texts": [items[i].text for i in indices],
    }


def spec_augment(
    features: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    time_masks: int,
    max_time_fraction: float,
    freq_masks: int,
    max_freq_bins: int,
) -> torch.Tensor:
    """SpecAugment on the 160-dim features (two stacked 80-bin frames), on the device.

    Masked values become 0, the per-bin mean after the extractor's normalization. Static shapes
    only (random starts and widths compared against ``arange``), so nothing syncs with the host.
    """
    batch, frames, dim = features.shape
    bins = dim // 2
    device = features.device
    keep = torch.ones(batch, frames, 2, bins, device=device, dtype=torch.bool)
    lengths = attention_mask.sum(-1).to(torch.float32)
    t = torch.arange(frames, device=device)
    f = torch.arange(bins, device=device)
    for _ in range(time_masks):
        width = torch.rand(batch, device=device) * max_time_fraction * lengths
        start = torch.rand(batch, device=device) * (lengths - width).clamp_min(0.0)
        hit = (t[None, :] >= start[:, None]) & (t[None, :] < (start + width)[:, None])
        keep &= ~hit[:, :, None, None]
    for _ in range(freq_masks):
        width = (torch.rand(batch, device=device) * max_freq_bins).floor()
        start = (torch.rand(batch, device=device) * (bins - width)).floor()
        hit = (f[None, :] >= start[:, None]) & (f[None, :] < (start + width)[:, None])
        keep &= ~hit[:, None, None, :]
    return features * keep.reshape(batch, frames, dim).to(features.dtype)


def ctc_loss(
    logits: torch.Tensor, batch: dict[str, Any], config: Any, blank_id: int
) -> torch.Tensor:
    """Mean over real rows of the per-utterance CTC loss divided by its target length."""
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # [T, B, V]
    input_lengths = w2vbert_tpu.ctc_output_lengths(config, batch["attention_mask"].sum(-1))
    per_row = F.ctc_loss(
        log_probs,
        batch["targets"],
        input_lengths,
        batch["target_lengths"],
        blank=blank_id,
        reduction="none",
        zero_infinity=True,
    )
    per_row = per_row / batch["target_lengths"].clamp_min(1)
    weights = batch["real_rows"]
    return (per_row * weights).sum() / weights.sum()


def edit_distance(a: list[str], b: list[str]) -> int:
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def evaluate(
    model: Any,
    items: list[Utterance],
    tokenizer: Any,
    args: argparse.Namespace,
    autocast: Any,
) -> dict[str, float]:
    model.eval()  # also refreshes the inference caches from the trained weights
    word_errors = words = char_errors = chars = 0
    losses = []
    with torch.no_grad():
        for indices in length_batches(items, args.batch_size, shuffle=False):
            batch = make_batch(items, indices, tokenizer, args.batch_size, args.device)
            with autocast():
                logits = model(
                    input_features=batch["input_features"],
                    attention_mask=batch["attention_mask"],
                ).logits
            losses.append(float(ctc_loss(logits, batch, model.config, args.blank_id)))
            lengths = w2vbert_tpu.ctc_output_lengths(model.config, batch["attention_mask"].sum(-1))[
                : len(indices)
            ]
            ids = w2vbert_tpu.ctc_greedy_decode(
                logits, lengths, blank_id=args.blank_id, ignore_ids=args.ignore_ids
            )
            for hyp_ids, ref in zip(ids, batch["texts"]):
                hyp = tokenizer.decode_ids(hyp_ids).strip()
                word_errors += edit_distance(hyp.split(), ref.split())
                words += max(1, len(ref.split()))
                char_errors += edit_distance(list(hyp), list(ref))
                chars += max(1, len(ref))
    model.train()
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "wer": word_errors / max(1, words),
        "cer": char_errors / max(1, chars),
    }


def load_model(args: argparse.Namespace, vocab_size: int) -> Any:
    from transformers import Wav2Vec2BertForCTC

    from w2vbert_tpu.model import _model_class_for

    common: dict[str, Any] = dict(
        dtype=torch.float32, device=args.device, compile=args.compile, backend=args.backend
    )
    if _model_class_for(args.model, {}) is Wav2Vec2BertForCTC:
        model = w2vbert_tpu.load(args.model, **common)
    else:
        # Base encoder: add a new CTC head sized to the tokenizer (randomly initialized).
        model = w2vbert_tpu.load(
            args.model,
            model_class=Wav2Vec2BertForCTC,
            vocab_size=vocab_size,
            pad_token_id=args.blank_id,
            **common,
        )
    return model


# Training-only settings that optimize() switches off (they can't be compiled). Checkpoints are
# saved with the original values, so they load exactly like the model training started from.
_TRAINING_ONLY = ("layerdrop", "mask_time_prob", "mask_feature_prob")


def save_checkpoint(model: Any, path: Path, original: dict[str, Any], tokenizer_path: str) -> None:
    current = {key: getattr(model.config, key) for key in original}
    for key, value in original.items():
        setattr(model.config, key, value)
    try:
        model.save_pretrained(path)
    finally:
        for key, value in current.items():
            setattr(model.config, key, value)
    if Path(tokenizer_path).is_file():
        (path / "tokenizer.model").write_bytes(Path(tokenizer_path).read_bytes())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument("--model", default="speech-uk/w2v-bert-v3")
    p.add_argument("--tokenizer", help="SentencePiece model (default: tokenizer.model in --model)")
    p.add_argument("--train", required=True, help="JSONL manifest")
    p.add_argument("--eval", help="JSONL manifest for periodic evaluation")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="tpu", help="tpu, or cpu for a quick check")
    p.add_argument("--backend", default="auto", choices=["auto", "pallas", "native"])
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision", default="fp32", choices=["fp32", "autocast"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-seconds", type=float, default=20.0)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--time-masks", type=int, default=2)
    p.add_argument("--max-time-fraction", type=float, default=0.05)
    p.add_argument("--freq-masks", type=int, default=2)
    p.add_argument("--max-freq-bins", type=int, default=27)
    p.add_argument("--blank-id", type=int, default=0)
    p.add_argument("--ignore-ids", type=int, nargs="*", default=[1], help="dropped when decoding")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    import sentencepiece as spm
    from huggingface_hub import hf_hub_download

    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer or hf_hub_download(args.model, "tokenizer.model")
    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    train_items = read_manifest(args.train, args.max_seconds)
    eval_items = read_manifest(args.eval, args.max_seconds) if args.eval else []
    hours = sum(u.seconds for u in train_items) / 3600
    print(f"train: {len(train_items)} clips ({hours:.1f} h), eval: {len(eval_items)} clips")

    from transformers import AutoConfig

    source_config = AutoConfig.from_pretrained(args.model)
    original = {key: getattr(source_config, key) for key in _TRAINING_ONLY}
    model = load_model(args, tokenizer.get_piece_size())
    model.train()
    no_decay = ("bias", "layer_norm", "LayerNorm")
    groups = [
        {"params": [], "weight_decay": args.weight_decay},
        {"params": [], "weight_decay": 0.0},
    ]
    for name, param in model.named_parameters():
        if param.requires_grad:
            groups[any(k in name for k in no_decay)]["params"].append(param)
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-8)

    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        return max(0.0, (args.steps - step) / max(1, args.steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    def autocast() -> Any:
        if args.precision == "autocast":
            return torch.autocast(device_type=args.device, dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def sync() -> None:
        if args.device == "tpu":
            torch.accelerator.synchronize()

    step = 0
    t0 = time.perf_counter()
    running = torch.zeros((), device=args.device)
    while step < args.steps:
        for indices in length_batches(train_items, args.batch_size, shuffle=True):
            batch = make_batch(train_items, indices, tokenizer, args.batch_size, args.device)
            features = spec_augment(
                batch["input_features"],
                batch["attention_mask"],
                time_masks=args.time_masks,
                max_time_fraction=args.max_time_fraction,
                freq_masks=args.freq_masks,
                max_freq_bins=args.max_freq_bins,
            )
            with autocast():
                logits = model(input_features=features, attention_mask=batch["attention_mask"])
            loss = ctc_loss(logits.logits, batch, model.config, args.blank_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += loss.detach()
            step += 1

            if step % args.log_every == 0:
                sync()
                elapsed = time.perf_counter() - t0
                print(
                    f"step {step}: loss {float(running) / args.log_every:.4f}, "
                    f"lr {scheduler.get_last_lr()[0]:.2e}, "
                    f"{elapsed / args.log_every:.2f} s/step",
                    flush=True,
                )
                running.zero_()
                t0 = time.perf_counter()
            if eval_items and step % args.eval_every == 0:
                metrics = evaluate(model, eval_items, tokenizer, args, autocast)
                print(f"step {step}: eval " + json.dumps(metrics), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                path = out_dir / f"step-{step}"
                save_checkpoint(model, path, original, tokenizer_path)
                print(f"saved {path}", flush=True)
            if step >= args.steps:
                break

    if eval_items:
        print("final eval " + json.dumps(evaluate(model, eval_items, tokenizer, args, autocast)))


if __name__ == "__main__":
    main()
