#!/usr/bin/env python3
"""Convert perplexity-ai/pplx-embed-v2-context-0.6b text tower + contextual head to GGUF (F16/F32)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import gguf
from safetensors import safe_open


def tensor_name(src: str) -> str:
    if src == "contextual_projection.weight":
        return src
    if src.startswith("language_model."):
        return "l." + src[len("language_model."):]
    raise ValueError(f"Unexpected non-visual tensor: {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = json.loads((args.src / "config.json").read_text())
    tcfg = cfg["text_config"]
    assert cfg["model_type"] == "pplx_contextual_qwen3_5"

    index_path = args.src / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    if (args.src / "contextual_head.safetensors").exists() and "contextual_head.safetensors" not in shards:
        shards.append("contextual_head.safetensors")

    keep_f32_suffixes = (
        "contextual_projection.weight",
        "norm.weight",
        "layernorm.weight",
        "A_log",
        "dt_bias",
        "conv1d.weight",
        "in_proj_a.weight",
        "in_proj_b.weight",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.out), "pplx_contextual_qwen3_5")
    writer.add_string("general.name", "perplexity-ai/pplx-embed-v2-context-0.6b")
    writer.add_uint32("pplx_contextual.embedding_dim", int(cfg.get("embedding_dim", 1024)))
    writer.add_uint32("pplx_contextual.hidden_size", int(tcfg["hidden_size"]))
    writer.add_uint32("pplx_contextual.intermediate_size", int(tcfg["intermediate_size"]))
    writer.add_uint32("pplx_contextual.num_hidden_layers", int(tcfg["num_hidden_layers"]))
    writer.add_uint32("pplx_contextual.num_attention_heads", int(tcfg["num_attention_heads"]))
    writer.add_uint32("pplx_contextual.num_key_value_heads", int(tcfg["num_key_value_heads"]))
    writer.add_uint32("pplx_contextual.head_dim", int(tcfg["head_dim"]))
    writer.add_uint32("pplx_contextual.linear_num_heads", int(tcfg["linear_num_value_heads"]))
    writer.add_uint32("pplx_contextual.linear_head_dim", int(tcfg["linear_value_head_dim"]))
    writer.add_uint32("pplx_contextual.linear_conv_kernel", int(tcfg["linear_conv_kernel_dim"]))
    writer.add_uint32("pplx_contextual.vocab_size", int(tcfg["vocab_size"]))
    writer.add_float32("pplx_contextual.rms_norm_eps", float(tcfg["rms_norm_eps"]))
    writer.add_float32("pplx_contextual.rope_theta", float(tcfg["rope_parameters"]["rope_theta"]))
    writer.add_float32("pplx_contextual.partial_rotary_factor", float(tcfg["partial_rotary_factor"]))
    writer.add_string("pplx_contextual.query_prefix", str(cfg.get("query_prefix", "[Q] ")))
    writer.add_string("pplx_contextual.document_prefix", str(cfg.get("document_prefix", "[D] ")))
    writer.add_string("pplx_contextual.boundary_marker", str(cfg.get("boundary_marker", "<|chunk_sep|>")))

    seen = set()
    exported = 0
    for shard in shards:
        with safe_open(args.src / shard, framework="pt", device="cpu") as f:
            for src_name in sorted(f.keys()):
                if src_name.startswith("visual.") or src_name == "lm_head.weight" or src_name.startswith("mtp."):
                    continue
                dst_name = tensor_name(src_name)
                if dst_name in seen:
                    continue
                seen.add(dst_name)
                t = f.get_tensor(src_name).float().numpy()
                is_f32 = any(dst_name.endswith(sfx) for sfx in keep_f32_suffixes) or t.ndim != 2
                arr = np.ascontiguousarray(t, dtype=np.float32 if is_f32 else np.float16)
                assert np.isfinite(arr).all(), f"Non-finite values in {src_name}"
                writer.add_tensor(dst_name, arr)
                exported += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Exported {exported} tensors to {args.out} ({args.out.stat().st_size / (1024*1024):.1f} MiB)")


if __name__ == "__main__":
    main()
