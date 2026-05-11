# CrispEmbed — Python wheel

Lightweight ggml-based text embedding inference. **23+ verified models**,
~9× faster than FastEmbed (ONNX) on the standard MiniLM-L6 benchmark.

Supports BERT, XLM-R, MPNet, NomicBERT, ModernBERT, DeBERTa-v2, Qwen3,
Gemma3, and BidirLM-Omni — including SPLADE sparse, ColBERT multi-vector,
and cross-encoder reranking through the same shared library.

This package bundles the native `libcrispembed.{so,dylib,dll}` plus its
ggml backend siblings, so a plain `pip install crispembed` works without
needing CMake on the user's machine.

## Quick start

```python
from crispembed import CrispEmbed

# Auto-downloads the GGUF from huggingface.co/cstr/<model>-GGUF on first use.
ce = CrispEmbed("all-MiniLM-L6-v2")
vec = ce.encode("Hello world")          # numpy float32 [384]
batch = ce.encode(["hello", "world"])   # numpy float32 [2, 384]
```

## Multimodal (BidirLM-Omni)

```python
ce = CrispEmbed("bidirlm-omni-2.5b")
text_vec  = ce.encode("a cat on a mat")
image_vec = ce.encode_image("cat.jpg")     # needs `pip install crispembed[image]`
# text_vec @ image_vec.T is a meaningful cross-modal similarity score.
```

## Building from source

Wheels are built by CI from the parent `CrispEmbed` repo via CMake. To
build locally, see the parent project's `README.md` and `build-macos.sh`
/ release.yml. The Python package's `pyproject.toml` here is just a thin
packaging shim around the prebuilt native libraries.

## License

MIT. See `LICENSE`.
