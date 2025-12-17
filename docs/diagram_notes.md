# ByteCaption Architecture Notes for Diagram

## End-to-end data path
- **Input**: JPEG image is not decoded to pixels; the raw byte stream is used.
- **Byte-stream preprocessing** (collate): `PILSave` re-encodes the PIL image to JPEG, then optional byte-level augmentations (byte stream corruption, shuffle, mask, uniform noise, byte permutation) are applied before padding to a fixed length.【F:corenet/data/collate_fns/byteformer_collate_functions.py†L28-L117】
- **Encoder (ByteFormer)**: Byte stream tokens (vocab size default 257) pass through the CoreNet ByteFormer backbone wrapped as a Hugging Face `PreTrainedModel`. The wrapper removes the classifier head and exposes the backbone sequence features plus an attention mask for the decoder.【F:PureT/byteformer_immigration.py†L20-L106】
- **Decoder (PureT)**: The PureT caption decoder consumes the token-level features. It projects ByteFormer features to the configured embedding dimension, computes global pooled features, and runs a Transformer decoder to generate captions with log-softmax outputs.【F:PureT/models/pure_byteformer.py†L33-L145】

## Key configuration knobs (useful to annotate in the figure)
- **ByteFormer pretrained setup**: Config and weights fixed to `conv_kernel_size=4, window_sizes=[128]` and `imagenet_jpeg_q60_k4_w128.pt` in `get_opts()`; change here to illustrate different encoder settings.【F:PureT/byteformer_immigration.py†L73-L106】
- **Vocabulary and padding**: Byte-level vocab size defaults to 257; padding index is used when collate pads variable-length byte streams.【F:PureT/byteformer_immigration.py†L84-L106】【F:corenet/data/collate_fns/byteformer_collate_functions.py†L101-L117】
- **Byte-level augmentations**: Diagram can show toggles for corruption `level` (`none`, `light`, `medium`, `heavy`) and types (`bit_flip`, `segment_dropout`, `header_truncation`, `tail_truncation`) injected after JPEG encoding.【F:corenet/data/transforms/image_bytes.py†L31-L118】
- **Decoder hyperparameters**: Decoder depth, heads, embedding dimension, dropout, and feed-forward dropout are driven by `cfg.MODEL.BILINEAR.{DIM, DECODE_LAYERS, HEAD, DECODE_DROPOUT, DECODE_FF_DROPOUT}`; vocabulary size is `cfg.MODEL.VOCAB_SIZE + 1`. These are common labels to expose near the decoder block.【F:PureT/models/pure_byteformer.py†L33-L145】
- **Training entry points**: `PureT/main.py` and `main_val.py` show that the `MODEL.TYPE` needs to be `PureT_byteformer` to route through the byte-stream branch; good to note in captions or footnotes for reproducibility.【F:PureT/README.md†L5-L64】

## Suggested diagram layout
1. **Input & augmentation strip**: Leftmost box labeled “JPEG bitstream”. Under it, a small pipeline showing `PILSave` → optional corruption (with badges for flip/drop/truncation/noise/permutation) → padding.
2. **Encoder block (ByteFormer)**: Draw a stack labeled “ByteFormer backbone (conv_kernel=4, window=128)” with arrows showing attention mask output and sequence features.
3. **Feature projection & pooling**: Small block turning `hidden_size` to `ATT_FEATS_EMBED_DIM`, plus a parallel path computing global mean-pooled feature `gx`.
4. **Decoder block (PureT)**: Multi-layer Transformer decoder annotated with depth/heads/dropout; inputs are pooled `gx`, encoder features, and autoregressive mask.
5. **Caption output**: Arrow to text tokens with vocabulary size and log-softmax noted.
6. **Callouts**: Add side notes for configurable items (pretrained weight path, augmentation level, whether backbone is frozen, beam search in inference) and datasets (COCO/Flickr8k) from the training README.【F:PureT/README.md†L31-L64】

Use color coding to distinguish byte-level operations (e.g., purple), encoder (blue), decoder (green), and training/inference controls (gray). Dotted lines can indicate optional augmentations or frozen-backbone mode.
