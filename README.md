# PersonaPlex — fine-tuning tiếng Việt

Fine-tune `nvidia/personaplex-7b-v1` bằng LoRA với trainer Tim, model local và conversations đã chuẩn bị. Luồng chạy gồm **prepare → train**; không tải model từ Hugging Face.

## Yêu cầu

- Linux/CUDA, Python ≥ 3.10 và environment có Moshi **training runtime tương thích Tim**; không dùng loader chỉ dành cho inference.
- Model local tại `/home/voice/data/voice/personaplex-7b-v1`, gồm `model.safetensors`, Mimi `tokenizer-e351c8d8-checkpoint125.safetensors`, SentencePiece `tokenizer_spm_32k_3.model` và `config.json` cho free-running.
- Prepared conversations: stereo 24 kHz **LEFT=agent, RIGHT=user**, transcript word-level, metadata, agent voice prompt mono và text prompt. Alignment và chuẩn bị prompts thực hiện bên ngoài trainer.

Trước khi scale dataset, hoàn tất acceptance overfit/save/reload/inference trên tập nhỏ theo [scripts/README.md](scripts/README.md). CPU tests chưa chứng minh training hoặc generation thành công trên GPU.

## Config

Các config dùng để chạy nằm trong `configs/`:

| File | Mục đích |
|---|---|
| `train_single_gpu.yaml` | Cấu hình đầu vào cho prepare: LoRA, text có dấu, seed và tỷ lệ val. |
| `train_synthetic.yaml` | Config train thực tế, được prepare tạo tại đường dẫn `--resolved-config`. |
| `acceptance.yaml` | Acceptance OtoSpeech và free-running probe trong training. |

Mặc định synthetic: LoRA rank 64, chunk 10s, batch 1, tích lũy 6 microbatches, 1024 optimizer steps; eval/checkpoint mỗi 64 steps. Recipe train giữ loss weights Tim; acceptance dùng weights riêng.

```yaml
data:
  vietnamese_text_mode: diacritics # diacritics | no_diacritics | telex
  eval_split_from_train: 0.05     # 5% conversations cho val; 0=tắt split
seed: 0
```

Split diễn ra **theo conversation trước chunking** và cố định theo seed; chunks của cùng conversation chỉ thuộc một tập. Giá trị `0` không tự eval trên train. Thay text mode cần export lại; inference phải dùng cùng mode.

## 1. Prepare

Activate environment hiện có, rồi đặt đường dẫn project trên server:

```bash
export PROJECT=/home/voice/code/VDT_02/baottn/personaplex-finetune-tim-style-main-v1
cd "$PROJECT"
```

```bash
python prepare_data.py \
  --manifest /home/voice/data/voice/vdt/data_ready/synthetic_500h/train.jsonl \
  --output /home/voice/data/voice/personaplex-exports/synthetic-vi-split \
  --config configs/train_single_gpu.yaml \
  --resolved-config configs/train_synthetic.yaml \
  --workers 8
```

Output export gồm `train.jsonl`, `val.jsonl`, `split.json`, WAV symlinks, JSON sidecars và reports. Config được ghi riêng vào **`configs/train_synthetic.yaml`**, không tạo YAML trong export. `--workers` là số tiến trình CPU validation; `0` chạy tuần tự.

Thư mục export và config đích phải chưa tồn tại. Đọc split/report rồi kiểm tra `train_data`, `eval_data`, `run_dir` trong config vừa tạo. Nếu run cũ đã tồn tại, chọn `run_dir` mới trước khi train.

## 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run \
  --standalone --nproc-per-node=1 train_local.py \
  --model-root /home/voice/data/voice/personaplex-7b-v1 \
  --config configs/train_synthetic.yaml \
  --free-running-config configs/acceptance.yaml \
  --token-cache /home/voice/data/voice/personaplex-exports/synthetic-vi-split/token_cache
```

`python -m torch.distributed.run` tương đương `torchrun`, dùng interpreter đang active. Training dùng voice/text conditioning của từng sample synthetic, loss-masked trên vùng prompt. Bỏ `--token-cache` nếu không cần cache.

Free-running mặc định chạy trước train và mỗi **10 optimizer steps**, dùng RIGHT/user của OtoSpeech `conv_0001`, cửa sổ 0–10s và prompts trong `configs/acceptance.yaml`. Đây là một probe cố định, chưa tự chọn audio từ synthetic val. Bỏ `--free-running-config` để tắt. Kiểm tra các đường dẫn audio/prompt trước khi chạy; free-running tạm dừng training và cần đủ CPU RAM/disk cho offload/inference.

## Log và artifacts

Theo `run_dir` mặc định:

```bash
tail -f /home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu/metrics.train.jsonl
tail -f /home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu/metrics.eval.jsonl
```

- Checkpoint: `<run_dir>/checkpoints/checkpoint_000064/consolidated/`.
- Free-running: `<run_dir>/free_running/step_000010/`; nghe `current/agent.wav` hoặc `dialogue_step.wav`, đọc `current/agent.txt`. Log inference nằm ở `step_000010.log`.
- Resume: thêm `--resume-from /absolute/run/checkpoints/checkpoint_000064` vào lệnh train, giữ config/horizon tương ứng.

Validation Tim giới hạn **40 batches mỗi lần**, không quét toàn bộ tập val lớn. Loss giảm cần được đối chiếu với text/audio sinh ra. GPU training và offload/restore chưa được xác minh bằng run model thật trong lần cập nhật này.

## Tài liệu chi tiết

- [Hướng dẫn server, split và resume](docs/train_synthetic_server.md)
- [GPU acceptance và free-running](scripts/README.md)
- [Prepared data contract](docs/PREPARED_DATA.md)
- [Sample filtering](docs/SAMPLE_FILTER.md)
- [Validation và tiêu chí hoàn tất](docs/VALIDATION.md)
- [Reference provenance](docs/REFERENCE_BASELINE.md), [third-party notices](docs/THIRD_PARTY_NOTICES.md)
