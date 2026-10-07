# Các lệnh độc lập để train synthetic tiếng Việt

Chạy trong environment Linux/CUDA hiện có, với Moshi training runtime tương thích Tim. Không tự tạo environment, cài dependency hay tải model.

## Config trong project

- `configs/train_single_gpu.yaml`: config đầu vào cho prepare, tiếng Việt có dấu, `data.eval_split_from_train: 0.05`, seed 0; LoRA rank 64, chunk 10s, batch 1, tích lũy 6 microbatches, 1024 steps; validation/checkpoint mỗi 64 steps.
- `configs/train_synthetic.yaml`: config train thực tế, tạo ở đường dẫn chỉ định bởi `--resolved-config` sau prepare; bind train/val manifest đã export. Chỉnh learning rate, duration, run_dir hoặc số steps ở file này trước khi train.
- `configs/acceptance.yaml`: acceptance OtoSpeech và free-running probe. Đã chuyển từ `config.full.yaml` ở root. Baseline step 0 và mỗi 10 optimizer steps; dùng RIGHT/user của `conv_0001`, voice prompt và file `system_prompt.txt` trên server.

Prepare chỉ ghi manifest, WAV symlinks, JSON sidecars và reports trong export. `--config` chỉ đọc cấu hình; không tự tạo YAML. Muốn xuất config train, phải truyền `--resolved-config` rõ ràng. File đích cần thư mục cha có sẵn và không được tồn tại; không ghi đè config đầu vào.

## 1. Đặt project

```bash
export PROJECT=/home/voice/code/VDT_02/baottn/personaplex-finetune-tim-style-main-v1
cd "$PROJECT"
```

## 2. Acceptance trước khi scale

Mỗi command dùng output mới; xem `scripts/README.md` để đọc kết quả từng phase.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/_run_all.py \
  --config "$PROJECT/configs/acceptance.yaml" \
  --runs-dir /home/voice/data/voice/personaplex-runs/acceptance-checks
```

## 3. Prepare synthetic và chia train/val

```bash
python "$PROJECT/prepare_data.py" \
  --manifest /home/voice/data/voice/vdt/data_ready/synthetic_500h/train.jsonl \
  --output /home/voice/data/voice/personaplex-exports/synthetic-vi-split \
  --config "$PROJECT/configs/train_single_gpu.yaml" \
  --resolved-config "$PROJECT/configs/train_synthetic.yaml" \
  --workers 8
```

Prepared synthetic phải có stereo 24 kHz LEFT=agent/RIGHT=user, words aligned, metadata channel mapping, agent voice mono và text prompt. Export không alignment, chọn voice region hay sửa transcript. Workers là tiến trình CPU validation; `0` xử lý tuần tự.

Split theo conversation trước chunking: chọn `ceil(N × ratio)` sample IDs bằng thứ hạng SHA-256 với seed, tối thiểu 1 và tối đa N−1. Tỷ lệ `0` không tạo val; ratio > 0 cần ít nhất 2 conversations và không kết hợp `data.eval_data` đã chỉ định. Mọi chunk của một conversation chỉ thuộc một tập. Kết quả không phụ thuộc thứ tự manifest, workers hoặc cache. ID trùng hoặc nhiều ID trỏ cùng audio nguồn bị từ chối; file audio duplicate thành đường dẫn khác cần dedupe ở bước nguồn.

Output gồm `train.jsonl`, `val.jsonl`, `split.json` (IDs/counts/seed/source manifest hash), các sidecars và báo cáo nguồn. Giữ nguyên dữ liệu prepared. Không có `train.yaml` trong export. Export cũ chưa split cần thư mục export mới.

## 4. Kiểm tra config và train

Kiểm tra `configs/train_synthetic.yaml`: `train_data`, `eval_data`, `run_dir`, `do_eval`, `eval_freq`. Nếu run mặc định đã tồn tại, chọn run_dir mới; không resume run đã train trên toàn dataset rồi coi val là held-out.

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --standalone --nproc-per-node=1 \
  "$PROJECT/train_local.py" \
  --model-root /home/voice/data/voice/personaplex-7b-v1 \
  --config "$PROJECT/configs/train_synthetic.yaml" \
  --free-running-config "$PROJECT/configs/acceptance.yaml" \
  --token-cache /home/voice/data/voice/personaplex-exports/synthetic-vi-split/token_cache
```

Lệnh này tương đương `torchrun`, dùng cùng interpreter active với prepare. Bỏ `--token-cache` nếu không cần cache. Bỏ `--free-running-config` nếu không chạy periodic generation; hoặc đặt `acceptance.inference.free_running_every_steps: 0` để tắt.

Training lấy conditioning từ từng sample synthetic. Free-running vẫn là probe OtoSpeech `conv_0001` ở cửa sổ 0–10s; không tự chọn sample synthetic từ val. Kiểm tra voice prompt, conversation và đường dẫn text prompt thực tế (`system_prompt.txt`, không có dấu chấm cuối `.txt.`).

## 5. Đọc kết quả và resume

```bash
tail -f /home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu/metrics.train.jsonl
tail -f /home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu/metrics.eval.jsonl
```

Các đường dẫn trên theo run_dir mặc định; đổi tương ứng nếu bạn chỉnh config. Checkpoint ở `<run_dir>/checkpoints/checkpoint_000064/consolidated/`. Free-running ở `<run_dir>/free_running/step_000010/` và `step_000010.log`: nghe `current/agent.wav` hoặc `dialogue_step.wav`, đọc `current/agent.txt`.

Resume bằng cách thêm `--resume-from /absolute/run/checkpoints/checkpoint_000064` vào lệnh train; giữ horizon/config tương ứng. Periodic inference yêu cầu output step chưa tồn tại.

Validation Tim hiện chỉ đo tối đa 40 batches mỗi lần, không quét toàn bộ tập val lớn. Free-running tạm dừng training, offload sang CPU và chạy inference subprocess; cần CPU RAM/disk phù hợp. Lỗi inference dừng training. Chưa xác minh CUDA/offload/restore với model thật trong lần cập nhật này. Prompt/crop phải vừa chunk 10s; nếu không, xem diagnostics rồi tăng duration phù hợp.
