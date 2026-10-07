# Train synthetic tiếng Việt trên server

Chạy từ repo đã upload, trong environment Linux/CUDA có Moshi training runtime tương thích Tim. Script dùng interpreter `python` đang active; không cài dependency hoặc tải model. Có thể đặt `PYTHON=/absolute/env/bin/python`.

## Cấu hình đã điền

- Model local: `/home/voice/data/voice/personaplex-7b-v1`.
- Prepared training: `/home/voice/data/voice/vdt/data_ready/synthetic_500h/train.jsonl`.
- Export Tim: `/home/voice/data/voice/personaplex-exports/synthetic-vi`.
- Run: `/home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu`.
- `configs/train_single_gpu.yaml`: tiếng Việt có dấu, train/val theo conversation (95%/5%), LoRA rank 64, chunk 10s, batch 1, tích lũy 6 microbatches, 1024 steps, checkpoint mỗi 64 steps. Giữ loss weights của recipe Tim (`4.0`, `0.04`, L2 `0.0001`).
- `config.full.yaml`: acceptance OtoSpeech, 8 fixed smoke chunks, loss weights acceptance (`50.0`, `0.3`, L2 `0`). Không dùng file này làm native training config.
- Free-running: baseline step 0 và mỗi 10 optimizer steps; dùng user channel RIGHT của OtoSpeech `conv_0001`, voice prompt của conversation đó và file `system_prompt.txt` trên server. Đây là mẫu probe OtoSpeech khi train synthetic; không phải evaluation synthetic.

Training lấy voice/text conditioning của từng sample synthetic từ prepared metadata/sidecars. `acceptance.inference` chỉ cấu hình probe sinh tự do. Đường dẫn text prompt đã bỏ dấu chấm cuối `.txt.`; kiểm tra tên file thật trước khi chạy.

## Split conversation khi prepare

`data.eval_split_from_train: 0.05` trong `configs/train_single_gpu.yaml` là tỷ lệ conversations giữ cho val; `0` tắt split. Prepare xếp hạng `sample_id` bằng SHA-256 với `seed`, chọn `ceil(N × ratio)` conversations cho val, tối thiểu 1 và tối đa N−1. Cần ít nhất 2 conversations. Membership không phụ thuộc thứ tự manifest, workers hoặc cache; giữ manifest/seed/ratio cho cùng split. Mọi chunk của một conversation chỉ thuộc một tập. Các ID trùng hoặc cùng đường dẫn audio nguồn dưới nhiều ID bị từ chối; dữ liệu đã duplicate thành file riêng cần dedupe ở bước chuẩn bị nguồn.

Export tạo `train.jsonl`, `val.jsonl`, `split.json` (IDs/counts/seed/source manifest hash) và `train.yaml` đã bind cả `data.train_data` và `data.eval_data`. Manifest prepared và audio nguồn không bị sửa. `eval_split_from_train` được bridge loại khỏi YAML native trước khi gọi Tim. Không kết hợp ratio > 0 với `data.eval_data` đã chỉ định. Ratio 0 giữ eval manifest riêng nếu có.

Config synthetic bật `do_eval: true`, `eval_freq: 64`: validation loss mỗi 64 steps và step cuối. Native Tim chỉ đo tối đa 40 batches mỗi lần eval, không quét toàn bộ val lớn. Free-running vẫn dùng probe OtoSpeech đã cấu hình; chưa tự lấy sample synthetic từ val. Acceptance đặt ratio 0 để giữ fixed-set overfit.

Export cũ chưa split phải export lại vào **thư mục mới**; không resume run cũ và gọi nó là run có held-out validation. Ví dụ `EXPORT_DIR=/home/voice/data/voice/personaplex-exports/synthetic-vi-split bash scripts/train_synthetic_server.sh prepare`, rồi dùng cùng `EXPORT_DIR` cho bước train; chọn `run_dir` mới nếu run mặc định đã tồn tại.

## Chạy

Trước khi scale, hoàn tất acceptance single GPU theo `scripts/README.md`; CPU checks không chứng minh overfit hay reload trên GPU.

```bash
cd /home/voice/code/VDT_02/baottn/personaplex-finetune-tim-style-main-v1
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_gpu_checks.sh config.full.yaml
```

Acceptance mặc định xuất artifacts trong `runs/` của repo, cần thư mục mới. Có thể dùng `_run_all.py --runs-dir /absolute/new/acceptance-run` để đặt output ngoài repo.

Sau khi acceptance đạt, export và train synthetic bằng hai bước để đọc báo cáo dữ liệu trước:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_synthetic_server.sh prepare
CUDA_VISIBLE_DEVICES=0 bash scripts/train_synthetic_server.sh train
```

Hoặc `bash scripts/train_synthetic_server.sh all` chạy tuần tự cả hai bước. Export đòi hỏi thư mục đích mới; không chạy `all` lại khi export đã tồn tại. Script ghi `train.log` cạnh manifest export và giữ exit code lỗi của trainer dù dùng `tee`.

```bash
tail -f /home/voice/data/voice/personaplex-exports/synthetic-vi/train.log
tail -f /home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu/metrics.train.jsonl
```

Checkpoint: `<run_dir>/checkpoints/checkpoint_000064/consolidated/`. Free-running: `<run_dir>/free_running/step_000010/` và `step_000010.log`. Free-running tạm dừng training, offload sang CPU và chạy inference subprocess; cần đủ CPU RAM/disk. Hook offload/restore chưa có bằng chứng chạy model thật trên GPU trong lần cập nhật này. Lỗi inference dừng training; đọc file log theo step. Đặt `acceptance.inference.free_running_every_steps: 0` trong `config.full.yaml` để tắt.

## Thay đổi hoặc resume

Sửa `configs/train_single_gpu.yaml` trước khi export; file `<EXPORT_DIR>/train.yaml` là bản cấu hình trainer thực sự sử dụng. Muốn đổi cấu hình sau export, sửa bản derived đó hoặc export mới. Muốn đổi text mode, export mới với mode khớp cả training và inference.

Script hỗ trợ `MODEL_ROOT`, `PREPARED_MANIFEST`, `EXPORT_DIR`, `TRAIN_CONFIG`, `FREE_RUNNING_CONFIG`, `EXPORT_WORKERS` và `RESUME_FROM` qua environment. Đổi `EXPORT_DIR` chỉ đổi output export, không đổi `run_dir`: chỉnh YAML nếu cần run mới. Không ghi đè run cũ.

```bash
RESUME_FROM=/home/voice/data/voice/personaplex-runs/synthetic-vi-single-gpu/checkpoints/checkpoint_000064 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_synthetic_server.sh train
```

Giữ horizon/config và checkpoint tương ứng; periodic inference cũng cần output step chưa tồn tại. Không xóa artifacts để né lỗi directory trùng; chọn run mới hoặc xử lý resume theo trainer hiện có.

Prepared synthetic phải có cùng contract bridge: stereo 24 kHz LEFT=agent/RIGHT=user, words aligned, metadata channel mapping, agent voice mono và text prompt. Export không alignment, chọn voice region hay sửa transcript. Nếu prompt/crop không vừa chunk 10s, xem diagnostics rồi tăng duration phù hợp; không kết luận dữ liệu server hợp lệ trước khi export thực tế.
