# GPU acceptance commands

Chạy các lệnh từ thư mục `personaplex-finetune-tim-style/`, dùng config đã chuẩn bị và một GPU. Mỗi lần chạy cần output directory mới.

## Config và tham số

Dùng [config.full.yaml](../config.full.yaml), có comment giải thích từng tham số. Thay các đường dẫn `/absolute/...` bằng model assets và prepared manifest thật trước khi chạy. File này dành cho acceptance harness, không chạy trực tiếp bằng trainer.

| Tham số lệnh | Ý nghĩa |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | Chỉ dùng GPU số 0. |
| `--config` | File YAML chứa config Tim và section `acceptance`. |
| `--sample-index` | Vị trí conversation bắt đầu trong manifest, tính từ 0. |
| `--output-dir` | Thư mục mới để lưu artifacts và logs. |
| `--num-samples` | Số conversations liên tiếp dùng cho mini-overfit. |
| `--max-steps` | Số optimizer steps và horizon OneCycle của phase 3. |
| `--checkpoint-step` | Khoảng cách lưu checkpoint; dùng bằng `--max-steps` để kiểm tra checkpoint cuối. |

Config mẫu dùng loss weights tường minh cho acceptance; nếu so sánh với một run Tim cụ thể, giữ đúng weights của run đó ở cả hai phía. Native evaluation tối đa 40 batches: chọn chunk duration sao cho toàn bộ fixed set nằm trong giới hạn này.

## 1. Interleaver và cache parity

Kiểm tra token streams, prompt length và masks giữa tokenizer Tim trực tiếp và token cache.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_interleaver_parity.py \
  --config config.full.yaml \
  --sample-index 0 \
  --output-dir runs/parity/interleaver
```

## 2. One-step parity

So sánh loss, LoRA parameters, optimizer và scheduler sau một step giữa trainer Tim gốc và local bridge.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/check_one_step_parity.py \
  --config config.full.yaml \
  --sample-index 0 \
  --output-dir runs/parity/one_step
```

## 3. Mini-overfit và save/reload

Train 10 conversations bằng trainer Tim trong 100 steps, rồi kiểm tra loss và LoRA tensors sau khi reload checkpoint trong process mới. Config cần `batch_size: 1`.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_gpu_acceptance.py \
  --config config.full.yaml \
  --num-samples 10 \
  --max-steps 100 \
  --sample-index 0 \
  --checkpoint-step 100 \
  --output-dir runs/gpu_acceptance
```

## 4. Chạy toàn bộ

Chạy tuần tự các kiểm tra trên và free-running nếu có `acceptance.inference`, dừng khi có lỗi và ghi `runs/gpu_acceptance_report.json`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_gpu_checks.sh config.full.yaml
```

## 5. Free-running inference và lưu kết quả

Chạy base model và model đã merge adapter trong **hai process riêng**, gọi nguyên `moshi.offline` của PersonaPlex. Cả hai dùng cùng input, prompts, seed và sampling settings. Đây là inference smoke, không phải parity với LMGen của Moshi training. Chưa kiểm chứng trên GPU.

Input lấy từ `acceptance.inference` trong config: WAV user **mono RIGHT channel**, agent voice prompt, file system prompt UTF-8 và `original_wav` stereo đã cắt đúng cùng cửa sổ. Không truyền stereo conversation vào `input_wav`.

Kết quả trong output directory:
- `dialogue_original.wav`: cửa sổ conversation gốc.
- `dialogue_base.wav`: agent từ base model + user input.
- `dialogue_step.wav`: agent từ checkpoint + user input.
- `manifest.json`: sample/step, cửa sổ, seed, generation settings, transcript base/current, reference tùy chọn và tên audio files.
- `base/`, `current/`: mono `agent.wav`, native `agent_text.json`, summary và merge artifacts; logs ở `base.log`, `current.log`.

Cả ba dialogue WAV là stereo **LEFT=agent, RIGHT=user**, 24 kHz. Script fail nếu input user không khớp RIGHT của original hoặc độ dài các outputs khác nhau. Transcript ghép từ native text token pieces, không phải ASR. CER/WER hiện để `null` và ghi rõ chưa tính, không giả lập metrics.

### Free-running trong lúc train

Đặt `acceptance.inference.free_running_every_steps: 20` để evaluation trước train (step 0) và sau optimizer steps 20, 40, ...; `0` tắt hook. Mini-overfit route `bridge` tự nhận cấu hình này; one-step parity và checkpoint reload không chạy hook. Không tự thêm evaluation cuối nếu step cuối không chia hết cho N. Khi resume, evaluation đầu tiên dùng trạng thái resume và tạo baseline base model cho session mới, không giả nhãn step 0.

Với launcher trực tiếp, truyền thêm `--free-running-config /absolute/path/config.full.yaml` vào `train_local.py`; `--config` vẫn là native training YAML đã materialize, không chứa `acceptance`. Kết quả nằm ở `<run_dir>/free_running/step_000000/`, `step_000020/`, ... mỗi thư mục có ba dialogue WAV và manifest. Adapter snapshots độc lập nằm ở `adapter_000000/`, ... không thay checkpoint/resume của trainer. Step manifest lấy optimizer step thực tế, không lấy nhãn `inference.step`.

Baseline base model chỉ sinh một lần; các step sau tái sử dụng và kiểm tra input hashes, prompts, seed cùng sampling settings. Training **tạm dừng đồng bộ** để inference: tensor storage của model/Mimi, gradients, FP32 master parameters và optimizer được offload CPU rồi restore trong `finally`; Parameter identities và RNG được giữ. Cần đủ CPU RAM cho training state cộng native inference model; mỗi mốc vẫn tốn thời gian merge/load 7B và dung lượng merged checkpoint. Chỉ hỗ trợ single GPU, LoRA, `do_ckpt: true`, `gen_eval.enable: false`; lỗi inference làm training fail-fast. Hook và offload/restore chưa được kiểm chứng trên GPU thật.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_free_running_inference.py \
  --config config.full.yaml \
  --checkpoint runs/gpu_acceptance/tim_run/checkpoints/checkpoint_000100/consolidated \
  --output-dir runs/free_running
```

`--checkpoint`: thư mục adapter consolidated; `--output-dir`: thư mục mới lưu kết quả. Có thể override config bằng `--input-wav`, `--voice-prompt`, `--text-prompt-file`, `--original-wav`, `--reference-text-file`; thêm `--greedy` để tắt sampling. Mặc định sampling theo source gốc: audio temperature 0.8/top-k 250, text temperature 0.7/top-k 25. Wrapper chỉ chuyển config lookup sang file local, không sửa LMGen/dep_q hoặc generation loop.
