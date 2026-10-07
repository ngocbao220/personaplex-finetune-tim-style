# GPU acceptance commands

Chạy các lệnh từ thư mục `personaplex-finetune-tim-style/`, dùng config đã chuẩn bị và một GPU. Mỗi lần chạy cần output directory mới.

## Config và tham số

Dùng [config.full.yaml](../config.full.yaml), có comment giải thích từng tham số. Config mặc định đã điền đường dẫn server OtoSpeech/model/probe; kiểm tra các file này tồn tại trước khi chạy. Full training synthetic dùng [train_synthetic_server.sh](train_synthetic_server.sh), xem [hướng dẫn server](../docs/train_synthetic_server.md). File này dành cho acceptance harness, không chạy trực tiếp bằng trainer.

| Tham số lệnh | Ý nghĩa |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | Chỉ dùng GPU số 0. |
| `--config` | File YAML chứa config Tim và section `acceptance`. |
| `--sample-index` | Vị trí conversation bắt đầu trong manifest, tính từ 0. |
| `--output-dir` | Thư mục mới để lưu artifacts và logs. |
| `--num-samples` | Số conversations liên tiếp dùng cho mini-overfit; bỏ qua tham số này thì smoke chọn 1, full-set chọn 10. |
| `--max-steps` | Số optimizer steps và horizon OneCycle của phase 3. |
| `--checkpoint-step` | Khoảng cách lưu checkpoint; dùng bằng `--max-steps` để kiểm tra checkpoint cuối. |
| `--smoke-chunks` | Phase 3: 1–40 chunks hợp lệ cố định cho train/eval/reload; 0=tắt smoke. Ghi đè `acceptance.smoke_chunks` trong YAML. |

Config mẫu dùng loss weights tường minh cho acceptance; nếu so sánh với một run Tim cụ thể, giữ đúng weights của run đó ở cả hai phía. Native evaluation tối đa 40 batches: chọn chunk duration sao cho toàn bộ fixed set nằm trong giới hạn này.

## 1. Interleaver và cache parity

Kiểm tra token streams, prompt length và masks giữa tokenizer Tim trực tiếp và token cache.

Phase này **tự lọc invalid chunks**, không cần `acceptance.filter_policy`. `--sample-index` vẫn chọn một conversation; script duyệt native chunks theo thứ tự, bỏ chunk lỗi và chạy parity trên chunk hợp lệ đầu tiên. Không chuyển sang conversation khác và không sửa token, prompt hay mask.

- Kiểm tra layout, prompt/mask và text/audio IDs bằng vocabulary cùng sentinel của runtime thực tế, kể cả text/prompt trước crop.
- Loại `text_overflow` khi token bị overwrite, còn pending/không bắt đầu được trong dialogue budget, hoặc lexical text bị mất sau crop/context insertion. Loại `prompt_overflow` khi prefix bị clamp và `context_overflow` khi context tokens bị cắt.
- Timestamp không hữu hạn, âm, đảo thứ tự hoặc vượt audio nguồn làm preflight dừng; không tự bỏ word lỗi. Word giao biên chunk bình thường không bị coi là timestamp invalid.

`rejections.jsonl` ghi sample/chunk identity, cửa sổ và lý do ngay khi loại; `observations.jsonl` ghi diagnostics/cache của các chunk đã xét. `summary.json` ghi số chunk đã xét/loại, chunk được chọn và runtime filter policy. Hết chunk hợp lệ thì thoát lỗi với `reason: no_valid_chunk`; lỗi nguồn có `stage: prepared_source`. Runtime/CUDA/cache errors vẫn dừng, không bị skip.

Nếu prompt chiếm nhiều frame, đặt `system_prompt.prompt_budget_frames` theo prompt thực tế (ít nhất bằng prefix dài nhất, nhỏ hơn tổng chunk frames). Budget quá nhỏ có thể khiến text cuối cửa sổ bị crop và mọi chunk đều bị loại. Xem diagnostics để sửa config/data; script không tự thay chunking. Filter này chỉ áp dụng cho phase 1, không tự bật filter training.

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

Config cần `batch_size: 1`. Với conversation dài khoảng 900 giây, dùng smoke nhỏ để tránh eval hàng chục/hàng trăm batches:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_gpu_acceptance.py \
  --config config.full.yaml \
  --num-samples 1 \
  --smoke-chunks 8 \
  --max-steps 100 \
  --sample-index 0 \
  --checkpoint-step 100 \
  --output-dir runs/gpu_smoke
```

Script lấy tối đa 8 chunks hợp lệ đầu tiên trong conversations đã chọn, giữ chunk duration/prompt/loss như config. Train lặp tuần tự trên **chính tập đó**; baseline, final và reload eval cùng token snapshot, không tokenize lại. Mỗi chunk được kiểm tra bằng filter out-of-bound/overflow như phase 1. Nếu nguồn hết trước 8, dùng số chunk hợp lệ thực tế và ghi rõ; không có chunk hợp lệ thì dừng.

Artifacts trong `bridge/`: `smoke_selection.json` ghi sample IDs, offsets, số chunk đã xét/loại và snapshot SHA-256; `smoke_samples.pt` chứa native Samples CPU; `smoke_rejections.jsonl` ghi lý do loại. `baseline.json`, `final_eval.json`, `pre_save.json`, `reload.json` ghi cùng hash và số batches. `comparison.json` có `coverage: fixed_chunk_smoke`: PASS chỉ áp dụng tập nhỏ đã chọn, **không chứng minh coverage toàn bộ conversations**.

`config.full.yaml` đặt `acceptance.smoke_chunks: 8`; CLI có thể đổi số lượng. `--smoke-chunks 0` giữ đường full-set cũ và vẫn dừng nếu hơn 40 eval batches. One-step parity không bị chuyển sang smoke subset. Không tự giảm số optimizer steps hoặc thay objective; `--max-steps` vẫn điều khiển horizon OneCycle. Smoke này kiểm tra teacher-forced train/save/reload, không thay native generation.

## 4. Chạy toàn bộ

Chạy tuần tự các kiểm tra trên và free-running nếu có `acceptance.inference`, dừng khi có lỗi và ghi `runs/gpu_acceptance_report.json`.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_gpu_checks.sh config.full.yaml
```

## 5. Free-running inference và lưu kết quả

Chạy base model và model đã merge adapter trong **hai process riêng**, gọi nguyên `moshi.offline` của PersonaPlex. Cả hai dùng cùng input, prompts, seed và sampling settings. Đây là inference smoke, không phải parity với LMGen của Moshi training. Chưa kiểm chứng trên GPU.

Input lấy từ `acceptance.inference`: `original_wav`/`input_file` hoặc `sample_id` trong prepared manifest. Dùng `user_channel: left/right`, `start_sec` và `window_seconds`; script tự tạo mono `user.wav` 24 kHz. CLI hỗ trợ `--sample-id`, `--start`/`--start-sec`, `--window-seconds`, `--input-file`/`--input-path`, `--user-channel`. Voice/text prompts lấy từ sample khi không override; file ngoài không có sample phải cung cấp đủ hai prompts. Xem mục 6 của README root cho ví dụ và quy tắc conditioning.

Kết quả trong output directory:
- `dialogue_original.wav`: cửa sổ conversation gốc.
- `dialogue_base.wav`: agent từ base model + user input.
- `dialogue_step.wav`: agent từ checkpoint + user input.
- `manifest.json`: sample/step, cửa sổ, seed, generation settings, transcript base/current, reference tùy chọn và tên audio files.
- `base/`, `current/`: mono `agent.wav`, native `agent_text.json`, summary và merge artifacts; logs ở `base.log`, `current.log`.

Dialogue WAV dùng 24 kHz; stereo giữ thứ tự kênh nguồn theo `user_channel`, ghi rõ trong manifest. Với nguồn mono, original giữ mono, base/current là LEFT=agent, RIGHT=user. Script fail nếu user không khớp kênh đã chọn hoặc độ dài các outputs khác nhau. Transcript ghép từ native text token pieces, không phải ASR. `data.vietnamese_text_mode` hỗ trợ `diacritics`, `no_diacritics`, `telex`; Telex giữ raw `agent.txt` và thêm `agent_unicode.txt`. Khi có `reference_text_file`, CER/WER so sánh native text trong representation đã chọn; thiếu reference thì để `null`. Mode được ghi trong manifest/checkpoint và baseline signature. Xem README root để export/train cùng mode.

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
