# Validation trên server GPU

Quy trình gồm **8 cổng kiểm tra**. Bước nào không đạt thì dừng, lưu logs và sửa trước khi chạy tiếp. Không có một lần chạy nào bảo đảm tuyệt đối code đúng; PASS không đồng nghĩa chất lượng hội thoại tốt.

**Trạng thái:** CPU regression tests đã được chạy; chưa có bằng chứng GPU end-to-end, native generation hay offload/restore với model thật. Các tiêu chí dưới đây là yêu cầu cần thực thi, không phải kết quả đã đạt.

## Chuẩn bị

Thay bằng đường dẫn tuyệt đối trên server:

```bash
export PROJECT=/absolute/path/personaplex-finetune-tim-style
export CONFIG="$PROJECT/config.full.yaml"
export RUNS=/absolute/runs/verification_01
cd "$PROJECT"
```

Mỗi phase/rerun cần output directory mới. Chỉnh `acceptance.model_root`, `acceptance.prepared_manifest`, `moshi_paths` và `acceptance.inference` theo assets thực tế. Acceptance YAML khác native trainer YAML; xem [README](../README.md) và [scripts](../scripts/README.md).

Lượt correctness đầu tiên: single GPU, LoRA enabled, `batch_size: 1`, `data.shuffle: false`, embedding frozen, `neftune_alpha: 0`, `gen_eval.enable: false`. Không augmentation/random split, seed cố định. **Không bật filter**, đặt `acceptance.inference.free_running_every_steps: 0` để tách lỗi training khỏi filtering/offload.

## 1. Environment và local assets

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python -c \
  'import torch; print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU count:", torch.cuda.device_count())'
```

**Đạt khi:** CUDA khả dụng, một GPU được expose; PersonaPlex weights, Mimi, SentencePiece và local inference config đầy đủ, không download ngầm. Training runtime phải hỗ trợ LoRA overrides của Tim, không dùng inference-only loader để train. GPU được nhận diện chưa chứng minh runtime tương thích.

Lưu versions/commit và assets provenance. Kiểm tra đủ VRAM, CPU RAM và disk cho offload/merge khi chạy periodic inference.

## 2. CPU tests và 10 prepared conversations

```bash
python3 -m unittest discover -s "$PROJECT/tests" -v
python "$PROJECT/prepare_data.py" \
  --manifest /absolute/prepared/manifest.jsonl \
  --output "$RUNS/prepared_export"
```

**Đạt khi:** tests pass; 10 conversations cố định hợp lệ, stereo 24 kHz **LEFT=agent, RIGHT=user**; timestamps/speaker labels đúng; agent voice prompt và text prompt có sẵn. Đọc ít nhất một human-readable sample inspection và lưu danh sách 10 IDs.

Export chỉ validate/convert prepared data, không alignment/sửa SRT/chọn voice regions. Crop phải deterministic hoặc không crop; không suy ra điều này từ `shuffle: false`. Xem [prepared contract](PREPARED_DATA.md).

## 3. Interleaver và cache parity

```bash
for INDEX in 0 1 2 3 4 5 6 7 8 9; do
  CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/check_interleaver_parity.py" \
    --config "$CONFIG" --sample-index "$INDEX" \
    --output-dir "$RUNS/interleaver_$INDEX" || exit 1
done
```

**Đạt khi:** cả 10 index đạt parity về token streams, prompt lengths và context masks giữa các routes được kiểm tra. Đọc inspection, xác nhận shape/layout và voice/text prompt hiện diện nhưng loss-masked. Index 0 PASS không chứng minh cả 10 conversations đúng.

## 4. Forward/backward và one-step parity

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/check_one_step_parity.py" \
  --config "$CONFIG" --sample-index 0 \
  --output-dir "$RUNS/one_step"
```

**Đạt khi:** forward/backward thành công; loss/gradient norm hữu hạn; LoRA nhận gradient và cập nhật; base frozen, không nhận gradient. Loss, LoRA parameters, optimizer/scheduler original và bridge đạt tiêu chí parity của script. Đọc reports, không chỉ exit code.

## 5. Overfit 10 samples, save và reload

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/run_gpu_acceptance.py" \
  --config "$CONFIG" --sample-index 0 --num-samples 10 \
  --max-steps 100 --checkpoint-step 100 \
  --output-dir "$RUNS/overfit"
```

**Đạt khi:** teacher-forced eval bao phủ fixed set; loss giảm rõ, mặc định **loss cuối / loss đầu < 0.8**; checkpoint lưu thành công; reload trong process mới đạt kiểm tra LoRA tensors và loss trước/sau save.

Native evaluator có giới hạn 40 batches: chọn duration/chunk để bao phủ tập mà không vượt cap. 100 steps chỉ là điểm bắt đầu, không bảo đảm overfit. Nếu không đạt, điều tra data/prompts/masks/objective; không hạ tiêu chuẩn để đổi FAIL thành PASS.

Giữ reference semantic/non-semantic và padded text weighting. Lưu total/text/semantic/non-semantic loss, learning rate, trainable parameter count và gradient norm. Ghi rõ metric runtime chưa cung cấp, không suy diễn số liệu.

## 6. Native inference standalone

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/run_free_running_inference.py" \
  --config "$CONFIG" \
  --checkpoint "$RUNS/overfit/tim_run/checkpoints/checkpoint_000100/consolidated" \
  --step 100 --output-dir "$RUNS/inference_100"
```

Chuẩn bị user mono WAV lấy từ RIGHT, original stereo cùng cửa sổ, voice prompt và text prompt. Checkpoint cần `lora.safetensors` và `config.json`; đổi đường dẫn nếu checkpoint step khác.

**Đạt khi:** có `dialogue_original.wav`, `dialogue_base.wav`, `dialogue_step.wav` và `manifest.json`; LEFT=agent/RIGHT=user, độ dài/user channel khớp, không empty/nonfinite. Manifest đúng checkpoint/sample/window/seed/prompts. Nghe cả ba WAV để phát hiện im lặng, nhiễu hoặc output vô nghĩa.

PASS chỉ xác nhận smoke/audio contract, không chứng minh tốt hơn base. Transcript là native text pieces, không phải ASR; CER/WER chưa được tính.

## 7. Filter invalid data riêng

Sau khi đường đi không-filter đã đạt:

1. Chuẩn bị policy với vocabulary bounds/token IDs từ runtime/tokenizer thực tế.
2. Bật `acceptance.filter_policy` bằng đường dẫn tuyệt đối tới policy JSON.
3. Kiểm tra sample hợp lệ và actual tokenized Sample lỗi có kiểm soát; không sửa prepared sources gốc.
4. Đọc `rejects-rank-0.jsonl` và retained coverage.

**Đạt khi:** lỗi bị loại đúng lý do, sample hợp lệ giữ nội dung/identity/thứ tự; không loại hết dataset; loader/tokenize errors vẫn propagate như thiết kế. Không đánh đồng 10 conversations đầu vào với 10 retained samples.

Filter không kiểm tra toàn diện noise/clipping/silence audio thô. Xem [policy và giới hạn](SAMPLE_FILTER.md).

## 8. Periodic inference và restore training

Chạy run mới, cùng fixed set/seed, tắt filter để tách periodic behavior. Đổi interval trong config:

```yaml
acceptance:
  inference:
    free_running_every_steps: 20
```

Đây là fragment; giữ các inference input/prompt fields còn lại.

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/run_gpu_acceptance.py" \
  --config "$CONFIG" --sample-index 0 --num-samples 10 \
  --max-steps 100 --checkpoint-step 100 \
  --output-dir "$RUNS/periodic_overfit"
```

**Đạt khi:** evaluations tại 0/20/40/60/80/100 optimizer steps; đúng provenance và snapshots riêng; baseline sinh một lần rồi reuse; không OOM; training tiếp tục sau mỗi evaluation, loss/gradient hữu hạn; save/reload cuối run vẫn đạt.

Xem `tim_run/free_running/step_000000/`, `step_000020/`, ... và adapter snapshots tương ứng. So sánh với periodic-disabled run ở bước 5, cùng config ngoại trừ interval/output: metrics và LoRA tensors cuối để phát hiện state/RNG drift. Điều tra mọi khác biệt; không mặc định bitwise equality trên CUDA khi chưa xác nhận deterministic kernels.

Hook chỉ hỗ trợ single-GPU LoRA, `do_ckpt: true`, `gen_eval.enable: false`. CPU offload không bảo đảm giải phóng toàn bộ VRAM; inference failure là fatal. Không chỉ kiểm tra audio được sinh mà bỏ qua training state sau restore.

## Báo cáo và điểm dừng

6 bước đầu xác nhận training → checkpoint → inference cốt lõi. Bước 7–8 xác nhận filter và periodic evaluation. Không đánh dấu GPU PASS bằng CPU tests.

Lưu exact commands/config snapshots, versions/commits/assets provenance, fixed sample IDs/crops/seeds, retained/rejected coverage, reports, loss đầu/cuối/ratio/component losses, LR/gradient/trainable count, peak GPU memory, CPU RAM/disk observations, checkpoint/reload results, audio/listening observations và PASS/FAIL/remaining issues từng cổng.

**Dừng sau milestone và báo cáo trước khi tăng dataset hoặc dùng distributed training.**