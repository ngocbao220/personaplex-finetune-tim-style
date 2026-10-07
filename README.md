# PersonaPlex × OtoSpeech — fine-tuning theo Tim

Fine-tune **`nvidia/personaplex-7b-v1` bằng LoRA** trên conversations OtoSpeech đã chuẩn bị sẵn. Project giữ trainer Tim và runtime PersonaPlex được copy, bổ sung bridges cho checkpoint local, prepared data, filter sample, token cache và kiểm tra training/inference.

**Mục tiêu đầu tiên:** 10 conversations cố định → tokenize/prompt/mask → overfit → save/reload adapter → inference với voice prompt + text prompt. Chưa mở rộng dataset hay distributed training trước khi đường đi này được xác nhận.

> **Trạng thái:** đã có CPU tests và GPU acceptance scripts. Chưa có kết quả GPU thật chứng minh end-to-end training, native generation hoặc offload/restore trong periodic free-running. Không coi CPU tests là bằng chứng model đã train thành công.

## 1. Nguyên tắc và data contract

- Khởi tạo từ **PersonaPlex**, không phải vanilla Moshi.
- Stereo conversation: **LEFT = agent, RIGHT = user**. User audio là context; agent text/audio là target.
- Prepared audio dùng 24 kHz; transcript có word-level timestamps và speaker labels.
- Voice/text system prompt là conditioning, vùng prompt phải được mask khỏi loss.
- Config mẫu đặt `first_codebook_weight_multiplier: 50.0` để non-semantic/semantic = `0.02`, và `text_padding_weight: 0.3`. Không âm thầm thay objective để làm loss đẹp hơn.
- Training dùng assets local và chế độ offline; thiếu file phải báo lỗi, không tự tải model từ Hugging Face.
- Forced alignment, sửa SRT, chọn voice-prompt region và sinh prompt bằng LLM **không thuộc training server**. Synthetic-data scripts upstream được giữ làm tham khảo, không phải quy trình OtoSpeech ở đây.

## 2. Môi trường và assets

Cần Linux/CUDA cho GPU acceptance, một GPU được expose, Python ≥ 3.10 và **Moshi training runtime tương thích patches của Tim**. Dependency declaration nằm trong `moshi-finetune/pyproject.toml`: Torch 2.6, Triton, PyYAML, safetensors, sphn và Moshi từ Git.

**Không dùng inference-only PersonaPlex loader để train.** Training loader phải hỗ trợ `lm_kwargs`/`lm_kwargs_overrides` cho LoRA. Dependency Moshi upstream chưa pin commit; cài package nguyên trạng không chứng minh runtime đúng với Tim. Chuẩn bị/audit môi trường trước khi chạy, lưu versions/commit vào report. Trên server offline, dùng environment/wheels đã chuẩn bị; không chạy cài đặt gây download ngầm.

Assets cần provision trước:

```text
/absolute/models/personaplex-7b-v1/
├── model.safetensors
├── tokenizer-e351c8d8-checkpoint125.safetensors
├── tokenizer_spm_32k_3.model
└── config.json
```

Local training bridge có thể không cần `config.json`, nhưng **native free-running yêu cầu config local** qua `moshi_paths.config_path`. Config phải phù hợp loader; sparse metadata không thay thế config inference hợp lệ. File tồn tại không đồng nghĩa contents/runtime đã được xác minh trên GPU.

Các lệnh dưới dùng:

```bash
export PROJECT=/Users/ngocbao/Documents/Document/research/main/speech/exps/finetune-personalplex/personaplex-finetune-tim-style
cd "$PROJECT"
```

Trên server khác, đổi `PROJECT` thành **đường dẫn tuyệt đối** tới project. Thay các `/absolute/...` bên dưới bằng assets/data/output thật.

## 3. Quick start: kiểm tra 10 conversations

### Bước 1 — cấu hình

Chỉnh `$PROJECT/config.full.yaml`:

- `acceptance.model_root`: snapshot PersonaPlex local.
- `acceptance.prepared_manifest`: manifest OtoSpeech đã chuẩn bị.
- `moshi_paths`: weights, Mimi, tokenizer và config local.
- `acceptance.inference`: user mono WAV, voice prompt, text prompt và original stereo window nếu chạy generation.

Đây là **config cho acceptance harness**, gồm section `acceptance`; không truyền nguyên file vào trainer Tim. Harness export dữ liệu, tạo native config và bind manifest. `data.train_data`/`eval_data` trống trong file mẫu sẽ được harness điền.

Giữ `batch_size: 1`, `data.shuffle: false`, LoRA enabled, embedding frozen, `neftune_alpha: 0`, `gen_eval.enable: false`. Harness chọn tập liên tiếp cố định, không random split và không augmentation. Crop/tokenization vẫn do Tim thực hiện; cần kiểm tra parity thực tế, không mặc định coi mọi crop đã deterministic.

### Bước 2 — CPU tests

```bash
python3 -m unittest discover -s "$PROJECT/tests" -v
```

Tests bao phủ bridge/config/filter/cache, source integrity và audio contract; không thay cho GPU acceptance.

### Bước 3 — GPU smoke tests theo thứ tự

Mỗi lệnh cần **output directory mới**. Dừng và đọc report/log khi một phase lỗi.

```bash
# Token streams, prompt length, masks và cache parity.
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/check_interleaver_parity.py" \
  --config "$PROJECT/config.full.yaml" --sample-index 0 \
  --output-dir /absolute/runs/parity/interleaver

# Loss, LoRA, optimizer và scheduler sau một optimizer step.
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/check_one_step_parity.py" \
  --config "$PROJECT/config.full.yaml" --sample-index 0 \
  --output-dir /absolute/runs/parity/one_step

# Overfit 10 conversations và reload checkpoint trong process mới.
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/run_gpu_acceptance.py" \
  --config "$PROJECT/config.full.yaml" --sample-index 0 \
  --num-samples 10 --max-steps 100 --checkpoint-step 100 \
  --output-dir /absolute/runs/gpu_acceptance
```

Mini-overfit mặc định yêu cầu teacher-forced eval loss cuối / đầu **< 0.8** (`--required-loss-ratio`). Reload kiểm tra LoRA tensors và loss trước/sau save. Đây là tiêu chí harness, không phải kết quả đạt sẵn. Native evaluator tối đa 40 batches: chọn duration/chunk sao cho fixed set được bao phủ.

Chạy toàn bộ bằng orchestrator:

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/_run_all.py" \
  --config "$PROJECT/config.full.yaml" --runs-dir /absolute/runs/all_checks
```

Orchestrator chạy tuần tự rồi native free-running sau acceptance nếu có `acceptance.inference`. Muốn bỏ generation trong lần đầu, bỏ section `acceptance.inference`; interval `0` **không tắt** smoke inference sau acceptance. Report tổng ở `/absolute/runs/all_checks/gpu_acceptance_report.json`.

## 4. Prepared data và launcher trực tiếp

Manifest JSONL tối giản:

```json
{"sample_id":"conversation_001","sample_dir":"conversation_001"}
```

Một sample directory chứa:

```text
conversation_001/
├── conversation.wav       # stereo 24 kHz PCM, LEFT=agent RIGHT=user
├── words.json             # word, start, end, speaker: agent/user
├── metadata.json          # agent_channel: left, user_channel: right, text_prompt
└── voice_prompt_left.wav  # agent voice prompt mono 24 kHz
```

`prompt.txt` được dùng khi metadata không có text prompt; filename voice prompt legacy cũng được hỗ trợ. Adapter validate và xuất sidecars/manifest cho Tim, **không** alignment hay sửa dữ liệu lỗi. Giữ prepared sources bất biến: export có thể symlink tới audio nguồn.

```bash
python "$PROJECT/prepare_data.py" \
  --manifest /absolute/prepared/manifest.jsonl \
  --output /absolute/exports/tim-prepared
```

Đặt `data.train_data` của **native Tim YAML** thành manifest do lệnh in ra; cấu hình eval riêng theo mục đích. Thêm `--config /absolute/configs/train-native.yaml` có thể xuất derived `train.yaml`, nhưng **chỉ đổi train_data**, không tự bật prompts hay đổi eval/shuffle/loss.

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 \
  "$PROJECT/train_local.py" \
  --model-root /absolute/models/personaplex-7b-v1 \
  --config /absolute/configs/train-native.yaml
```

Native config không có section `acceptance`; bind exported Tim manifest, không đưa prepared manifest trực tiếp vào Tim loader. Giữ `system_prompt.enable: true` để dùng prepared voice/text prompts. Chạy acceptance trên 10 samples trước khi train quy mô lớn.

## 5. Filter invalid data

**Đã có filter, nhưng mặc định không bật.** Nó kiểm tra actual Tim `Sample` sau tokenize/crop/prompt/injection, không tái tạo sequence và không thay nội dung sample hợp lệ.

| Kiểm tra | Lý do loại |
|---|---|
| Integer tensor `[1,17,T]`, `T > 0` | `invalid_stream_layout` |
| Prompt length nằm trong sequence | `invalid_prompt_length` |
| Context mask đúng shape/dtype/device | `invalid_context_mask` |
| Token IDs trong vocabulary hoặc sentinel padding hợp lệ | `text_token_out_of_bounds` / `audio_token_out_of_bounds` |
| Agent text ngoài prompt/injection, nếu yêu cầu | `no_retained_agent_text` |

Policy JSON cần `text_cardinality`, `audio_cardinality`, `nonlexical_text_ids` lấy từ **model/tokenizer/runtime thực tế**. Optional: `zero_padding_id` (mặc định `-1`), `require_agent_text` (`true`), `max_consecutive_rejections` (`1000`). Không copy magic IDs từ tokenizer khác.

Với acceptance bridge, thêm vào section `acceptance` của config:

```yaml
filter_policy: /absolute/configs/filter-policy.json
```

Với launcher trực tiếp, thêm cả hai flags:

```text
--filter-policy /absolute/configs/filter-policy.json
--filter-report-dir /absolute/runs/new-filter-reports
```

Report `rejects-rank-0.jsonl` ghi candidate index, lý do và provenance nếu có. Sample hợp lệ giữ identity/nội dung/thứ tự; filter áp dụng cả train/eval. Quá giới hạn loại liên tiếp sẽ dừng, không loop vô hạn.

**Giới hạn:** không phát hiện đầy đủ noise/clipping/silence hay dense token overflow trước crop; lỗi đọc/tokenize vẫn propagate, không âm thầm skip. Candidate indices không phải ID ổn định toàn dataset. Uneven rejection giữa ranks chưa được acceptance. Bật filter có thể giảm tập giữ lại: kiểm tra report/coverage, không đánh đồng 10 conversations đầu vào với 10 samples hợp lệ.

## 6. Native free-running: original / base / checkpoint

Free-running khác teacher-forced eval: model tự sinh agent audio/text từ user audio và prompts. Wrapper gọi bundled PersonaPlex `moshi.offline`; base và merged checkpoint chạy tuần tự trong process riêng.

Trong `acceptance.inference`, chuẩn bị:

- `input_wav`: **mono user RIGHT channel**, không phải stereo conversation.
- `original_wav`: stereo original đã cắt đúng cùng cửa sổ.
- `voice_prompt`: agent WAV hoặc native embeddings `.pt`.
- `text_prompt_file`: system prompt UTF-8.
- `sample_id`, `window_start_sec`; optional `reference_text_file`.

```bash
CUDA_VISIBLE_DEVICES=0 python "$PROJECT/scripts/run_free_running_inference.py" \
  --config "$PROJECT/config.full.yaml" \
  --checkpoint /absolute/runs/gpu_acceptance/tim_run/checkpoints/checkpoint_000100/consolidated \
  --step 100 --output-dir /absolute/runs/free_running_100
```

Checkpoint cần `lora.safetensors` và `config.json` có LoRA rank/scaling. Wrapper merge bằng pipeline gốc, yêu cầu local inference config. `--step` chỉ gán provenance ở standalone, không đặt lịch. `--greedy` tắt sampling; mặc định native audio temperature/top-k = `0.8/250`, text = `0.7/25`.

```text
/absolute/runs/free_running_100/
├── dialogue_original.wav
├── dialogue_base.wav
├── dialogue_step.wav
├── manifest.json
├── base/                  # agent.wav, native text pieces và summary
└── current/               # agent.wav, text pieces, summary và merge artifacts
```

Ba dialogue WAV đều stereo **LEFT=agent, RIGHT=user**, resampled 24 kHz. Wrapper từ chối audio rỗng/nonfinite, độ dài không khớp, hoặc original RIGHT khác user input (`atol=1e-4`). Transcript lấy native text pieces, **không phải ASR**; CER/WER hiện `null`, chưa tính.

### Chạy định kỳ trong training

```yaml
acceptance:
  inference:
    free_running_every_steps: 20
    # Giữ các input/prompt/window fields như trên.
```

- `0`: tắt hook trong training, mặc định.
- `N > 0`: evaluation trước bước train đầu tiên và sau mỗi **N optimizer steps**, sau optimizer/scheduler update; không đếm microbatches.
- Không tự chạy cuối nếu step cuối không chia hết cho N. Khi resume, lần đầu dùng trạng thái resume với nhãn step thực tế, không giả step 0.
- Mini-overfit route `bridge` tự nhận config; one-step parity và reload không chạy hook.
- Launcher trực tiếp cần thêm `--free-running-config /absolute/configs/acceptance.yaml`; `--config` vẫn là native training YAML riêng.

Kết quả ở `<run_dir>/free_running/step_000000/`, `step_000020/`, ...; snapshots riêng ở `adapter_000000/`, ... không thay checkpoint resume. Base baseline sinh một lần trong session rồi tái sử dụng; kiểm tra input hashes, prompts, seed và generation signature trước reuse. `--baseline-dir` hỗ trợ reuse ở standalone.

**Chi phí/giới hạn:** training tạm dừng đồng bộ; hook offload tensor storage của model/Mimi, gradients, FP32 master weights và optimizer sang CPU, restore trong `finally`, giữ Parameter identities và Torch RNG. Cần đủ CPU RAM, GPU headroom cho tensors còn sống và disk cho adapters/merged models. Offload không bảo đảm giải phóng toàn bộ VRAM. Chỉ hỗ trợ single GPU, LoRA, `do_ckpt: true`, `gen_eval.enable: false`. Lỗi inference làm training fail-fast; artifacts no-overwrite nên rerun tránh directory trùng. **Offload/restore và periodic generation chưa được kiểm chứng với model thật trên GPU.**

## 7. Kiểm chứng và tiêu chí hoàn tất

Lần CPU verification gần nhất được ghi nhận: **57 tests, OK**, gồm 4 periodic-hook tests và 6 acceptance/audio-script tests. Đây là ảnh chụp kết quả, không cam kết số test giữ nguyên. Kiểm tra lại bằng unittest ở mục 3.

Milestone chỉ hoàn tất khi có bằng chứng GPU: 10 samples hợp lệ và coverage đủ; prompt có mặt nhưng masked; stream mapping đúng; forward/backward thành công; LoRA nhận gradient và base frozen; loss fixed set giảm rõ; adapter save/reload đúng; native inference chạy với voice/text prompts. Lưu exact commands/config, losses, GPU memory, versions và remaining issues. Hiện không tuyên bố đã đạt milestone.

## 8. Cấu trúc và tài liệu

| Thành phần | Vai trò |
|---|---|
| `moshi-finetune/` | Trainer Tim được giữ làm reference |
| `personaplex/` | Bundled native inference runtime |
| `tim_compat/` | Bridges local assets/data/filter/cache và periodic hook |
| `train_local.py`, `prepare_data.py` | Launcher local và prepared export |
| `scripts/` | GPU checks, mini-overfit/reload, free-running |
| `pipeline/` | Upstream utilities; native LoRA merge được wrapper tái sử dụng |
| `tests/` | CPU regression/contract tests |

Tài liệu chi tiết dưới `$PROJECT` (Markdown hỗ trợ được gom vào `docs/`):

- [Quy trình validation GPU](docs/VALIDATION.md): 8 cổng kiểm tra, tiêu chí đạt và báo cáo.
- `scripts/README.md`: tham số và lệnh GPU acceptance.
- `docs/PREPARED_DATA.md`: prepared schema/export/config binding.
- `docs/SAMPLE_FILTER.md`: policy, rejection và giới hạn filter.
- `docs/LOCAL_CHECKPOINT.md`: loader local/runtime contract; đọc các giới hạn dữ liệu cũ cùng prepared-data documentation hiện tại.
- `docs/REFERENCE_BASELINE.md`: source provenance/integrity.
- `docs/README.upstream.md`: README upstream giữ nguyên bytes, được integrity tests kiểm tra; không phải hướng dẫn OtoSpeech hiện tại.
- `docs/TIM_STYLE_CHANGES.md`: thay đổi và trạng thái verification.
- `docs/HYDRA.md`: launcher Hydra tùy chọn, không cần cho quick start.
- `docs/THIRD_PARTY_NOTICES.md`, `LICENSE`: nguồn upstream/giấy phép. Kết quả pharma/synthetic-data upstream không phải kết quả OtoSpeech của project này.