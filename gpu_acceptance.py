"""Single-GPU fixed-ten acceptance using Tim construction, loss and checkpointer."""
import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--model-root', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--required-loss-ratio', type=float, default=.8)
    parser.add_argument('--generation-frames', type=int, default=40)
    parser.add_argument('--allow-tim-objective', action='store_true',
                        help='Explicitly accept Tim dep_q=16 objective, not workspace agent-only objective')
    opts = parser.parse_args(argv)
    if not opts.allow_tim_objective:
        parser.error('Tim dep_q=16 differs from workspace objective; explicit --allow-tim-objective required')
    if opts.steps < 10 or not 0 < opts.required_loss_ratio < 1 or opts.generation_frames < 1:
        parser.error('steps >= 10, 0 < loss ratio < 1 and positive generation frames required')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('single-GPU acceptance only')
    from tim_compat.local_checkpoint import LocalAssets, LocalCheckpointInfo
    assets = LocalAssets.resolve(opts.model_root)
    records = [json.loads(line) for line in opts.manifest.read_text().splitlines() if line.strip()]
    if len(records) != 10 or len({r['path'] for r in records}) != 10:
        parser.error('manifest must contain exactly ten distinct prepared conversations')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(ROOT / 'moshi-finetune'))
    import numpy as np
    import sphn
    import torch
    import torch.distributed as dist
    from moshi.models import loaders
    from finetune.args import TrainArgs
    from finetune.wrapped_model import get_fsdp_model
    from finetune.data.interleaver import Interleaver, InterleavedTokenizer, Batch
    from finetune.loss import compute_loss_with_mask
    from finetune.checkpointing import Checkpointer
    from finetune.utils import TrainState
    from tim_compat.tokenization import ObservedTokenizer, digest_file

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; this is not a CPU acceptance substitute')
    args = TrainArgs.load(str(opts.config), drop_extra_fields=False)
    if not args.lora.enable or args.full_finetuning or args.lora.ft_embed:
        raise ValueError('acceptance requires LoRA-only training without embedding tuning')
    if not args.system_prompt.enable or args.neftune_alpha != 0:
        raise ValueError('hybrid prompts required; augmentation must be disabled')
    opts.output.mkdir(parents=True, exist_ok=False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(0)
    rendezvous = tempfile.NamedTemporaryFile(delete=False)
    rendezvous.close()
    dist.init_process_group('nccl', init_method='file://' + rendezvous.name, rank=0, world_size=1)
    try:
        info = LocalCheckpointInfo(assets, loaders)
        model = get_fsdp_model(args, info)
        mimi = info.get_mimi(device='cuda').eval()
        for parameter in mimi.parameters():
            parameter.requires_grad = False
        spm = info.get_text_tokenizer()
        interleaver = Interleaver(spm, mimi.frame_rate, model.text_padding_token_id,
            model.end_of_text_padding_id, model.zero_token_id, keep_main_only=True)
        tokenizer = InterleavedTokenizer(mimi, interleaver, args.duration_sec,
            system_prompt_enabled=True, audio_silence_frames=args.system_prompt.audio_silence_frames,
            prompt_budget_frames=args.system_prompt.prompt_budget_frames)
        samples = []
        with (opts.output / 'tokenization.jsonl').open('x') as report:
            observed = ObservedTokenizer(tokenizer, digest_file(assets.mimi_weights),
                                        cache_dir=opts.output / 'tokens', report=report)
            for record in records:
                path = Path(record['path']).resolve()
                metadata = json.loads(path.with_suffix('.json').read_text())
                if not metadata.get('text_prompt') or not metadata.get('voice_prompt'):
                    raise ValueError(f'missing hybrid prompt: {path}')
                wav, sr = sphn.read(str(path))
                assert wav.ndim == 2 and wav.shape[0] == 2, 'LEFT=agent, RIGHT=user stereo required'
                if sr != mimi.sample_rate:
                    wav = sphn.resample(wav, src_sample_rate=sr, dst_sample_rate=mimi.sample_rate)
                wav = wav[:, :int(tokenizer.chunk_step_sec * mimi.sample_rate)]
                reference = tokenizer(wav, 0., str(path))
                miss = observed(wav, 0., str(path))
                hit = observed(wav, 0., str(path))
                for candidate in (miss, hit):
                    assert torch.equal(reference.codes, candidate.codes), 'cache/reference code mismatch'
                    assert reference.prompt_length == candidate.prompt_length
                    assert (reference.context_mask is None) == (candidate.context_mask is None)
                    if reference.context_mask is not None:
                        assert torch.equal(reference.context_mask, candidate.context_mask)
                assert reference.codes.shape[:2] == (1, 17)
                assert 0 < reference.prompt_length < reference.codes.shape[-1]
                samples.append(reference)
        print(f'Fixed 10, start=0, no shuffle/augmentation; first shape={samples[0].codes.shape}, '
              f'prompt={samples[0].prompt_length}; LEFT=agent RIGHT=user', flush=True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        assert trainable and all('lora' in n for n, p in model.named_parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim.lr,
            betas=(.9, .95), eps=1e-8, weight_decay=args.optim.weight_decay)

        def objective(sample):
            batch = Batch.collate([sample])
            conditions = model.condition_provider.prepare(batch.condition_attributes) if batch.condition_attributes else None
            output = model(codes=batch.codes, condition_tensors=conditions)
            text = compute_loss_with_mask(output.text_logits, batch.codes[:, :model.audio_offset],
                output.text_mask, mode='text', text_padding_weight=args.text_padding_weight,
                text_padding_ids={model.text_padding_token_id, model.end_of_text_padding_id},
                prompt_lengths=batch.prompt_lengths, context_masks=batch.context_masks)
            audio, per_cb = compute_loss_with_mask(output.logits,
                batch.codes[:, model.audio_offset:model.audio_offset+model.dep_q], output.mask,
                mode='audio', first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                prompt_lengths=batch.prompt_lengths, return_per_codebook=True)
            total = text + args.audio_loss_weight * audio if args.audio_loss_weight > 0 else text
            if args.lora_l2_weight > 0:
                total = total + args.lora_l2_weight * sum(p.pow(2).sum() for n, p in
                    model.named_parameters() if p.requires_grad and 'lora_B' in n)
            return total, text, audio, per_cb

        def mean_loss():
            model.eval()
            with torch.no_grad():
                value = sum(objective(sample)[0].item() for sample in samples) / 10
            model.train()
            return value

        initial = mean_loss()
        with (opts.output / 'metrics.jsonl').open('x') as metrics:
            for step in range(opts.steps):
                optimizer.zero_grad(set_to_none=True)
                total, text, audio, per_cb = objective(samples[step % 10])
                assert torch.isfinite(total), 'nonfinite loss'
                total.backward()
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                assert torch.isfinite(norm), 'nonfinite gradient norm'
                optimizer.step()
                row = dict(step=step+1, total=total.item(), text=text.item(), audio=audio.item(),
                    per_codebook=per_cb, lr=optimizer.param_groups[0]['lr'], grad_norm=norm.item(),
                    trainable_parameters=sum(p.numel() for p in trainable),
                    peak_gpu_bytes=torch.cuda.max_memory_allocated())
                metrics.write(json.dumps(row)+'\n')
                metrics.flush()
                print(json.dumps(row), flush=True)
        final = mean_loss()
        model.eval()
        with torch.no_grad():
            expected = objective(samples[0])[0].detach().clone()
        state = TrainState(opts.steps, step=opts.steps)
        checkpointer = Checkpointer(model, state, opts.output, info.lm_config, optimizer=optimizer)
        checkpointer.save_checkpoint(True, dtype=getattr(torch, args.param_dtype))
        adapter = checkpointer.dst_dir / 'lora.safetensors'
        del optimizer, checkpointer, trainable, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        model = get_fsdp_model(args, info, resume_lora_path=str(adapter))
        model.eval()
        with torch.no_grad():
            reloaded = objective(samples[0])[0]
        torch.testing.assert_close(expected, reloaded, rtol=1e-4, atol=1e-4)
        # Native current-model generation, with the original LMGen prompt phases.
        from moshi.models.lm import LMGen
        from finetune.data.interleaver import wrap_with_system_tags
        metadata_path = Path(records[0]['path']).resolve().with_suffix('.json')
        metadata = json.loads(metadata_path.read_text())
        voice = Path(metadata['voice_prompt'])
        if not voice.is_absolute():
            voice = metadata_path.parent / voice
        generator = LMGen(model, sample_rate=mimi.sample_rate, device='cuda',
            audio_silence_frame_cnt=args.system_prompt.audio_silence_frames, use_sampling=False)
        generator.load_voice_prompt(str(voice))
        generator.text_prompt_tokens = spm.encode(wrap_with_system_tags(metadata['text_prompt']))
        generated = []
        # Tim's dep_q=16 model has no external user slots in standard LMGen.
        # Do not silently rebuild dep_q=8 or force the user target as agent audio.
        if model.n_q - model.dep_q != 8:
            summary = dict(initial_loss=initial, final_loss=final, ratio=final/initial,
                adapter=str(adapter), reload_loss=reloaded.item(), inference='BLOCKED',
                reason='native LMGen user slots = n_q - dep_q; Tim 16 - 16 = 0, requires 8',
                command=sys.argv, peak_gpu_bytes=torch.cuda.max_memory_allocated())
            (opts.output / 'summary.json').write_text(json.dumps(summary, indent=2))
            raise RuntimeError(summary['reason'] + '; no inference acceptance claimed')
        with torch.no_grad(), mimi.streaming(1), generator.streaming(1):
            generator.step_system_prompts(mimi)
            mimi.reset_streaming()
            user = samples[0].codes[:, 9:17, samples[0].prompt_length:]
            for frame in range(min(opts.generation_frames, user.shape[-1])):
                tokens = generator.step(user[..., frame:frame+1])
                if tokens is not None:
                    generated.append(tokens.cpu())
        assert generated, 'native inference produced no frames'
        torch.save(torch.cat(generated, dim=-1), opts.output / 'generated_tokens.pt')
        summary = dict(initial_loss=initial, final_loss=final, ratio=final/initial,
            required_ratio=opts.required_loss_ratio, adapter=str(adapter),
            reload_loss=reloaded.item(), inference_frames=len(generated),
            config=str(opts.config.resolve()), manifest=str(opts.manifest.resolve()),
            torch=torch.__version__, gpu=torch.cuda.get_device_name(),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), command=sys.argv,
            objective='UNMODIFIED TIM: dep_q=16; not agent-only PersonaPlex objective')
        (opts.output / 'summary.json').write_text(json.dumps(summary, indent=2))
        assert final < initial * opts.required_loss_ratio, 'ten-sample overfit threshold failed'
    finally:
        dist.destroy_process_group()
        Path(rendezvous.name).unlink(missing_ok=True)


if __name__ == '__main__':
    main()