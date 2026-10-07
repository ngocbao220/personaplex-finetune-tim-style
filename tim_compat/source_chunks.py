"""Reject quarantined source windows before native tokenization, not after it."""
import json
import math
from contextlib import contextmanager
from pathlib import Path


class SourceChunkGate:
    def __init__(self):
        self.metadata = {}

    def rejection(self, path, start, end):
        # Keep the exported WAV's logical path: its symlink target has no Tim sidecar.
        path = Path(path).absolute()
        if path not in self.metadata:
            self.metadata[path] = json.loads(path.with_suffix('.json').read_text())
        metadata = self.metadata[path]
        errors = metadata.get('source_alignment_errors', [])
        if not errors:
            return None
        duration = metadata['audio_duration_sec']
        end = min(end, duration)
        for error in errors:
            word = error['word']
            low, high = sorted((float(word['start']), float(word['end'])))
            # No audio interval exists outside the source: reject the adjacent edge.
            affected = ((high <= 0 and start == 0) or
                        (low >= duration and math.isclose(end, duration, rel_tol=0, abs_tol=1e-9)) or
                        (max(start, low) < min(end, high)) or
                        (low == high and start <= low < end))
            if affected:
                return dict(stage='prepared_source_chunk', reason='timestamp_out_of_bounds',
                    sample_id=metadata.get('sample_id', path.stem), path=str(path),
                    chunk_start=start, chunk_end=end, audio_duration_sec=duration,
                    source_errors=[error], triggering_error=error)
        return None


def record_rejection(report, rejection):
    if report is not None:
        report.write(json.dumps(rejection, ensure_ascii=False) + '\n')
        report.flush()
    print(f"Rejected source chunk {rejection['sample_id']}: "
          f"{rejection['chunk_start']}..{rejection['chunk_end']}s; "
          f"reason={rejection['reason']}", flush=True)


@contextmanager
def source_chunk_loader(data_loader, *, report=None, report_path=None,
                        max_consecutive_rejections=1000):
    """Native iterators yield a private sentinel; remove it before batching/filtering."""
    original = data_loader.build_dataset
    skipped = object()
    owned_report = None

    def record(rejection):
        nonlocal owned_report
        sink = report
        if sink is None and report_path is not None:
            if owned_report is None:
                # Native trainer creates run_dir first. Append on resume, never truncate.
                owned_report = Path(report_path).open('a', encoding='utf-8')
            sink = owned_report
        record_rejection(sink, rejection)

    class TokenizerGate:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer
            self.gate = SourceChunkGate()

        def __getattr__(self, name):
            return getattr(self.tokenizer, name)

        def __call__(self, wav, start, path):
            rejection = self.gate.rejection(path, start, start + wav.shape[-1] / self.mimi.sample_rate)
            if rejection is not None:
                record(rejection)
                return skipped
            return self.tokenizer(wav, start, path)

    def filtered(*args, **kwargs):
        positional = list(args)
        if 'instruct_tokenizer' in kwargs:
            kwargs['instruct_tokenizer'] = TokenizerGate(kwargs['instruct_tokenizer'])
        else:
            positional[1] = TokenizerGate(positional[1])
        consecutive = 0
        rejected = yielded = 0
        for sample in original(*positional, **kwargs):
            if sample is skipped:
                consecutive += 1
                rejected += 1
                if consecutive >= max_consecutive_rejections:
                    raise RuntimeError(f'source chunk filter reached maximum consecutive rejections; see {report_path or "source chunk report"}')
            else:
                consecutive = 0
                yielded += 1
                yield sample
        if rejected and not yielded:
            raise RuntimeError(f'no valid source chunk in finite dataset; see {report_path or "source chunk report"}')

    data_loader.build_dataset = filtered
    try:
        yield
    finally:
        data_loader.build_dataset = original
        if owned_report is not None:
            owned_report.close()
