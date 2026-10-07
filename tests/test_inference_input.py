import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tim_compat.inference_input import select_source, audio_window


class InferenceInputTest(unittest.TestCase):
    def test_channels_and_window(self):
        source = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]])
        original, user, start = audio_window(source, 2, 'left', .5, 1.)
        np.testing.assert_array_equal(original, source[:, 1:3])
        np.testing.assert_array_equal(user, source[:1, 1:3])
        self.assertEqual(start, .5)
        _, user, _ = audio_window(source, 2, 'right', None, None)
        np.testing.assert_array_equal(user, source[1:])
        for channel, start, duration in [('bad', 0, None), ('right', -1, None),
                                         ('right', 2, None), ('right', 0, 0),
                                         ('right', float('nan'), 1)]:
            with self.assertRaises(ValueError):
                audio_window(source, 2, channel, start, duration)
        with self.assertRaises(ValueError):
            audio_window(source * float('nan'), 2, 'right', 0, None)

    def test_manifest_id_and_external_override(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'samples/a').mkdir(parents=True)
            wav = root / 'samples/a/conversation.wav'; wav.write_bytes(b'wav')
            manifest = root / 'manifest.jsonl'
            manifest.write_text(json.dumps({'sample_id': 'a', 'sample_dir': 'samples/a'})+'\n')
            settings = {'sample_id': 'a'}
            self.assertEqual(select_source(settings, manifest), wav.resolve())
            with self.assertRaises(ValueError):
                select_source({'sample_id': 'missing'}, manifest)
            self.assertEqual(select_source({'input_file': str(wav), 'sample_id': 'label'}, manifest), wav.resolve())
            manifest.write_text(json.dumps({'sample_id':'a','audio_path':'../outside.wav'})+'\n')
            with self.assertRaises(ValueError): select_source(settings, manifest)

    def test_mono_external_input(self):
        audio = np.ones((1, 8))
        original, user, _ = audio_window(audio, 4, 'right', 0, 1)
        np.testing.assert_array_equal(original, user)

    def test_external_audio_keeps_sample_conditioning(self):
        from tim_compat.inference_input import conditioning
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); sample = root / 'samples/a'; sample.mkdir(parents=True)
            (sample / 'voice_prompt_left.wav').write_bytes(b'voice')
            (sample / 'metadata.json').write_text(json.dumps({'text_prompt_left':'Persona A'}))
            manifest = root / 'manifest.jsonl'
            manifest.write_text(json.dumps({'sample_id':'a','sample_dir':'samples/a'})+'\n')
            voice, text, text_file = conditioning({'sample_id':'a','input_file':'external.wav'}, manifest)
            self.assertEqual(voice, (sample / 'voice_prompt_left.wav').resolve())
            self.assertEqual(text, 'Persona A'); self.assertIsNone(text_file)
            voice, text, _ = conditioning({'sample_id':'missing', 'voice_prompt':str(voice),
                                          'text_prompt':'Override', 'input_file':'external.wav'}, manifest)
            self.assertEqual(text, 'Override')
            with self.assertRaises(ValueError): conditioning({'input_file':'external.wav'}, manifest)

    def test_prepared_exact_window_rejects_overrun(self):
        with self.assertRaises(ValueError):
            audio_window(np.ones((2, 8)), 4, 'right', 1, 2, exact=True)
        original, _, _ = audio_window(np.ones((2, 8)), 4, 'right', 1, 2)
        self.assertEqual(original.shape[-1], 4)

    def test_real_wav_resampling_and_crop(self):
        import sphn
        from tim_compat.inference_input import load_audio_window
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'source.wav'
            source = np.stack((np.full(16000, .25, dtype=np.float32),
                               np.full(16000, -.25, dtype=np.float32)))
            sphn.write_wav(str(path), source, 16000)
            original, user, start = load_audio_window(path, {'user_channel':'left', 'start_sec':.25,
                                                            'window_seconds':.5, 'input_file':str(path)})
            self.assertEqual(original.shape, (2, 12000))
            self.assertEqual(user.shape, (1, 12000))
            np.testing.assert_array_equal(user[0], original[0])
            self.assertAlmostEqual(start, .25)

    def test_parent_routes_selected_user_and_reuses_baseline(self):
        """Exercise CLI-to-child routing and stable signatures, without running a model."""
        import builtins
        import contextlib
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch
        import sphn
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import run_free_running_inference as runner
        original_import = builtins.__import__
        def imported(name, *args, **kwargs):
            if name == 'moshi':
                return SimpleNamespace(offline=SimpleNamespace())
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'original.wav'; voice = root / 'voice.wav'
            sphn.write_wav(str(source), np.stack((np.full(24000, .2, dtype=np.float32),
                                                np.full(24000, -.2, dtype=np.float32))), 24000)
            sphn.write_wav(str(voice), np.ones((1, 2400), dtype=np.float32) * .1, 24000)
            checkpoint = root / 'checkpoint'; checkpoint.mkdir()
            (checkpoint / 'lora.safetensors').write_bytes(b'placeholder')
            (checkpoint / 'config.json').write_text(json.dumps({'lora_rank':64,'lora_scaling':2}))
            base = root / 'base.safetensors'; base.write_bytes(b'base')
            reference = root / 'reference.txt'; reference.write_text('tiếng Việt', encoding='utf-8')
            values = {'seed':0,'moshi_paths':{}, 'data':{'vietnamese_text_mode':'telex'}}
            acceptance = {'model_root':str(root), 'inference':{'original_wav':str(source),
                'voice_prompt':str(voice),'text_prompt':'Persona','user_channel':'left',
                'start_sec':.25,'window_seconds':.5, 'reference_text_file':str(reference)}}
            calls = []
            def child(command, **kwargs):
                calls.append(command)
                user, rate = sphn.read(command[command.index('--input-wav')+1])
                self.assertEqual(user.shape, (1, 12000))
                self.assertGreater(float(user.mean()), 0)
                output = Path(command[command.index('--output-dir')+1]); output.mkdir()
                sphn.write_wav(str(output / 'agent.wav'), user * .5, rate)
                (output / 'agent_text.json').write_text('["BOS", "tieengs Vieetj"]')
                return SimpleNamespace(returncode=0)
            assets = SimpleNamespace(config=checkpoint / 'config.json',moshi_weights=base)
            adapter = SimpleNamespace(keys=lambda:['layer.lora_B.weight'])
            old_path = sys.path[:]
            try:
                with patch.object(runner, 'read_config', return_value=(values, acceptance)), \
                     patch('tim_compat.local_checkpoint.LocalAssets.resolve', return_value=assets), \
                     patch('torch.cuda.is_available', return_value=True), \
                     patch('torch.cuda.device_count', return_value=1), \
                     patch('safetensors.safe_open', return_value=contextlib.nullcontext(adapter)), \
                     patch.dict(sys.modules, {'merge_lora':SimpleNamespace(merge=lambda:None)}), \
                     patch('builtins.__import__', side_effect=imported), \
                     patch.object(runner.subprocess, 'run', side_effect=child):
                    for n in (1, 2):
                        argv = ['infer','--config',str(root / 'config.yaml'),'--checkpoint',str(checkpoint),
                                '--output-dir',str(root / f'run{n}')]
                        if n == 2: argv += ['--baseline-dir',str(root / 'run1')]
                        with patch.object(sys, 'argv', argv): self.assertEqual(runner.main(), 0)
                    acceptance['inference']['user_channel'] = 'right'
                    argv = ['infer','--config',str(root / 'config.yaml'),'--checkpoint',str(checkpoint),
                            '--output-dir',str(root / 'mismatch'),'--baseline-dir',str(root / 'run1')]
                    with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'mismatch'):
                        runner.main()
                    (checkpoint / 'config.json').write_text(json.dumps({'lora_rank':64, 'lora_scaling':2,
                                                                         'vietnamese_text_mode':'no_diacritics'}))
                    with patch.object(sys, 'argv', argv), self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
                        runner.main()
                self.assertEqual(len(calls), 3)  # first base/current, second current only
                manifest = json.loads((root / 'run2/manifest.json').read_text())
                self.assertEqual(manifest['channels'], {'left':'user','right':'agent'})
                self.assertEqual(manifest['window_start_sec'], .25)
                self.assertEqual(manifest['window_duration_sec'], .5)
                self.assertEqual(manifest['cer'], 0)
                self.assertEqual(manifest['wer'], 0)
                self.assertEqual(manifest['raw_reference'], 'tiếng Việt')
                self.assertEqual(manifest['vietnamese_text_mode'], 'telex')
                self.assertEqual(manifest['hypothesis'], 'tieengs Vieetj')
                self.assertEqual(manifest['hypothesis_unicode'], 'tiếng Việt')
                self.assertEqual((root / 'run2/current/agent_unicode.txt').read_text(), 'tiếng Việt')
            finally:
                sys.path[:] = old_path
