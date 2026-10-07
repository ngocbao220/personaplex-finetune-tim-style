"""CPU reproduction of native-vs-reload initialization RNG drift."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]


def initialization(path, checkpoint_name):
    tree = ast.parse(path.read_text())
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            function = node.value.func
            if (isinstance(function, ast.Attribute) and function.attr == 'get_mimi'
                    and isinstance(function.value, ast.Name) and function.value.id == checkpoint_name):
                nodes.append(node)
            if (isinstance(function, ast.Name) and function.id == 'get_fsdp_model'
                    and len(node.value.args) > 1 and isinstance(node.value.args[1], ast.Name)
                    and node.value.args[1].id == checkpoint_name):
                nodes.append(node)
    nodes.sort(key=lambda node: node.lineno)
    # Select the main training initialization, not the generation teardown helper.
    mimi = [n for n in nodes if isinstance(n.value.func, ast.Attribute)]
    model = [n for n in nodes if isinstance(n.value.func, ast.Name)]
    chosen_model = model[-1] if checkpoint_name == 'checkpoint_info' else model[0]
    preceding_mimi = [n for n in mimi if n.lineno < chosen_model.lineno]
    selected = ([preceding_mimi[-1]] if preceding_mimi else []) + [chosen_model]
    return compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec')


class ReloadInitializationTest(unittest.TestCase):
    def test_reload_matches_native_rng_before_model_construction(self):
        saved = {}
        class Info:
            def get_mimi(self, **kwargs):
                # A real codec constructor consumes RNG before loading its weights.
                return torch.nn.Linear(3, 3)
        info = Info()
        def model(args, checkpoint, resume_lora_path=None):
            result = torch.nn.Module()
            result.frozen_lora_A = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
            result.trained_lora = torch.nn.Parameter(torch.randn(3, 3))
            if resume_lora_path:
                result.trained_lora.data.copy_(saved['trained_lora'])
            return result
        args = SimpleNamespace(moshi_paths=SimpleNamespace(), seed=0)
        namespace = dict(args=args, opts=SimpleNamespace(reload=Path('/checkpoint')),
                         get_fsdp_model=model, checkpoint_info=info, info=info,
                         resume_lora_path=None)
        torch.manual_seed(0)
        exec(initialization(ROOT / 'moshi-finetune/train.py', 'checkpoint_info'), namespace)
        native = namespace['model']
        saved['trained_lora'] = native.trained_lora.detach().clone()
        torch.manual_seed(0)
        exec(initialization(ROOT / 'scripts/_tim_worker.py', 'info'), namespace)
        reloaded = namespace['model']
        self.assertTrue(torch.equal(native.trained_lora, reloaded.trained_lora))
        self.assertTrue(torch.equal(native.frozen_lora_A, reloaded.frozen_lora_A),
                        'reload skips native codec RNG consumption before frozen LoRA initialization')
