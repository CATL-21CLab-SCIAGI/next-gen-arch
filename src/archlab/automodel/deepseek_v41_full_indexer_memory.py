"""Eliminate redundant score copies in the unchanged no-grad CSA2 selector.

A private function compiled from the pinned selector changes only relu/multiply
into in-place operations on the newly allocated score tensor. GEMM shapes,
rounding boundaries, reduction, masking and top-k are identical. Upstream
modules, source files, and class methods remain untouched.
"""
import ast
import inspect
import textwrap
from types import MethodType
import torch


class _ScoreStorage(ast.NodeTransformer):
    replacements = 0

    def visit_Assign(self, node):
        node = self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name) or node.targets[0].id != 'scores':
            return node
        value = node.value
        if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute) or value.func.attr != 'sum':
            return node
        product = value.func.value
        if not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
            return node
        if ast.unparse(product.left) != 'scores.relu()' or ast.unparse(product.right) != 'weights.unsqueeze(-1)':
            raise ValueError('the pinned indexer score expression changed')
        relu = ast.Call(func=ast.Attribute(value=ast.Name(id='scores', ctx=ast.Load()), attr='relu_', ctx=ast.Load()), args=[], keywords=[])
        value.func.value = ast.Call(func=ast.Attribute(value=relu, attr='mul_', ctx=ast.Load()), args=[product.right], keywords=[])
        self.replacements += 1
        return node


_PRIVATE_SELECTOR = None


def memory_efficient_selector():
    global _PRIVATE_SELECTOR
    if _PRIVATE_SELECTOR is None:
        from nemo_automodel.components.models.deepseek_v41.attention import _Indexer
        original = inspect.unwrap(_Indexer.forward)
        parsed = ast.parse(textwrap.dedent(inspect.getsource(original)))
        parsed.body[0].decorator_list = []
        transformer = _ScoreStorage()
        parsed = transformer.visit(parsed)
        if transformer.replacements != 1:
            raise ValueError('expected exactly one native no-grad indexer score expression')
        namespace = dict(original.__globals__)
        exec(compile(ast.fix_missing_locations(parsed), __file__, 'exec'), namespace)
        _PRIVATE_SELECTOR = torch.no_grad()(namespace['forward'])
    return _PRIVATE_SELECTOR


def bind_memory_efficient_selector(indexer):
    return MethodType(memory_efficient_selector(), indexer)
