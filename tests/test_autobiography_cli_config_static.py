"""AST-only CLI configuration tests: no MCP/model/runtime imports or processes.

Execute the actual parser prefix and canonical load_config dispatch with spies;
configuration parsing/validation itself remains owned by config's existing tests.
Run directly with python -B (only standard library required).
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'src/agentic_memory'


def parser_prefix():
    tree = ast.parse((SOURCE / 'autobiography_server.py').read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    body = []
    for node in main.body:
        if isinstance(node, ast.ImportFrom) and node.module in ('mcp.server.mcpserver', 'config'):
            continue
        body.append(node)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'cfg' for t in node.targets):
            break
    assert isinstance(body[-1], ast.Assign) and body[-1].targets[0].id == 'cfg'
    return compile(ast.Module(body=body, type_ignores=[]), 'actual-personal-cli-prefix', 'exec')


class PersonalCliConfigStaticTests(unittest.TestCase):
    def launch(self, argv, env, loader):
        namespace = {'load_config': loader}
        with patch.dict(os.environ, env, clear=True), patch.object(sys, 'argv', ['personal', *argv]):
            with contextlib.redirect_stderr(io.StringIO()):
                exec(parser_prefix(), namespace)
        return namespace

    def test_missing_or_empty_selection_is_parser_error_without_loading(self):
        for env in ({}, {'AGENTIC_MEMORY_CONFIG': ''}):
            with self.subTest(env=env):
                calls = []
                with self.assertRaises(SystemExit) as result:
                    self.launch(['serve'], env, lambda path: calls.append(path))
                self.assertEqual(result.exception.code, 2)
                self.assertEqual(calls, [])

    def test_explicit_settings_expand_and_override_manifest_selection(self):
        calls = []
        namespace = self.launch(['--settings', '~/explicit.toml', '--role', 'author', '--session', 's', 'serve'],
                                {'HOME': '/synthetic', 'AGENTIC_MEMORY_CONFIG': '/ignored.json'},
                                lambda path: calls.append(path) or 'standalone')
        self.assertEqual(calls, [Path('/synthetic/explicit.toml')])
        self.assertEqual(namespace['args'].role, 'author')
        self.assertEqual(namespace['args'].session, 's')

    def test_manifest_uses_canonical_none_adapter_and_propagates_errors(self):
        calls = []
        namespace = self.launch(['--role', 'operator', 'serve'], {'AGENTIC_MEMORY_CONFIG': '/explicit.json'},
                                lambda path: calls.append(path) or 'embedded')
        self.assertEqual(calls, [None])
        self.assertEqual(namespace['cfg'], 'embedded')
        def malformed(path):
            self.assertIsNone(path)
            raise ValueError('malformed explicit manifest')
        with self.assertRaisesRegex(ValueError, 'malformed explicit manifest'):
            self.launch(['serve'], {'AGENTIC_MEMORY_CONFIG': '/malformed.json'}, malformed)

    def test_actual_config_dispatch_does_not_fallback_after_bad_manifest(self):
        tree = ast.parse((SOURCE / 'config.py').read_text())
        loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_config')
        branch = next(n for n in loader.body if isinstance(n, ast.If))
        # Actual canonical env-first dispatch, not a rewritten approximation.
        function = ast.FunctionDef(name='dispatch', args=loader.args, body=[branch, ast.Return(value=ast.Constant('standalone'))],
                                   decorator_list=[], returns=None, type_comment=None)
        function.args.args[0].annotation = None
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / 'manifest.json'
            manifest.write_text('{malformed')
            def load_manifest(path):
                return json.loads(path.read_text())
            namespace = {'os': os, 'Path': Path, 'load_manifest': load_manifest}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), 'actual-config-dispatch', 'exec'), namespace)
            with patch.dict(os.environ, {'AGENTIC_MEMORY_CONFIG': str(manifest)}, clear=True):
                with self.assertRaises(json.JSONDecodeError):
                    namespace['dispatch']()
                self.assertEqual(namespace['dispatch'](Path('/explicit.toml')), 'standalone')

    def test_invalid_role_still_rejected(self):
        with self.assertRaises(SystemExit):
            self.launch(['--role', 'administrator', 'serve'], {'AGENTIC_MEMORY_CONFIG': '/explicit.json'},
                        lambda path: self.fail('invalid role reached config loader'))


if __name__ == '__main__':
    unittest.main()
