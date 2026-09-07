"""Public entry points must route options without triggering other experiments."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dwasim import cli


class EntryPointTests(unittest.TestCase):
    def test_help_does_not_start_worker(self):
        with patch.object(cli, 'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.experiments_main(['--help']), 0)
        execute.assert_not_called()

    def test_no_argument_does_not_start_worker(self):
        with patch.object(cli, 'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.experiments_main([]), 0)
        execute.assert_not_called()

    def test_options_pass_through_unchanged(self):
        options = ['--dataset', 'DBLP', '--output-dir', 'path with spaces']
        with patch.object(cli, 'execute', return_value=7) as execute:
            self.assertEqual(cli.experiments_main(['power', *options]), 7)
        execute.assert_called_once_with('dwasim.experiments.power', options)

    def test_every_route_selects_one_module(self):
        for name, (module, _) in cli.EXPERIMENTS.items():
            with self.subTest(name=name), patch.object(cli, 'execute', return_value=0) as execute:
                self.assertEqual(cli.experiments_main([name]), 0)
                execute.assert_called_once_with('dwasim.experiments.' + module, [])

    def test_fixed_protocol_help_is_safe(self):
        for name in ('representation', 'mechanisms'):
            with patch.object(cli, 'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.experiments_main([name, '--help']), 0)
            execute.assert_not_called()

    def test_fixed_protocol_rejects_unknown_options(self):
        with patch.object(cli, 'execute') as execute, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.experiments_main(['mechanisms', '--dataset', 'ACM'])
        execute.assert_not_called()

    def test_invalid_experiment_does_not_start_worker(self):
        with patch.object(cli, 'execute') as execute, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.experiments_main(['not-an-experiment'])
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
