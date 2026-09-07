"""Three public entry points; experiment protocols remain separate."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

EXPERIMENTS = {
    'main': ('main', 'ACM/DBLP component, baseline and retrieval evaluation'),
    'imdb': ('multilabel', 'IMDB multi-label evaluation'),
    'ablation': ('ablation', 'Retuned component deletion and matched simple controls'),
    'paths': ('path_fusion', 'Matched single-path, uniform and learned fusion'),
    'regularization': ('regularization', 'Screening and shrinkage controls'),
    'power': ('power', 'Shared integer-power calibration'),
    'kernel-target': ('kernel_target', 'Centered kernel-target alternative'),
    'representation': ('representation', 'Complete-profile and half-profile controls'),
    'mechanisms': ('mechanisms', 'Controlled mechanism experiments'),
    'cost': ('pipeline_cost', 'Serial end-to-end cost benchmark'),
    'kernel-cost': ('kernel_cost', 'Query-batch kernel timing'),
    'single-path': ('single_path', 'Corrected single-path protocol'),
    'normalization': ('normalization', 'Two-path normalization comparison'),
    'normalization-controls': ('normalization_controls', 'Normalization controls'),
    'development': ('development', 'Earlier nested development analysis'),
}


def execute(module, arguments):
    project = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(project) + os.pathsep + environment.get('PYTHONPATH', '')
    return subprocess.call([sys.executable, '-m', module, *arguments], env=environment)


def experiments_main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=EXPERIMENTS, help='Experiment to run')
    parser.epilog = '\n'.join(f'{name}: {description}' for name, (_, description) in EXPERIMENTS.items())
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not arguments or arguments[0] in ('-h', '--help'):
        parser.print_help()
        return 0
    # Leave each protocol's dataset choices and options to its own parser.
    selected = parser.parse_args(arguments[:1]).experiment
    module = 'dwasim.experiments.' + EXPERIMENTS[selected][0]
    if selected in ('representation', 'mechanisms') and arguments[1:] in (['--help'], ['-h']):
        print(EXPERIMENTS[selected][1] + '; fixed protocol with no options.')
        return 0
    if selected in ('representation', 'mechanisms') and arguments[1:]:
        parser.error(selected + ' takes no options')
    return execute(module, arguments[1:])


def data_main():
    return execute('dwasim.datasets', sys.argv[1:])


def figures_main():
    return execute('dwasim.reporting.assets', sys.argv[1:])
