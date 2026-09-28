"""Development runner for Problem 3: executes cell sources in one shared namespace (like a kernel),
prints outputs, saves figures, and times each cell. Usage: python runner.py <module> [<first> <last>]"""
import sys, io, time, contextlib, importlib, pickle, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIGDIR = os.path.join(HERE, 'fig')
os.makedirs(FIGDIR, exist_ok=True)
_fig_counter = [0]
_cur = ['']


def _show(*a, **k):
    for num in plt.get_fignums():
        _fig_counter[0] += 1
        plt.figure(num).savefig(os.path.join(FIGDIR, f'{_cur[0]}_{_fig_counter[0]}.png'), dpi=90, bbox_inches='tight')
    plt.close('all')


def _display(obj):
    if hasattr(obj, 'data') and hasattr(obj, 'to_html'):   # Styler
        cap = getattr(obj, 'caption', None)
        if cap:
            print('##', cap)
        print(obj.data.to_string())
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        print(obj.to_string())
    else:
        print(obj)


def run(cells, names=None, ns=None):
    ns = ns if ns is not None else {}
    ns.setdefault('display', _display)
    plt.show = _show
    for i, src in enumerate(cells):
        name = names[i] if names else f'c{i}'
        _cur[0] = name
        t = time.perf_counter()
        print(f'\n==================== {name}')
        exec(compile(src, name, 'exec'), ns)
        _show()
        print(f'[{name}: {time.perf_counter() - t:.1f}s]')
    return ns
