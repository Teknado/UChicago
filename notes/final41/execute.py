"""Execute Final_Autumn_2026_final.ipynb in place, top to bottom, with the repository root as working directory.
Run from the repository root: python3 notes/final41/execute.py"""
import time, nbformat, nbclient, os
P = 'Final_Autumn_2026_final.ipynb'
nb = nbformat.read(P, as_version=4)
t = time.time()
nbclient.NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': os.getcwd()}}).execute()
nbformat.write(nb, P)
code = [c for c in nb.cells if c.cell_type == 'code']
errs = [o for c in code for o in c.get('outputs', []) if o.get('output_type') == 'error']
print(f'executed in {time.time() - t:.0f} s; code cells {len(code)}, with execution count {sum(c.execution_count is not None for c in code)}, errors {len(errs)}')
