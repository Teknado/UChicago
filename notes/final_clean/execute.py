"""Execute a notebook in place, top to bottom, with the repository root as working directory.
Run from the repository root: python3 notes/final_clean/execute.py NOTEBOOK.ipynb"""
import sys, time, os, nbformat, nbclient
P = sys.argv[1]
nb = nbformat.read(P, as_version=4)
t = time.time()
nbclient.NotebookClient(nb, timeout=7200, kernel_name='python3', resources={'metadata': {'path': os.getcwd()}}).execute()
nbformat.write(nb, P)
code = [c for c in nb.cells if c.cell_type == 'code']
errs = [o for c in code for o in c.get('outputs', []) if o.get('output_type') == 'error']
print(f'{P}: executed in {time.time() - t:.0f} s; code cells {len(code)}, with execution count '
      f'{sum(c.execution_count is not None for c in code)}, errors {len(errs)}')
