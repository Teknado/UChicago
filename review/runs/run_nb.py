"""Execute a copy of a notebook with the repository root as the working directory.
The input notebook is never modified; outputs go to review/runs/<name>__rerun.ipynb."""
import sys, time, nbformat, nbclient, os
src = sys.argv[1]
out = os.path.join('review/runs', os.path.basename(src).replace('.ipynb', '__rerun.ipynb'))
nb = nbformat.read(src, as_version=4)
t = time.time()
try:
    nbclient.NotebookClient(nb, timeout=7200, kernel_name='python3', allow_errors=True,
                            resources={'metadata': {'path': os.getcwd()}}).execute()
finally:
    nbformat.write(nb, out)
errs = [(i, o.get('ename'), o.get('evalue')) for i, c in enumerate(nb.cells) if c.cell_type == 'code'
        for o in c.get('outputs', []) if o.get('output_type') == 'error']
print(out, f'{time.time()-t:.0f}s', 'errors:', errs)
