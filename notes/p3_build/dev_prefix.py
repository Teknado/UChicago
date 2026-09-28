
import sys, pickle, io, contextlib
import os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runner, cells_a, cells_b, cells_c, cells_d, cells_e
code = lambda L: [s for t, s in L if t == "code"]
def prefix_ns(quiet=True):
    cells = [cells_a.SETUP_EDIT] + code(cells_a.CELLS_A) + code(cells_b.CELLS_B) + code(cells_c.CELLS_C) + code(cells_d.CELLS_D)
    e1_defs = cells_e.E1.split("RES, LOGS, RUNTIME = {}, {}, {}")[0]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns = runner.run(cells + [e1_defs])
    d = pickle.load(open(os.path.join(HERE, "ledger_results.pkl"), "rb"))
    ns.update(d)
    with contextlib.redirect_stdout(buf):
        runner.run(code(cells_e.CELLS_E)[1:], ns=ns)       # E2..E7 (tables) so later cells have their objects
    return ns
