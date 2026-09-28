import sys, pickle, io, contextlib
import os
S = os.path.dirname(os.path.abspath(__file__))                   # notes/p3_build; run from the repository root
sys.path.insert(0, S)
import runner, cells_a, cells_b, cells_c, cells_d, cells_e, cells_f, cells_g, cells_h
code = lambda L: [s for t, s in L if t == "code"]
names = lambda p, L: [f'{p}{i}' for i in range(len(code(L)))]
cells = [cells_a.SETUP_EDIT] + code(cells_a.CELLS_A) + code(cells_b.CELLS_B) + code(cells_c.CELLS_C) + code(cells_d.CELLS_D)
nm = ['setup'] + names('A', cells_a.CELLS_A) + names('B', cells_b.CELLS_B) + names('C', cells_c.CELLS_C) + names('D', cells_d.CELLS_D)
e1_defs = cells_e.E1.split("RES, LOGS, RUNTIME = {}, {}, {}")[0]
ns = runner.run(cells + [e1_defs], nm + ['E1defs'])
ns.update(pickle.load(open(S + "/ledger_results.pkl", "rb")))
runner.run(code(cells_e.CELLS_E)[1:], [f'E{i+2}' for i in range(len(code(cells_e.CELLS_E)) - 1)], ns=ns)
runner.run(code(cells_f.CELLS_F), [f'F{i+1}' for i in range(len(code(cells_f.CELLS_F)))], ns=ns)
runner.run(code(cells_g.CELLS_G), [f'G{i+1}' for i in range(len(code(cells_g.CELLS_G)))], ns=ns)
runner.run(code(cells_h.CELLS_H), ['H1', 'H2'], ns=ns)
