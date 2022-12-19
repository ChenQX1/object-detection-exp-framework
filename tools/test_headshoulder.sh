#bin/bash
python eval/run_headshoulder.py cfgs/headshoulder_solver.py
python eval/eval_detect_scores.py cfgs/headshoulder_solver.py
