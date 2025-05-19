# Intrusion-Detection System Replication

Reproduce the results of:
- Kasongo & Sun (2020) “Deep‐RNN framework for network intrusion detection”
- Kilincer et al. (2021) “Machine-learning-based intrusion-detection benchmark”

on the NSL-KDD and UNSW-NB15 datasets, then export a tidy text report with one command.

### Setup:
```
pip install -r requirements.txt
```

### Command examples:

ML baselines (Decision-Tree, k-NN, SVM) example:
```
python intrusion_replication.py --dataset nsl --task multiclass --model dt --report results/nsl_dt.txt
```

DL baselines (LSTM, Simple-RNN) example:
```
python intrusion_replication.py --dataset unsw --task multiclass --model lstm --report results/unsw_lstm.txt
```
