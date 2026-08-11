import json

for modality, f in [('ECG','ecg_benchmark_results.json'),('EEG','eeg_benchmark_results.json'),('EMG','emg_benchmark_results.json')]:
    with open(f'ai/saved_models/{f}') as file:
        data = json.load(file)
    print(f'\n=== {modality} Results ===')
    print(f'{"Model":<20} {"Accuracy":>10} {"F1":>10} {"Precision":>10} {"Recall":>10} {"Energy_mJ":>12} {"Latency_ms":>12} {"Sparsity":>10}')
    print('-' * 94)
    for model, m in data.items():
        print(f'{model:<20} {m["accuracy"]:>10.4f} {m["f1"]:>10.4f} {m["precision"]:>10.4f} {m["recall"]:>10.4f} {m["energy_mj"]:>12.6f} {m["latency_ms"]:>12.3f} {m.get("sparsity", 0):>10.4f}')