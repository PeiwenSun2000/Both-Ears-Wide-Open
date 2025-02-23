from cal_avg import calculate_metrics

folder1 = 'results/exp1'
folder2 = 'results/exp2'

metrics = calculate_metrics(folder1, folder2)

for key, value in metrics.items():
    print(f"{key}: {value}") 