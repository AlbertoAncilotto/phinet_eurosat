from sklearn.metrics import precision_score, recall_score, f1_score

file_paths = ['checkpoints/a050_unbalanced.log']
for file_path in file_paths:
    with open(file_path, 'r') as log:
        test_labels = list(map(int, log.readline().split(', ')))
        label_correct = list(map(eval, log.readline().split(', ')))

    predict_labels = [test_labels[i] if correct else -1 for i, correct in enumerate(label_correct)]

    precision = precision_score(test_labels, predict_labels, average='macro')
    recall = recall_score(test_labels, predict_labels, average='macro')
    f1 = f1_score(test_labels, predict_labels, average='macro')
    print(file_path)
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
