from datasets import load_dataset, load_metric
metric = load_metric("sacrebleu")
fake_preds = ["hello there", "general kenobi"]
fake_labels = [["hello there"], ["general kenobi"]]
ans = metric.compute(predictions=fake_preds, references=fake_labels)
print(f'ans:{ans}')