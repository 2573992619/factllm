import pandas as pd

def accuracy(data,model_name):
    df = pd.DataFrame(data, columns=['subject', 'Predicted','Answer'])
    print(df)
    df['is_correct'] = df['Predicted'] == df['Answer']
    domain_acc = df.groupby('subject')['is_correct'].mean().reset_index()
    domain_acc.columns = ['subject', 'accuracy']
    domain_acc.to_csv(f"output/{model_name}_domain_accuracy.csv", index=False)
    domain_acc.set_index("subject").to_json(f"output/{model_name}_domain_accuracy.json")

    return domain_acc

