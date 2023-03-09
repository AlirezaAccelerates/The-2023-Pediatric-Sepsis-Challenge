import csv
import numpy as np

preds = np.random.uniform(size=2686)

preds = preds.round().astype(int)

print(preds)

with open('outputs/outputs.csv', 'w') as f:
    csv_writer = csv.DictWriter(f, fieldnames=['studyid_adm', 'prediction'])
    csv_writer.writeheader()
    for i, pred in enumerate(preds):
        rowdict = {'studyid_adm': i+1,
                'prediction': pred
                }
        csv_writer.writerow(rowdict)