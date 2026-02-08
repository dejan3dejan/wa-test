import pandas as pd
import glob
import os

results = []
files = glob.glob('results/eval_results_*.csv')

for f in sorted(files):
    try:
        df = pd.read_csv(f)
        name = os.path.basename(f).replace('eval_results_test_queries_', '').replace('.csv', '')
        
        metrics = {
            'Namespace': name,
            'Queries': len(df),
            'Hit@1': f"{df['hit@1'].mean():.2%}",
            'Hit@5': f"{df['hit@5'].mean():.2%}",
            'MRR': f"{df['mrr'].mean():.4f}"
        }
        results.append(metrics)
    except Exception as e:
        print(f"Error processing {f}: {e}")

summary_df = pd.DataFrame(results)
print(summary_df.to_string(index=False))
