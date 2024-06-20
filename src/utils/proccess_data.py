import pandas as pd

df = pd.read_csv('dataset.csv')['text'].values
res = [
    ' '.join(txt.split(' ')[:3])
    for txt in df
]

res_df = pd.DataFrame({
    'text':
    res
})

res_df.to_csv('startings.csv', index=False)
