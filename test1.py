import requests
import pandas as pd
import itertools

query_url = """https://api.fda.gov/drug/label.json?search=openfda.manufacturer_name:%22AstraZeneca%22&limit=37"""
pat_count = requests.get(query_url).json()
pat_count = pd.DataFrame(pat_count.get('results'))
data = pd.read_csv("datas.csv")
names = {}
for label, df in data.groupby(by='year'):
    n = df[df['year'] == label]['generic_name'].values
    names[label] = [''.join(i for i in itertools.chain(*n))]
    print(names[label])
drug_names_df = pd.DataFrame(names).transpose()
mean = data.groupby(by='year').mean()
df = mean.merge(drug_names_df, left_index=True, right_index=True)
print(df)
df.columns = ['year', 'avg_number_of_ingredients', 'drug_names']
df = df[['year','drug_names', 'avg_number_of_ingredients']]
print(df)
