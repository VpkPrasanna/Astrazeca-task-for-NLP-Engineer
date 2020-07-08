import pandas as pd
import requests
import itertools
import time
query_url = r'https://api.fda.gov/drug/label.json?search=openfda.manufacturer_name:"AstraZeneca"'
pat_count = requests.get(query_url).json()
pat_count = pd.DataFrame(pat_count.get('results'))
# l = []
# for i in pat_count['results']:
#     effective_time = int(i['effective_time'][:-4])
#     print(effective_time)
#     if i['openfda'] == {}:
#         continue
#     manufacturer_name = i['openfda']['manufacturer_name']
#     generic_name = i['openfda']['generic_name']
#     spl_product_data_elements = len(i['spl_product_data_elements'][0].split(','))
#
#     l.append([generic_name, effective_time, spl_product_data_elements, manufacturer_name])
# print(pd.DataFrame(l, columns=['generic_name', 'year', 'num_ingredients', 'manufacturer']))


def _fetch_data():
    """
    Query the openFDA API using the url specified as argument. If >99
    items in a search, then perform multiple requests.

    Returns (pandas.DataFrame): All results

    """
    l = []
    for req in range(self.num_full_requests):
        time.sleep(5.0)
        start = req * self._max_limit
        end = start + self._max_limit
        print(f'collecting {start} to {end} of {self._number_of_search_results()}')
        r = requests.get(f'{self.url}&skip={start}&limit=99').json()
        try:
            l.append(self.extract_relevant_data(r))
        except NotImplementedError:
            raise NotImplementedError('You must override the "extract_relevant_data" method')

    if self.size_of_last_request != 0:
        start = self._number_of_search_results() - self.size_of_last_request
        last_request = requests.get(
            f'{self.url}&skip={start}&limit={self.size_of_last_request}').json()
        l.append(self.extract_relevant_data(last_request))

    return pd.concat(l)



names = {}
for label, df in data.groupby(by=['year', 'route']):
    n = df[(df['year'] == label[0]) & (df['route'] == label[1])]['generic_name'].values
    names[label] = [[i for i in itertools.chain(*n)]]
drug_names_df = pd.DataFrame(names).transpose()
drug_names_df.index.names = ['year', 'route']
mean = data.groupby(by=['year', 'route']).mean()
drug_names_df.columns = ['drug_name']

df = pd.concat([mean, drug_names_df], axis=1)
df = df[['drug_name', 'num_ingredients']]
df.rename(columns={'num_ingredients': 'avg_number_of_ingredients'}, inplace=True)
print(df)