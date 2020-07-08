import pandas as pd
import requests
import json
import time
import numpy as np

# Url to get the OPENFDA Data
query_url = """https://api.fda.gov/drug/label.json?search=openfda.manufacturer_name:%22AstraZeneca%22&limit=37"""
max_limit = 99


def _number_of_search_results(url):
    response = requests.get(url)
    return json.loads(response.content)['meta']['results']['total']


search_result = _number_of_search_results(url=query_url)


def _compute_chuncksizes(result, max_limit):
    number_of_full_requests = result // max_limit
    size_of_last_request = result % max_limit
    assert number_of_full_requests * max_limit + size_of_last_request == result
    return {
        'num_full_requests': number_of_full_requests,
        'size_of_last_request': size_of_last_request
    }


# Fetching data
num_of_request = _compute_chuncksizes(result=search_result, max_limit=max_limit)["num_full_requests"]
size_of_last = _compute_chuncksizes(result=search_result, max_limit=max_limit)["size_of_last_request"]
print(num_of_request)
print(size_of_last)
l = []


def extract_relevant_data(r):
    l = []
    for i in r['results']:
        effective_time = int(i['effective_time'][:-4])

        if i['openfda'] == {}:
            continue
        manufacturer_name = i['openfda']['manufacturer_name']

        # handle situation when the route key is not present
        try:
            route = i['openfda']['route']
            if isinstance(route, list) and len(route) == 1:
                route = route[0]
        except KeyError:
            route = np.nan
        generic_name = i['openfda']['generic_name']
        spl_product_data_elements = len(i['spl_product_data_elements'][0].split(','))

        l.append([generic_name, route, effective_time, spl_product_data_elements, manufacturer_name])
    return pd.DataFrame(l, columns=['generic_name', 'route', 'year', 'num_ingredients', 'manufacturer'])


for req in range(num_of_request):
    time.sleep(5.0)
    start = req * max_limit
    end = start + max_limit
    print(f'collecting {start} to {end} of {num_of_request()}')
    r = requests.get(f'{query_url}&skip={start}&limit=99').json()
    try:
        val = extract_relevant_data(r=r)
        l.append(val)
    except NotImplementedError:
        raise NotImplementedError('You must override the "extract_relevant_data" method')
if size_of_last != 0:
    start = num_of_request - size_of_last
    last_request = requests.get(
        f'{query_url}&skip={start}&limit={size_of_last}').json()
    print(last_request)
    value = extract_relevant_data(r=last_request)
    l.append(value)
final_df = pd.concat(l)
print(final_df)
