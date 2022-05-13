import csv
import pandas as pd

from ast import literal_eval
from typing import Optional, List


def read_data(infile):
    data = pd.read_csv(infile, sep='\t')
    data['tags'] = data['tags'].apply(literal_eval)
    return data


def write_data(outfile, data, header = Optional[List[str]]):
	with open(outfile, 'w') as f:
		writer = csv.writer(f, delimiter='\t')
		writer.writerow(header)
		for row in data:
			writer.writerow(row)

