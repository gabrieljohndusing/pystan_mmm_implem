import pandas as pd
from mmm import MMMModule

if __name__ == '__main__':
    model = MMMModule()
    df = pd.read_csv('data.csv')
    kpi = 'sales'
    model.mmm(df, kpi)