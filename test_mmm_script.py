from mmm import MMMModule

model = MMMModule()
start_date = '2019-08-01'
end_date = '2020-08-01'

df = model.make_dataframe(start_date, end_date, 'data_cleaned.csv', 'wk_strt_dt', include_econ_indicators=True)
print(df.head(5))

model.get_forecast('data_cleaned.csv', 'wk_strt_dt','sales',12)