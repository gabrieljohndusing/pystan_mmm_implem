from mmm import MMMModule

model = MMMModule()
start_date = '2019-08-01'
end_date = '2020-08-01'

df = model.make_dataframe(start_date, end_date, include_econ_indicators=True)
print(df.head(5))

df2 = model.make_dataframe(start_date, end_date, include_econ_indicators=False)
print(df2.head(5))