from mmm import MMMModule

model = MMMModule()
start_date = '2019-08-01'
end_date = '2020-08-01'

df = model.make_dataframe(start_date, end_date)
print(df.head(5))