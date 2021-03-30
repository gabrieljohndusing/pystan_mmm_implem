from mmm import MMMModule

model = MMMModule()
start_date = '2019-08-01'
end_date = '2020-08-01'

dataset_info = {
    'filepath': 'data_cleaned.csv',
    'date_column': 'wk_strt_dt',
    'ad_spend_list': ['mdsp_dm','mdsp_inst','mdsp_nsp','mdsp_auddig','mdsp_audtr','mdsp_vidtr','mdsp_viddig','mdsp_so','mdsp_on','mdsp_sem']
}

df = model.make_dataframe(start_date, end_date, include_econ_indicators=True, dataset_info=dataset_info)
print(df.head(5))

df2 = model.make_dataframe(start_date, end_date, include_econ_indicators=False)
print(df2.head(5))