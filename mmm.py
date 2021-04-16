import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys
import time
from datetime import datetime, timedelta
from stats_can import StatsCan
import pystan
import os
os.environ['CC'] = 'gcc-8'
os.environ['CXX'] = 'g++-8'
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
### 
from fbprophet import Prophet
import pickle
import pdb

class MMMModule:
    def __init__(self):
        self.iter = 10
        self.max_lag = 8
    
    def apply_adstock(self, x, L, P, D):
        '''
        params:
        x: original media variable, array
        L: length
        P: peak, delay in effect
        D: decay, retain rate
        returns:
        array, adstocked media variable
        '''
        x = np.append(np.zeros(L-1), x)
        
        weights = np.zeros(L)
        for l in range(L):
            weight = D**((l-P)**2)
            weights[L-1-l] = weight
        
        adstocked_x = []
        for i in range(L-1, len(x)):
            x_array = x[i-L+1:i+1]
            xi = sum(x_array * weights)/sum(weights)
            adstocked_x.append(xi)
        adstocked_x = np.array(adstocked_x)
        return adstocked_x
    
    def adstock_transform(self, df, md_cols, adstock_params):
        '''
        params:
        df: original data
        md_cols: list, media variables to be transformed
        adstock_params: dict, 
            e.g., {'sem': {'L': 8, 'P': 0, 'D': 0.1}, 'dm': {'L': 4, 'P': 1, 'D': 0.7}}
        returns: 
        adstocked df
        '''
        md_df = pd.DataFrame()
        for md_col in md_cols:
            md = md_col.split('_')[-1]
            L, P, D = adstock_params[md]['L'], adstock_params[md]['P'], adstock_params[md]['D']
            xa = self.apply_adstock(df[md_col].values, L, P, D)
            md_df[md_col] = xa
        return md_df
    
    def hill_transform(self, x, ec, slope):
        return 1 / (1 + (x / ec)**(-slope))
    
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def apply_mean_center(self, x):
        mu = np.mean(x)
        xm = x/mu
        return xm, mu
    
    def mean_center_transform(self, df, cols):
        '''
        returns: 
        mean-centered df
        scaler, dict
        '''
        df_new = pd.DataFrame()
        sc = {}
        for col in cols:
            x = df[col].values
            df_new[col], mu = self.apply_mean_center(x)
            sc[col] = mu
        return df_new, sc
    
    def mean_log1p_transform(self, df, cols):
        '''
        returns: 
        mean-centered, log1p transformed df
        scaler, dict
        '''
        df_new = pd.DataFrame()
        sc = {}
        for col in cols:
            x = df[col].values
            xm, mu = self.apply_mean_center(x)
            sc[col] = mu
            df_new[col] = np.log1p(xm)
        return df_new, sc

    def extract_ctrl_model(self, fit_result, pos_vars, pn_vars, extract_param_list=False):
        ctrl_model = {}
        ctrl_model['pos_vars'] = pos_vars
        ctrl_model['pn_vars'] = pn_vars
        ctrl_model['beta1'] = fit_result['beta1'].mean(axis=0).tolist()
        ctrl_model['beta2'] = fit_result['beta2'].mean(axis=0).tolist()
        ctrl_model['alpha'] = fit_result['alpha'].mean()
        if extract_param_list:
            ctrl_model['beta1_list'] = fit_result['beta1'].tolist()
            ctrl_model['beta2_list'] = fit_result['beta2'].tolist()
            ctrl_model['alpha_list'] = fit_result['alpha'].tolist()
        return ctrl_model
    
    def ctrl_model_predict(self, ctrl_model, df):
        pos_vars, pn_vars = ctrl_model['pos_vars'], ctrl_model['pn_vars'] 
        X1, X2 = df[pos_vars], df[pn_vars]
        beta1, beta2 = np.array(ctrl_model['beta1']), np.array(ctrl_model['beta2'])
        alpha = ctrl_model['alpha']
        y_pred = np.dot(X1, beta1) + np.dot(X2, beta2) + alpha
        return y_pred
    
    def extract_mmm(self, fit_result, max_lag, media_vars, ctrl_vars, extract_param_list=True):
        mmm = {}
        
        mmm['max_lag'] = max_lag
        mmm['media_vars'], mmm['ctrl_vars'] = media_vars, ctrl_vars
        mmm['decay'] = decay = fit_result['decay'].mean(axis=0).tolist()
        mmm['peak'] = peak = fit_result['peak'].mean(axis=0).tolist()
        mmm['beta'] = fit_result['beta'].mean(axis=0).tolist()
        mmm['tau'] = fit_result['tau'].mean()
        if extract_param_list:
            mmm['decay_list'] = fit_result['decay'].tolist()
            mmm['peak_list'] = fit_result['peak'].tolist()
            mmm['beta_list'] = fit_result['beta'].tolist()
            mmm['tau_list'] = fit_result['tau'].tolist()
        
        adstock_params = {}
        media_names = [col.replace('mdip_', '') for col in media_vars]
        for i in range(len(media_names)):
            adstock_params[media_names[i]] = {
                'L': max_lag,
                'P': peak[i],
                'D': decay[i]
            }
        mmm['adstock_params'] = adstock_params
        return mmm
    
    def mmm_decompose_contrib(self, mmm, df, original_sales):
        # adstock params
        adstock_params = mmm['adstock_params']
        # coefficients, intercept
        beta, tau = mmm['beta'], mmm['tau']
        # variables
        media_vars, ctrl_vars = mmm['media_vars'], mmm['ctrl_vars']
        num_media, num_ctrl = len(media_vars), len(ctrl_vars)
        # X_media2: adstocked, mean-centered media variables + 1
        X_media2 = self.adstock_transform(df, media_vars, adstock_params)
        X_media2, sc_mmm2 = self.mean_center_transform(X_media2, media_vars)
        X_media2 = X_media2 + 1
        # X_ctrl2, mean-centered control variables + 1
        X_ctrl2, sc_mmm2_1 = self.mean_center_transform(df[ctrl_vars], ctrl_vars)
        X_ctrl2 = X_ctrl2 + 1
        # y_true2, mean-centered sales variable + 1
        y_true2, sc_mmm2_2 = self.mean_center_transform(df, ['sales'])
        y_true2 = y_true2 + 1
        sc_mmm2.update(sc_mmm2_1)
        sc_mmm2.update(sc_mmm2_2)
        # X2 <- media variables + ctrl variable
        X2 = pd.concat([X_media2, X_ctrl2], axis=1)

        # 1. compute each media/control factor: 
        # log-log model: log(sales) = log(X[0])*beta[0] + ... + log(X[13])*beta[13] + tau
        # multiplicative model: sales = X[0]^beta[0] * ... * X[13]^beta[13] * e^tau
        # each factor = X[i]^beta[i]
        # intercept = e^tau
        factor_df = pd.DataFrame(columns=media_vars+ctrl_vars+['intercept'])
        for i in range(num_media):
            colname = media_vars[i]
            factor_df[colname] = X2[colname] ** beta[i]
        for i in range(num_ctrl):
            colname = ctrl_vars[i]
            factor_df[colname] = X2[colname] ** beta[num_media+i]
        factor_df['intercept'] = np.exp(tau)

        # 2. calculate the product of all factors -> y_pred
        # baseline = intercept * control factor = e^tau * X[13]^beta[13]
        y_pred = factor_df.apply(np.prod, axis=1)
        factor_df['y_pred'], factor_df['y_true2'] = y_pred, y_true2
        factor_df['baseline'] = factor_df[['intercept']+ctrl_vars].apply(np.prod, axis=1)

        # 3. calculate each media factor's contribution
        # media contribution = total volume – volume upon removal of the media factor
        mc_df = pd.DataFrame(columns=media_vars+['baseline'])
        for col in media_vars:
            mc_df[col] = factor_df['y_true2'] - factor_df['y_true2']/factor_df[col]
        mc_df['baseline'] = factor_df['baseline']
        mc_df['y_true2'] = factor_df['y_true2']

        # 4. scale contribution
        # predicted total media contribution: product of all media factors
        mc_df['mc_pred'] = mc_df[media_vars].apply(np.sum, axis=1)
        # true total media contribution: total volume - baseline
        mc_df['mc_true'] = mc_df['y_true2'] - mc_df['baseline']
        # predicted total media contribution is slightly different from true total media contribution
        # scale each media factor’s contribution by removing the delta volume proportionally
        mc_df['mc_delta'] = mc_df['mc_true'] - mc_df['mc_pred']
        for col in media_vars:
            mc_df[col] = mc_df[col] - mc_df['mc_delta']*mc_df[col]/mc_df['mc_pred']

        # 5. scale mc_df based on original sales
        mc_df['sales'] = original_sales
        for col in media_vars+['baseline']:
            mc_df[col] = mc_df[col]*mc_df['sales']/mc_df['y_true2']
        
        print('rmse (log-log model): ', 
            mean_squared_error(np.log(y_true2), np.log(y_pred)) ** (1/2))
        print('mape (multiplicative model): ', 
            self.mean_absolute_percentage_error(y_true2, y_pred))
        return mc_df
    
    def calc_media_contrib_pct(self, mc_df, media_vars, sales_col, period=52):
        '''
        returns:
        mc_pct: percentage over total sales
        mc_pct2: percentage over incremental sales (sales contributed by media channels)
        '''
        mc_pct = {}
        mc_pct2 = {}
        s = 0
        if period is None:
            for col in (media_vars+['baseline']):
                mc_pct[col] = (mc_df[col]/mc_df[sales_col]).mean()
        else:
            for col in (media_vars+['baseline']):
                mc_pct[col] = (mc_df[col]/mc_df[sales_col])[-period:].mean()
        for m in media_vars:
            s += mc_pct[m]
        for m in media_vars:
            mc_pct2[m] = mc_pct[m]/s
        return mc_pct, mc_pct2
    
    def create_hill_model_data(self, df, mc_df, adstock_params, media):
        y = mc_df['mdip_'+media].values
        L, P, D = adstock_params[media]['L'], adstock_params[media]['P'], adstock_params[media]['D']
        x = df['mdsp_'+media].values
        x_adstocked = self.apply_adstock(x, L, P, D)
        # centralize
        mu_x, mu_y = x_adstocked.mean(), y.mean()
        sc = {'x': mu_x, 'y': mu_y}
        x = x_adstocked/mu_x
        y = y/mu_y
            
        model_data = {
            'N': len(y),
            'y': y,
            'X': x
        }
        return model_data, sc
    
    def train_hill_model(self, df, mc_df, adstock_params, media, sm):
        '''
        params:
        df: original data
        mc_df: media contribution df derived from MMM
        adstock_params: adstock parameter dict output by MMM
        media: 'dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem'
        sm: stan model object    
        returns:
        a dict of model data, scaler, parameters
        '''
        data, sc = self.create_hill_model_data(df, mc_df, adstock_params, media)
        fit = sm.sampling(data=data, iter=self.iter, chains=4)
        fit_result = fit.extract()
        hill_model = {
            'beta_hill_list': fit_result['beta_hill'].tolist(),
            'ec_list': fit_result['ec'].tolist(),
            'slope_list': fit_result['slope'].tolist(),
            'sc': sc,
            'data': {
                'X': data['X'].tolist(),
                'y': data['y'].tolist(),
            }
        }
        return hill_model
    
    def extract_hill_model_params(self, hill_model, method='mean'):
        if method=='mean':
            hill_model_params = {
                'beta_hill': np.mean(hill_model['beta_hill_list']), 
                'ec': np.mean(hill_model['ec_list']), 
                'slope': np.mean(hill_model['slope_list'])
            }
        elif method=='median':
            hill_model_params = {
                'beta_hill': np.median(hill_model['beta_hill_list']), 
                'ec': np.median(hill_model['ec_list']), 
                'slope': np.median(hill_model['slope_list'])
            }
        return hill_model_params
    
    def hill_model_predict(self, hill_model_params, x):
        beta_hill, ec, slope = hill_model_params['beta_hill'], hill_model_params['ec'], hill_model_params['slope']
        y_pred = beta_hill * self.hill_transform(x, ec, slope)
        return y_pred

    def evaluate_hill_model(self, hill_model, hill_model_params):
        x = hill_model['data']['X']
        y_true = [w * hill_model['sc']['y'] for w in hill_model['data']['y']]
        y_pred = self.hill_model_predict(hill_model_params, x) * hill_model['sc']['y']
        print('mape on original data: ', 
            self.mean_absolute_percentage_error(y_true, y_pred))
        return y_true, y_pred
    
    def make_dataframe(self, start_date, end_date, user_data_filepath, user_data_date_column, include_econ_indicators=True):
        '''
        When function is called, returns a Pandas dataframe with time interval every sunday 
        including the first one before the `start_date` and the last one before `end_date`.
        start_date, end_date: input in the yyyy-mm-dd format
        include_econ_indicators: True by default. Pulls unemployment, monthly CPI, and monthly GDP data from Statistics Canada using stats_can API
        user_data_filepath: string with path to the user provided adspend and kpi data
        user_data_date_column: name of column with date. The first and last dates must match `start_date` and `end_date` respectively
        '''
        user_df = pd.read_csv(user_data_filepath)
        user_df[user_data_date_column] = pd.to_datetime(user_df[user_data_date_column])
        user_df['ref_yr_mth'] = [item for item in zip(user_df[user_data_date_column].dt.year, user_df[user_data_date_column].dt.month)]

        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)

        df = pd.DataFrame()
        df['date'] = pd.date_range(start_date_ts, end_date_ts, freq = 'W-SUN')
        df['ref_date'] = [item for item in zip(df['date'].dt.year, df['date'].dt.month)]

        if include_econ_indicators:
            sc = StatsCan()
            unem_df = sc.vectors_to_df_remote('v2062815', periods = 360)
            unem_df.columns = ['unemployment_rate']
            unem_df = unem_df.reset_index()

            gdp_df = sc.vectors_to_df_remote('v65201210', periods = 360)
            gdp_df.columns = ['monthly_gdp']
            gdp_df = gdp_df.reset_index()

            cpi_df = sc.vectors_to_df_remote('v41690973', periods = 360)
            cpi_df.columns = ['monthly_cpi']
            cpi_df = cpi_df.reset_index()

            econ_df = pd.merge(unem_df, gdp_df, on='refPer', how='inner')
            econ_df = econ_df.merge(cpi_df, on='refPer')


            econ_df['refPer_yr_mth'] = [item for item in zip(econ_df['refPer'].dt.year, econ_df['refPer'].dt.month)]

            df_merge_orig_gdp = pd.merge(user_df, 
                                    econ_df, 
                                    how = 'left',
                                    left_on='ref_yr_mth', 
                                    right_on='refPer_yr_mth')

            return df_merge_orig_gdp.drop(['refPer_yr_mth','ref_yr_mth','refPer',], axis = 1)
        else:
            return user_df

    def get_forecast(self, user_data_filepath, user_date_column, kpi_column, future_periods):
        """
        user_data_filepath (str): Path to csv file with data
        user_date_column (str): Name of date column
        kpi_column (str): Name of kpi column that you want to forecast
        future_periods (int): How many weeks into the future you want to make forecasts
        """
        data = pd.read_csv(user_data_filepath)
        df = data.loc[:,[user_date_column, kpi_column]]
        df.columns = ['ds','y']

        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods = future_periods, freq = 'W')
        forecast = m.predict(future)

        f = plt.figure(figsize=(18,12))
        plt.plot(pd.to_datetime(forecast.loc[:df.shape[0]-1,'ds']), df['y'], color = 'blue', label=f'True {kpi_column} Value')
        plt.plot(pd.to_datetime(forecast.loc[:,'ds']), forecast['yhat'], color = 'green', label=f'Forecasted {kpi_column} Value')
        f.savefig('forecasts.png')

        return f

   
    def calc_roas(self, mc_df, ms_df, period=None):
        roas = {}
        md_names = [col.split('_')[-1] for col in ms_df.columns]
        for i in range(len(md_names)):
            md = md_names[i]
            sp, mc = ms_df['mdsp_'+md], mc_df['mdip_'+md]
            if period is None:
                md_roas = mc.sum()/sp.sum()
            else:
                md_roas = mc[-period:].sum()/sp[-period:].sum()
            roas[md] = md_roas
        return roas

    # calc weekly ROAS
    def calc_weekly_roas(self, mc_df, ms_df):
        weekly_roas = pd.DataFrame()
        md_names = [col.split('_')[-1] for col in ms_df.columns]
        for md in md_names:
            weekly_roas[md] = mc_df['mdip_'+md]/ms_df['mdsp_'+md]
        weekly_roas.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        return weekly_roas
    
    def calc_mroas(self, hill_model, hill_model_params, period=52):
        '''
        calculate mROAS for a media
        params:
        hill_model: a dict containing model data and scaling factor
        hill_model_params: a dict containing beta_hill, ec, slope
        period: in weeks, the period used to calculate ROAS and mROAS. 52 is last one year.
        return:
        mROAS value
        '''
        mu_x, mu_y = hill_model['sc']['x'], hill_model['sc']['y']
        # get current media spending level over the period specified
        cur_sp = np.array(hill_model['data']['X'])
        if period is not None:
            cur_sp = cur_sp[-period:]
        cur_mc = sum(self.hill_model_predict(hill_model_params, cur_sp) * mu_y)
        # next spending level: increase by 1%
        next_sp = cur_sp * 1.01
        # media contribution under next spending level
        next_mc = sum(self.hill_model_predict(hill_model_params, next_sp) * mu_y)
        
        # mROAS
        delta_mc = next_mc - cur_mc
        delta_sp = sum(next_sp * mu_x) - sum(cur_sp * mu_x)
        mroas = delta_mc/delta_sp
        return mroas

    def mmm(self, df, kpi):
        mdip_cols = [col for col in df.columns if 'mdip_' in col]
        mdsp_cols = [col for col in df.columns if 'mdsp_' in col]
        me_cols = [col for col in df.columns if 'me_' in col]
        mrkdn_cols = [col for col in df.columns if 'mrkdn_' in col]
        hldy_cols = [col for col in df.columns if 'hldy_' in col]
        seas_cols = [col for col in df.columns if 'seas_' in col]
        base_vars = me_cols + mrkdn_cols + hldy_cols + seas_cols

        df_ctrl, sc_ctrl = self.mean_center_transform(df, [kpi] + me_cols + mrkdn_cols)
        df_ctrl = pd.concat([df_ctrl, df[hldy_cols + seas_cols]], axis=1)

        pos_vars = [col for col in base_vars if col not in seas_cols]
        X1 = df_ctrl[pos_vars].values
        pn_vars = seas_cols
        X2 = df_ctrl[pn_vars].values

        ctrl_data = {
            'N': len(df_ctrl),
            'K1': len(pos_vars), 
            'K2': len(pn_vars), 
            'X1': X1,
            'X2': X2, 
            'y': df_ctrl[kpi].values,
            'max_intercept': min(df_ctrl[kpi])
        }

        ctrl_code1 = '''
        data {
          int N; // number of observations
          int K1; // number of positive predictors
          int K2; // number of positive/negative predictors
          real max_intercept; // restrict the intercept to be less than the minimum y
          matrix[N, K1] X1;
          matrix[N, K2] X2;
          vector[N] y; 
        }

        parameters {
          vector<lower=0>[K1] beta1; // regression coefficients for X1 (positive)
          vector[K2] beta2; // regression coefficients for X2
          real<lower=0, upper=max_intercept> alpha; // intercept
          real<lower=0> noise_var; // residual variance
        }

        model {
          // Define the priors
          beta1 ~ normal(0, 1); 
          beta2 ~ normal(0, 1); 
          noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
          // The likelihood
          y ~ normal(X1*beta1 + X2*beta2 + alpha, sqrt(noise_var));
        }
        '''
        
        sm1 = pystan.StanModel(model_code=ctrl_code1, verbose=True)
        fit1 = sm1.sampling(data=ctrl_data, iter=self.iter, chains=4)
        fit1_result = fit1.extract()

        base_sales_model = self.extract_ctrl_model(fit1_result, pos_vars=pos_vars, pn_vars=pn_vars)
        base_sales = self.ctrl_model_predict(base_sales_model, df_ctrl)
        df['base_' + kpi] = base_sales*sc_ctrl[kpi]

        df_mmm, sc_mmm = self.mean_log1p_transform(df, [kpi, 'base_' + kpi])
        mu_mdip = df[mdip_cols].apply(np.mean, axis=0).values
        num_media = len(mdip_cols)
        X_media = np.concatenate((np.zeros((self.max_lag - 1, num_media)), df[mdip_cols].values), axis=0)
        X_ctrl = df_mmm['base_' + kpi].values.reshape(len(df), 1)
        model_data2 = {
            'N': len(df),
            'max_lag': self.max_lag, 
            'num_media': num_media,
            'X_media': X_media, 
            'mu_mdip': mu_mdip,
            'num_ctrl': X_ctrl.shape[1],
            'X_ctrl': X_ctrl, 
            'y': df_mmm[kpi].values
        }

        model_code2 = '''
        functions {
          // the adstock transformation with a vector of weights
          real Adstock(vector t, row_vector weights) {
            return dot_product(t, weights) / sum(weights);
          }
        }
        data {
          // the total number of observations
          int<lower=1> N;
          // the vector of sales
          real y[N];
          // the maximum duration of lag effect, in weeks
          int<lower=1> max_lag;
          // the number of media channels
          int<lower=1> num_media;
          // matrix of media variables
          matrix[N+max_lag-1, num_media] X_media;
          // vector of media variables' mean
          real mu_mdip[num_media];
          // the number of other control variables
          int<lower=1> num_ctrl;
          // a matrix of control variables
          matrix[N, num_ctrl] X_ctrl;
        }
        parameters {
          // residual variance
          real<lower=0> noise_var;
          // the intercept
          real tau;
          // the coefficients for media variables and base sales
          vector<lower=0>[num_media+num_ctrl] beta;
          // the decay and peak parameter for the adstock transformation of
          // each media
          vector<lower=0,upper=1>[num_media] decay;
          vector<lower=0,upper=ceil(max_lag/2)>[num_media] peak;
        }
        transformed parameters {
          // the cumulative media effect after adstock
          real cum_effect;
          // matrix of media variables after adstock
          matrix[N, num_media] X_media_adstocked;
          // matrix of all predictors
          matrix[N, num_media+num_ctrl] X;
          
          // adstock, mean-center, log1p transformation
          row_vector[max_lag] lag_weights;
          for (nn in 1:N) {
            for (media in 1 : num_media) {
              for (lag in 1 : max_lag) {
                lag_weights[max_lag-lag+1] <- pow(decay[media], (lag - 1 - peak[media]) ^ 2);
              }
             cum_effect <- Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
             X_media_adstocked[nn, media] <- log1p(cum_effect/mu_mdip[media]);
            }
          X <- append_col(X_media_adstocked, X_ctrl);
          } 
        }
        model {
          decay ~ beta(3,3);
          peak ~ uniform(0, ceil(max_lag/2));
          tau ~ normal(0, 5);
          for (i in 1 : num_media+num_ctrl) {
            beta[i] ~ normal(0, 1);
          }
          noise_var ~ inv_gamma(0.05, 0.05 * 0.01);
          y ~ normal(tau + X * beta, sqrt(noise_var));
        }
        '''

        sm2 = pystan.StanModel(model_code=model_code2, verbose=True)
        fit2 = sm2.sampling(data=model_data2, iter=self.iter, chains=3)
        fit2_result = fit2.extract()

        mmm = self.extract_mmm(fit2, max_lag=self.max_lag, media_vars=mdip_cols, ctrl_vars=['base_' + kpi])

        beta_media = {}
        for i in range(len(mmm['media_vars'])):
            md = mmm['media_vars'][i]
            betas = []
            for j in range(len(mmm['beta_list'])):
                betas.append(mmm['beta_list'][j][i])
            beta_media[md] = np.array(betas)

        f = plt.figure(figsize=(18,15))
        for i in range(len(mmm['media_vars'])):
            ax = f.add_subplot(5,3,i+1)
            md = mmm['media_vars'][i]
            x = beta_media[md]
            mean_x = x.mean()
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-')
            ax.axvline(median_x, color='g', linestyle='-')
            ax.set_title(md)
        f.savefig('media_coef.png', dpi=f.dpi)


        mc_df = self.mmm_decompose_contrib(mmm, df, df[kpi])
        adstock_params = mmm['adstock_params']
        mc_pct, mc_pct2 = self.calc_media_contrib_pct(mc_df, mdip_cols, kpi, period=52)
    
        model_code3 = '''
        functions {
          // the Hill function
          real Hill(real t, real ec, real slope) {
          return 1 / (1 + (t / ec)^(-slope));
          }
        }

        data {
          // the total number of observations
          int<lower=1> N;
          // y: vector of media contribution
          vector[N] y;
          // X: vector of adstocked media spending
          vector[N] X;
        }

        parameters {
          // residual variance
          real<lower=0> noise_var;
          // regression coefficient
          real<lower=0> beta_hill;
          // ec50 and slope for Hill function of the media
          real<lower=0,upper=1> ec;
          real<lower=0> slope;
        }

        transformed parameters {
          // a vector of the mean response
          vector[N] mu;
          for (i in 1:N) {
            mu[i] <- beta_hill * Hill(X[i], ec, slope);
          }
        }

        model {
          slope ~ gamma(3, 1);
          ec ~ beta(2, 2);
          beta_hill ~ normal(0, 1);
          noise_var ~ inv_gamma(0.05, 0.05 * 0.01); 
          y ~ normal(mu, sqrt(noise_var));
        }
        '''

        sm3 = pystan.StanModel(model_code=model_code3, verbose=True)
        hill_models = {}
        to_train = ['dm', 'inst', 'nsp', 'auddig', 'audtr', 'vidtr', 'viddig', 'so', 'on', 'sem']
        for media in to_train:
            print('training for media: ', media)
            hill_model = self.train_hill_model(df, mc_df, adstock_params, media, sm3)
            hill_models[media] = hill_model

        hill_model_params_mean, hill_model_params_med, mroas_1y = {}, {}, {}
        ms_df = pd.DataFrame()
        for md in list(hill_models.keys()):
            params1 = self.extract_hill_model_params(hill_models[md], method='mean')
            params1['sc'] = hill_models[md]['sc']
            hill_model_params_mean[md] = params1
            print('evaluating media: ', md)
            _ = self.evaluate_hill_model(hill_models[md], params1)
            x = np.array(hill_models[md]['data']['X']) * hill_models[md]['sc']['x']
            ms_df['mdsp_' + md] = x
            mroas_1y[md] = self.calc_mroas(hill_models[md], params1, period=52)

        f = plt.figure(figsize=(18,16))
        hm_keys = list(hill_models.keys())
        for i in range(len(hm_keys)):
            ax = f.add_subplot(4,3,i+1)
            md = hm_keys[i]
            hm = hill_models[md]
            hmp = hill_model_params_mean[md]
            x, y = hm['data']['X'], hm['data']['y']
            mu_x, mu_y = hm['sc']['x'], hm['sc']['y']
            ec, slope = hmp['ec'], hmp['slope']
            x_sorted = np.array(sorted(x))
            y_fit = self.hill_model_predict(hmp, x_sorted)
            ax = sns.scatterplot(x=[t*mu_x for t in x], y=[t*mu_y for t in y], alpha=0.2)
            ax = sns.lineplot(x=[t* mu_x for t in x_sorted], y=[t*mu_y for t in y_fit], color='r', 
                         label='ec=%.2f, slope=%.2f'%(ec, slope))
            ax.set_title(md)
        f.savefig('hill.png', dpi=f.dpi)

        roas_1y = self.calc_roas(mc_df, ms_df, period=52)
        weekly_roas = self.calc_weekly_roas(mc_df, ms_df)
        roas1y_df = pd.DataFrame(index=weekly_roas.columns.tolist())
        roas1y_df['roas_mean'] = weekly_roas[-52:].apply(np.mean, axis=0)
        roas1y_df['roas_median'] = weekly_roas[-52:].apply(np.median, axis=0)

        f = plt.figure(figsize=(18,12))
        for i in range(len(weekly_roas.columns)):
            md = weekly_roas.columns[i]
            ax = f.add_subplot(4,3,i+1)
            x = weekly_roas[md][-52:]
            mean_x = np.mean(x)
            median_x = np.median(x)
            ax = sns.distplot(x)
            ax.axvline(mean_x, color='r', linestyle='-', alpha=0.5)
            ax.axvline(median_x, color='g', linestyle='-', alpha=0.5)
            ax.set(xlabel=None)
            ax.set_title(md)
        f.savefig('weekly_roas.png', dpi=f.dpi)

        roas1y_df = pd.concat([
            roas1y_df[['roas_mean', 'roas_median']],
            pd.DataFrame.from_dict(mroas_1y, orient='index', columns=['mroas']),
            pd.DataFrame.from_dict(roas_1y, orient='index', columns=['roas_avg'])
        ], axis=1)
        roas1y_df.to_csv('roas1y_df1.csv')

        f = plt.figure(figsize=(18,12))
        plt.bar(roas1y_df.index, roas1y_df.loc[:,'roas_mean'], alpha=0.2, label='ROAS')
        plt.plot(roas1y_df.index, roas1y_df.loc[:,'mroas'], alpha=0.25, label='mROAS')
        plt.legend()
        f.savefig('roas_and_mroas.png', dpi=f.dpi)

        return df

