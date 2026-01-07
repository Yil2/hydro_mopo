# this is the code for train and test by linear regression (Europe) OR scaling method (globally)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import alphas
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


class ModelTrain:

    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.pred_years = self.config['pred_years']

    def model_train(self, X_train, y_train, X_test, y_max):

        model = LinearRegression(positive=True)
        model.fit(X_train, y_train)
        y_model = model.predict(X_test)*y_max.values[0]  # this is for the unknown prediction 
        y_pred = model.predict(X_train)*y_max.values[0]  # this is for the training performance


        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        corr, _ = pearsonr(y_train, y_pred)

        #print(f'Linear Regression Results: RMSE={rmse}, MAE={mae}, R2={r2}, Correlation={corr}')
        fig, ax = plt.subplots(figsize=(10, 10), nrows=2)
        ax[0].plot(y_train.index, y_pred, label='Fitted', color='blue')
        ax[0].plot(y_train.index, y_train*y_max.values[0], label='Observed', color='red')
        ax[0].set_title(f'Training Results: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}, Corr={corr[0]:.3f}')
        ax[0].legend()

        ax[1].plot(X_test.index, y_model, label='Predicted Inflow', color='green')
        ax[1].set_title('Modelled Inflow for years: ' + ', '.join(map(str, self.pred_years)))

        ax[0].set_xlabel('Date')
        ax[0].set_ylabel('Energy (MWh)')
        ax[1].set_xlabel('Date')
        ax[1].set_ylabel('Energy (MWh)')

        plt.tight_layout()
        plt.savefig(self.path_dict['pred_data_path'] / f'model_training_results_{"_".join(map(str, self.pred_years))}.png')
        plt.close()

        return pd.DataFrame(y_model, index=X_test.index, columns=y_train.columns)
    

    def max_scale(self, df):
        df_norm=df/(abs(df).max())

        return pd.DataFrame(df_norm)

    def load_process_data(self):
        
        def optimal_lags(df1, df2_orginal, max_lag):

            df2 = df2_orginal.copy()
            df2 = self.max_scale(df2)
            df1 = self.max_scale(df1)

            start_time = max(df1.index.min(), df2.index.min())
            end_time = min(df1.index.max(), df2.index.max())
            df2 = df2[start_time:end_time]
            df1 = df1[start_time:end_time]
            for col in df2.columns:
                best_corr = 0
                best_lag = 0
                for lag in range(0, max_lag + 1):
                    shifted = df2[col].shift(lag)
                    corr, _ = pearsonr(df1.iloc[:, 0], shifted.fillna(0))
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                df2_orginal[col] = df2_orginal[col].shift(best_lag)


            return df2_orginal
        

        disc= pd.read_csv(self.path_dict['disc_file'], index_col=0, parse_dates=True)
        y = pd.read_csv(self.path_dict['data_file'], index_col=0, parse_dates=True) 
        #TODO: the historical data is stored with ; separator, change to default comma later
        disc.index = pd.to_datetime(disc.index, utc=True)
        y.index = pd.to_datetime(y.index, utc=True)
        y = y.resample('W-SUN').sum().dropna(axis=0)
        disc_weekly= disc.resample('W-SUN').sum().dropna(axis=0)

        disc_lagged = optimal_lags(y, disc_weekly, max_lag=5).dropna(axis=0)
    

        return disc_lagged, y
    

    def train_test_split(self, X, y, pred_years):
        y_max = y.max()   

        if y.index.year.isin(pred_years).all():
            print(f"Prediction years {pred_years} is avialble historically, no need to model")
            # TODO: some years in the prediction year can be unavailable, need to handle this case
        else:
            X_model= X[X.index.year.isin(pred_years)]

            X_train = X[~X.index.year.isin(pred_years)]
            y_train = y[~y.index.year.isin(pred_years)]

            start_time = max(X_train.index.min(), y_train.index.min())
            end_time = min(X_train.index.max(), y_train.index.max())
            X_train = X_train[start_time:end_time]
            y_train = y_train[start_time:end_time]


            X_model = self.max_scale(X_model)
            X_train = self.max_scale(X_train)
            y_train = self.max_scale(y_train)

            #y_test = y[X.index.year < pred_years.min()]

            return X_train, y_train, X_model, y_max
        

    def modelled_data_main(self):
        X, y = self.load_process_data()
        X_train, y_train, X_model, y_max = self.train_test_split(X, y, self.pred_years)
        y_model = self.model_train(X_train, y_train, X_model, y_max)
        y_model.to_csv(self.path_dict['pred_data_path'] / f'predicted_inflow_{"_".join(map(str, self.pred_years))}.csv', sep=';')
        print("Model training and prediction completed.")