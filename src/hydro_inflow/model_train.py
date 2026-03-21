# this is the code for train and test model without tuning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


class ModelTrain:

    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.pred_years = self.config['pred_years']
        self.country_code = config_obj.country_code
        self.hydro_type = config_obj.hydro_type
        self.training_model = self.config['algorithm']
        #TODO: move pred_years to config_data_handel
        if self.config['pred_years'] == 'None':
            self.pred_years = list(range(2015, 2025))  # default prediction years
        elif self.config['pred_years'] == 'all':
            self.pred_years = list(range(1980, 2025))
        else:
            self.pred_years = self.config['pred_years']  

    def to_1d(self, y):
    # DataFrame -> take first column
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"Expected 1 target column, got {y.shape[1]}")
            return y.iloc[:, 0].to_numpy()

        if isinstance(y, pd.Series):
            return y.to_numpy()

        return np.asarray(y).reshape(-1)


    def calc_pearson(self,a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return pearsonr(a, b)[0]
    
    def tune_hybrid_model_weight(self, X, y, years, model_a, model_b, w_grid=None):

        """
        Tune weight w for: y = w*A + (1-w)*B using LOYO.
        Returns best_w, best_rmse
        """
        if w_grid is None:
            w_grid = np.linspace(0, 1, 11) 

        logo = LeaveOneGroupOut()
        best = {"w": 0.5, "rmse": np.inf}

        for w in w_grid:
            preds, trues = [], []

            for tr_idx, te_idx in logo.split(X, y, groups=years):
                A = clone(model_a)
                B = clone(model_b)

                A.fit(X.iloc[tr_idx], y[tr_idx])
                B.fit(X.iloc[tr_idx], y[tr_idx])

                p = w * A.predict(X.iloc[te_idx]) + (1 - w) * B.predict(X.iloc[te_idx])

                preds.append(p)
                trues.append(y[te_idx])

            yhat = np.concatenate(preds)
            ytrue = np.concatenate(trues)
            mse = mean_squared_error(ytrue, yhat)
            rmse = float(np.sqrt(mse))
            #rmse = float(np.sqrt(np.mean((ytrue - yhat) ** 2)))

            if rmse < best["rmse"]:
                best = {"w": float(w), "rmse": rmse}

        return best["w"]

    def model_train(self, X_train, y_train, X_model, cap, cross_val=True, save_cv_fig=True):
        """
        Trains final model on full training data; 
        optionally performs leave-one-year-out CV for evaluation.

        Returns:
        y_model: predictions for X_model (or None if X_model empty)
        """

        # ---------- Build estimator ----------
        training_model = self.training_model.lower() # convert to lowercase

        param_distributions = {
            "n_estimators": [800],      
            "max_depth": [5,9,12,15], 
            "min_samples_leaf": [1,2,4],
            "max_features": ["sqrt", 0.33],
            "bootstrap": [True],
            'oob_score': [False],
            'max_samples': [0.8],
        }

        forest_reg_cv= GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=param_distributions,           
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,               # parallelize across CV × configs
            verbose=1,
            refit=True               # refit best on full data
        )

        grid_search = False

        if training_model == "random forest":
            if grid_search:
                base_model = forest_reg_cv
            else:
                base_model = RandomForestRegressor(
                    n_estimators=800,
                    max_features="sqrt",
                    max_depth=9,
                    min_samples_leaf=1,
                    max_samples=0.8,
                    oob_score=True,
                    n_jobs=-1,
                    random_state=42,   
                )

        elif training_model == "linear regression":
            base_model = LinearRegression(positive=True, fit_intercept=False)

        elif training_model == "hybrid":
        
            p_num = X_train.shape[0]
            alphas = [0.1, 1.0, 5.0, 10.0, 15.0, max(2, int(p_num / 2)), max(2, int(p_num))]

            if grid_search:
                rf_base = forest_reg_cv
            else:
                rf_base = RandomForestRegressor(
                    n_estimators=800,
                    max_features="sqrt",
                    max_depth=9,
                    min_samples_leaf=1,
                    max_samples=0.8,
                    oob_score=True,
                    n_jobs=-1,
                    random_state=42,
                )

            ridge_base = RidgeCV(alphas=alphas, cv=5, scoring="neg_root_mean_squared_error")
            #TODO: tune weights
            base_model = VotingRegressor(
                estimators=[("rf", rf_base), ("ridge", ridge_base)],
                weights=[0.5, 0.5],
            )

        else:
            raise ValueError(f"Unknown training_model: {self.training_model}")

        # ---------- Optional LOYO (outer) CV evaluation ----------
        #cv_metrics = None
        #TODO: update to optionally perform LOYO
        if cross_val == True:
            # groups = year 
            years = y_train.index.year
            logo = LeaveOneGroupOut()

            y_true_all = []
            y_pred_all = []
            idx_all = []

            # split leaves out one year
            for train_idx, test_idx in logo.split(X_train, self.to_1d(y_train), groups=years):
                X_tr = X_train.iloc[train_idx]
                X_te = X_train.iloc[test_idx]

                if X_te.empty:
                    continue  # 

                y_tr = self.to_1d(y_train.iloc[train_idx] if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.asarray(y_train)[train_idx])
                y_te = self.to_1d(y_train.iloc[test_idx]  if isinstance(y_train, (pd.Series, pd.DataFrame)) else np.asarray(y_train)[test_idx])
                #TODO: update to tune weights optionally
                tuning_weights = True
                if self.training_model.lower() == "hybrid":
                    if tuning_weights == True:
                        years_tr = X_tr.index.year.to_numpy()

                        w_rf = self.tune_hybrid_model_weight(X_tr, y_tr, years_tr, rf_base, ridge_base, [0.1,  0.3,0.5, 0.7,  0.9])

                        rf = clone(rf_base).fit(X_tr, y_tr)
                        rg = clone(ridge_base).fit(X_tr, y_tr)

                        pred = w_rf * rf.predict(X_te) + (1 - w_rf) * rg.predict(X_te)
                    else:
                        m = clone(base_model).fit(X_tr, y_tr)
                        pred = m.predict(X_te)
                    
                else:
                    m = clone(base_model).fit(X_tr, y_tr)
                    pred = m.predict(X_te)

                # enforce positivity
                pred = np.maximum(pred, 0.0)
                if cap is not None:
                    pred = np.minimum(pred, cap*24)

                y_true_all.append(y_te)
                y_pred_all.append(pred)
                idx_all.append(X_te.index)


            if idx_all:
                idx_all = np.concatenate([idx.to_numpy() for idx in idx_all])
                y_true_all = np.concatenate(y_true_all)
                y_pred_all = np.concatenate(y_pred_all)

                # Sort by time 
                order = np.argsort(idx_all)
                idx_sorted = pd.to_datetime(idx_all[order])
                y_true_sorted = y_true_all[order]
                y_pred_sorted = y_pred_all[order]

                mse = mean_squared_error(y_true_sorted, y_pred_sorted)
                rmse = float(np.sqrt(mse))
                mae = mean_absolute_error(y_true_sorted, y_pred_sorted)
                r2 = r2_score(y_true_sorted, y_pred_sorted)
                corr = self.calc_pearson(y_true_sorted, y_pred_sorted)

                if save_cv_fig:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(idx_sorted, y_true_sorted, label="Observed")
                    ax.plot(idx_sorted, y_pred_sorted, label="Modelled")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Energy (MWh)")
                    ax.set_title(
                        f"{self.country_code} LOYO CV: r={corr:.2f}, NSE={r2:.2f}, "
                        f"MAE={mae:.0f} MWh, RMSE={rmse:.0f} MWh"
                    )
                    ax.legend()
                    fig.savefig(self.path_dict["fitting_result"],
                                bbox_inches="tight")
                    plt.close(fig)

        # ---------- Final fit on ALL training data ----------
        y_tr_all = self.to_1d(y_train)
        #TODO: update to tune weights
        if self.training_model.lower() == "hybrid":
            if tuning_weights == True:
                w_rf_final = self.tune_hybrid_model_weight(X_train, self.to_1d(y_train), years, rf_base, ridge_base, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1])
                print(f"Hybrid model weight: {w_rf_final}")
            else:
                w_rf_final = 0.5

            rf = clone(rf_base).fit(X_train, y_tr_all)
            rg = clone(ridge_base).fit(X_train, y_tr_all)
            y_pred_train = w_rf_final * rf.predict(X_train) + (1 - w_rf_final) * rg.predict(X_train)
            y_pred_train[y_pred_train<0] = 0
            if X_model is not None and not X_model.empty:
                y_model = w_rf_final * rf.predict(X_model) + (1 - w_rf_final) * rg.predict(X_model)
                y_model[y_model<0] = 0
                if cap is not None:
                    y_model = np.minimum(y_model, cap*24)
            else:
                y_model = None

        else:
            model = clone(base_model)
            model.fit(X_train, y_tr_all)

            y_pred_train = model.predict(X_train) # check fitting performance

            y_pred_train = np.maximum(y_pred_train, 0.0)

            if X_model is not None and not X_model.empty:
                y_model = model.predict(X_model)
                y_model = np.maximum(y_model, 0.0)
                if cap is not None:
                    y_model = np.minimum(y_model, cap*24)
            else:
                y_model = None

        if cap is not None:
            y_pred_train = np.minimum(y_pred_train, cap*24)

        #y_train = y_train*y_max.values[0]
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        mae = mean_absolute_error(y_train, y_pred_train)
        r2 = r2_score(y_train, y_pred_train)
        corr = self.calc_pearson(y_train, y_pred_train)
    
        fig, ax = plt.subplots(figsize=(10, 10 if y_model is not None else 5), nrows=2 if y_model is not None else 1)

        if y_model is not None:
            ax_plt = ax[0]
        else:
            ax_plt = ax
        ax_plt.plot(y_train.index, y_pred_train, label='Fitted', color='blue')
        ax_plt.plot(y_train.index, y_train, label='Observed', color='red')
        ax_plt.set_title(f'Fitting Results: RMSE={int(rmse)}MWh, MAE={int(mae)}MWh, R2={r2:.3f}, Corr={corr:.3f}')
        ax_plt.legend()
        ax_plt.set_xlabel('Date')
        ax_plt.set_ylabel('Energy (MWh)')

        if y_model is not None:
            ax[1].plot(y_model, label=f'Predicted {X_model.index.year.unique().to_list()}', color='green')
            ax[1].set_title('Modelled for years: ' + ', '.join(map(str, X_model.index.year.unique().to_list())))
            ax[1].set_xlabel('Date')
            ax[1].set_ylabel('Energy (MWh)')

            plt.tight_layout()
        plt.savefig(self.path_dict['pred_fig_path'] / f'{self.country_code}_model_training_results.png')
        plt.close()


        y_model = pd.DataFrame(y_model, index=X_model.index, columns=y_train.columns)
        
        return y_model


    def load_process_data(self, freq='W-SUN'):
        

        disc= pd.read_csv(self.path_dict['disc_file'], index_col=0, parse_dates=True)
        y = pd.read_csv(self.path_dict['data_file'], index_col=0, parse_dates=True) 
        disc.index = pd.to_datetime(disc.index, utc=True)
        disc = disc[~disc.index.duplicated(keep="first")] # incase some x variables are duplicated
        
        y.index = pd.to_datetime(y.index, utc=True)
        if self.hydro_type == 'hror' :
            if self.country_code not in ["ITSU",'ITCS',"ITCN"]:
                y = y.resample('h').mean()
                print('15min data is converted to 1hour data')
                cap = y.max().values[0]   #physical caps for daily generation of run-of-river hydropower 
                if freq == 'ME':
                    cap = cap*30
                elif freq == 'W-SUN':
                    cap = cap*7
            else:
                cap = y.max().values[0]
                if freq == 'ME':
                    cap = cap/7*30

        else:
            cap = None

        y = y.resample(freq).sum().dropna(axis=0).shift(freq='24h')
        if self.hydro_type == 'hror' and freq == 'W-SUN' or freq == 'ME':
            y = y.drop([y.index[0],y.index[-1]])

        disc_weekly= disc.resample(freq).sum().dropna(axis=0).shift(freq='24h')

        # if self.hydro_type == 'hror':
        #     disc_weekly = self.fill_missing_first_day(disc_weekly)
        # else:
        #     disc_weekly = self.fill_partial_boundary_weeks(disc_weekly)

        #disc_lagged = optimal_lags(y, disc_weekly, max_lag=5).dropna(axis=0)
        disc_lagged = disc_weekly.copy()
    

        return disc_lagged, y, cap
    

    def train_test_split(self, X, y, pred_years):
        if pred_years == 'all':
            pred_years = np.range(1980,2026).tolist()
        else:
            pass
        #All historical data are used for training
        if self.hydro_type == 'hror':
            if self.country_code == 'CZ':
                y_train = y[y.index.year >=2019]
            elif self.country_code in ['BA', 'NO1', 'NO2', "NO3", "NO4", "NO5"]:
                y_train = y[y.index.year >=2022]

            elif self.country_code == 'ITCN':
                y_train = y[y.index.year >=2022]
            elif self.country_code == 'ITSA':
                y_train = y[y.index.year >=2016]
            else:
                y_train = y

        else:
            y_train = y

        X_train = X[X.index.year.isin(y.index.year.unique())]


        X_model= X[X.index.year.isin(pred_years)]

        start_time = max(X_train.index.min(), y_train.index.min())
        end_time = min(X_train.index.max(), y_train.index.max())
        X_train = X_train[start_time:end_time]
        y_train = y_train[start_time:end_time]


        return X_train, y_train, X_model
    
    def fill_missing_first_day(self, df):
        df = df.copy()
        years = df.index.year.unique()

        for year in years:
            year_mask = df.index.year == year
            first_day = df.loc[year_mask].index.min()          
            jan1 = pd.Timestamp(year=year, month=1, day=1)      

            if first_day != jan1:
                # fill Jan 1 with the first available day in that year (Jan 2)
                df.loc[jan1] = df.loc[first_day].to_numpy()
                print(f'Filling missing first day for year {year}')

        return df.sort_index()
    


    def fill_partial_boundary_weeks(self, df): # this is for weekly resolution
        df_w = df.copy()
        idx = pd.DatetimeIndex(df_w.index)

        week_year = pd.Series((idx - pd.Timedelta(days=3)).year, index=df_w.index, name="week_year")
        years = np.array(sorted(week_year.unique()))
        year_set = set(map(int, years))

        for yr in years:
            yr = int(yr)
            mask = (week_year == yr)
            if not mask.any():
                continue   

            first_week = idx[mask].min()
            last_week = idx[mask].max()

            jan_1 = pd.Timestamp(year=yr, month=1, day=1)
            dec_31 = pd.Timestamp(year=yr, month=12, day=31)

            if (yr-1) not in year_set:
                week1 = first_week - pd.Timedelta(days=6)

                start_in_year = max(week1, jan_1)
                days_present = (first_week - start_in_year).days +1 
                if days_present < 7:
                    df_w.loc[first_week] = df_w.loc[first_week] *(7/days_present)

                if (yr+1) not in year_set:
                    week1 = last_week - pd.Timedelta(days=6)
                    end_in_year = min(last_week, dec_31)
                    days_present = (end_in_year - week1).days +1
                    if days_present < 7:
                        df_w.loc[last_week] = df_w.loc[last_week] *(7/days_present)
            if len(years):
                all_years = set(range(int(years.min()), int(years.max()) + 1))
                gaps = sorted(all_years - year_set)
            else:
                gaps = []
            
        print("There are %d years missing: %s" % (len(gaps), gaps))
        return df_w




    def modelled_data_main(self):

        if self.hydro_type == 'hror':
            freq = 'D'
            if self.country_code in ['HR', 'CZ', 'LT', 'FI']:
                freq = 'ME' # monthly resolution for somea areas
            elif self.country_code in ['PL', "ITSU",'ITCS',"ITCN"]:
                freq = 'W-SUN'
        else:
            freq = 'W-SUN'
        
        X, y, cap = self.load_process_data(freq)
        if self.pred_years == 'None':
            self.pred_years = list(range(y.index.year.min(), y.index.year.max()+1))
        else:
            pass

        X_train, y_train, X_model = self.train_test_split(X, y, self.pred_years.copy())
        y_model = self.model_train(X_train, y_train, X_model, cap)
        y_model.index = pd.to_datetime(y_model.index.strftime("%Y-%m-%d")) 
        if freq == 'W-SUN':
            y_model.index = y_model.index - pd.Timedelta(days=7)
        elif freq == 'ME':
            y_model = y_model.set_index(y_model.index+pd.DateOffset(months=-1)) #MONTHLY
        else:
            y_model = y_model.set_index(y_model.index+pd.DateOffset(days=-1))

        y_model.to_csv(self.path_dict['pred_data_file'], sep=',')

        print("Model training and prediction completed.")


