# this is the code for train and test by linear regression (Europe) OR scaling method (globally)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut
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
        #TODO: write pred_years by config_data_handel
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
            rmse = float(np.sqrt(np.mean((ytrue - yhat) ** 2)))

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

        if training_model == "random forest":
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
                # if self.training_model.lower() == "hybrid":
                #     years_tr = X_tr.index.year.to_numpy()

                #     w_rf = self.tune_hybrid_model_weight(X_tr, y_tr, years_tr, rf_base, ridge_base, [0.1,  0.3,0.5, 0.7,  0.9])

                #     rf = clone(rf_base).fit(X_tr, y_tr)
                #     rg = clone(ridge_base).fit(X_tr, y_tr)

                #     pred = w_rf * rf.predict(X_te) + (1 - w_rf) * rg.predict(X_te)

                
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
                    ax.plot(idx_sorted, y_pred_sorted, label="Predicted")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Energy (MWh)")
                    ax.set_title(
                        f"{self.country_code} LOYO CV: PCC={corr:.2f}, R2={r2:.2f}, "
                        f"MAE={mae:.2f} MWh, RMSE={rmse:.2f} MWh"
                    )
                    ax.legend()
                    fig.savefig(self.path_dict["fitting_result"],
                                bbox_inches="tight")
                    plt.close(fig)

        # ---------- Final fit on ALL training data ----------
        y_tr_all = self.to_1d(y_train)
        #TODO: update to tune weights
        # if self.training_model.lower() == "hybrid":
        #     w_rf_final = self.tune_hybrid_model_weight(X_train, self.to_1d(y_train), years, rf_base, ridge_base, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        #     print(f"Hybrid model weight: {w_rf_final}")
        #     rf = clone(rf_base).fit(X_train, y_tr_all)
        #     rg = clone(ridge_base).fit(X_train, y_tr_all)
        #     y_pred_train = w_rf_final * rf.predict(X_train) + (1 - w_rf_final) * rg.predict(X_train)
        #     y_pred_train[y_pred_train<0] = 0
        #     if X_model is not None and not X_model.empty:
        #         y_model = w_rf_final * rf.predict(X_model) + (1 - w_rf_final) * rg.predict(X_model)
        #         y_model[y_model<0] = 0
        #     else:
        #         y_model = None

        # else:
        model = clone(base_model)
        model.fit(X_train, y_tr_all)

        y_pred_train = model.predict(X_train) # check fitting performance
        y_pred_train = np.maximum(y_pred_train, 0.0)
        if cap is not None:
            y_pred_train = np.minimum(y_pred_train, cap*24)

        if X_model is not None and not X_model.empty:
            y_model = model.predict(X_model)
            y_model = np.maximum(y_model, 0.0)
            if cap is not None:
                y_model = np.minimum(y_model, cap*24)
        else:
            y_model = None

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
        y.index = pd.to_datetime(y.index, utc=True)
        if freq == 'D':
            y = y.resample('h').mean()
            print('Resampling to daily frequency')

            cap = y.max().values[0]   #physical caps for daily generation of run-of-river hydropower 
        
        else:
            cap = None

        y = y.resample(freq).sum().dropna(axis=0)
        disc_weekly= disc.resample(freq).sum().dropna(axis=0)

        #disc_lagged = optimal_lags(y, disc_weekly, max_lag=5).dropna(axis=0)
        disc_lagged = disc_weekly.copy()
    

        return disc_lagged, y, cap
    

    def train_test_split(self, X, y, pred_years):
        if pred_years == 'all':
            pred_years = np.range(1980,2026).tolist()
        else:
            pass
        #All historical data are used for training
        y_train = y
        X_train = X[X.index.year.isin(y.index.year.unique())]


        X_model= X[X.index.year.isin(pred_years)]

        start_time = max(X_train.index.min(), y_train.index.min())
        end_time = min(X_train.index.max(), y_train.index.max())
        X_train = X_train[start_time:end_time]
        y_train = y_train[start_time:end_time]


        return X_train, y_train, X_model
    

    def modelled_data_main(self):

        if self.hydro_type == 'hror':
            #TODO: physical cap of maximum daily generation
            freq = 'D'
        else:
            freq = 'W-SUN'

        X, y, cap = self.load_process_data(freq)
        if self.pred_years == 'None':
            self.pred_years = list(range(y.index.year.min(), y.index.year.max()+1))
        else:
            pass

        X_train, y_train, X_model = self.train_test_split(X, y, self.pred_years.copy())
        y_model = self.model_train(X_train, y_train, X_model, cap)

        y_model.to_csv(self.path_dict['pred_data_file'], sep=',')

        print("Model training and prediction completed.")


