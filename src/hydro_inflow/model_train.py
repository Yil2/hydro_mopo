import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut


class ModelTrain:
    DEFAULT_RECENT_YEARS = list(range(2015, 2025))
    DEFAULT_ALL_YEARS = list(range(1980, 2025))
    HYBRID_WEIGHT_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
    HYBRID_FINAL_WEIGHT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def __init__(self, config_obj, path_obj):
        self.config = config_obj.config
        self.path_dict = path_obj.path_dict
        self.country_code = config_obj.country_code
        self.hydro_type = config_obj.hydro_type
        self.training_model = self.config["algorithm"]
        self.pred_years = self._normalize_pred_years(self.config.get("pred_years"))

    def _normalize_pred_years(self, pred_years):
        if pred_years in (None, "None"):
            return self.DEFAULT_RECENT_YEARS.copy()
        if pred_years == "all":
            return self.DEFAULT_ALL_YEARS.copy()
        if isinstance(pred_years, list):
            return sorted({int(year) for year in pred_years})
        raise ValueError(f"Unsupported pred_years value: {pred_years}")

    def _build_rf_estimator(self, grid_search=False):
        param_grid = {
            "n_estimators": [800],
            "max_depth": [5, 9, 12, 15],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", 0.33],
            "bootstrap": [True],
            "oob_score": [False],
            "max_samples": [0.8],
        }

        if grid_search:
            return GridSearchCV(
                estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
                param_grid=param_grid,
                scoring="neg_root_mean_squared_error",
                cv=5,
                n_jobs=-1,
                verbose=1,
                refit=True,
            )

        return RandomForestRegressor(
            n_estimators=800,
            max_features="sqrt",
            max_depth=9,
            min_samples_leaf=1,
            max_samples=0.8,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )

    def _build_base_model(self, X_train, grid_search=False):
        training_model = self.training_model.lower()

        if training_model == "random forest":
            return self._build_rf_estimator(grid_search), None, None

        if training_model == "linear regression":
            return LinearRegression(positive=True, fit_intercept=False), None, None

        if training_model == "hybrid":
            p_num = X_train.shape[0]
            alphas = [0.1, 1.0, 5.0, 10.0, 15.0, max(2, int(p_num / 2)), max(2, int(p_num))]
            rf_base = self._build_rf_estimator(grid_search)
            ridge_base = RidgeCV(alphas=alphas, cv=5, scoring="neg_root_mean_squared_error")
            base_model = VotingRegressor(
                estimators=[("rf", rf_base), ("ridge", ridge_base)],
                weights=[0.5, 0.5],
            )
            return base_model, rf_base, ridge_base

        raise ValueError(f"Unknown training_model: {self.training_model}")

    def _apply_prediction_constraints(self, prediction, cap):
        prediction = np.maximum(np.asarray(prediction), 0.0)
        if cap is not None:
            prediction = np.minimum(prediction, cap * 24)
        return prediction

    def _to_target_frame(self, values, index, y_train):
        columns = y_train.columns if isinstance(y_train, pd.DataFrame) else ["value"]
        return pd.DataFrame(values, index=index, columns=columns)

    def to_1d(self, y):
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"Expected 1 target column, got {y.shape[1]}")
            return y.iloc[:, 0].to_numpy()

        if isinstance(y, pd.Series):
            return y.to_numpy()

        return np.asarray(y).reshape(-1)

    def calc_pearson(self, a, b):
        a = np.asarray(a).ravel()
        b = np.asarray(b).ravel()
        if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
            return np.nan
        return pearsonr(a, b)[0]

    def tune_hybrid_model_weight(self, X, y, years, model_a, model_b, w_grid=None):
        if w_grid is None:
            w_grid = np.linspace(0, 1, 11)

        logo = LeaveOneGroupOut()
        best = {"w": 0.5, "rmse": np.inf}

        for w in w_grid:
            preds, trues = [], []

            for tr_idx, te_idx in logo.split(X, y, groups=years):
                model_left = clone(model_a)
                model_right = clone(model_b)

                model_left.fit(X.iloc[tr_idx], y[tr_idx])
                model_right.fit(X.iloc[tr_idx], y[tr_idx])
                pred = w * model_left.predict(X.iloc[te_idx]) + (1 - w) * model_right.predict(X.iloc[te_idx])

                preds.append(pred)
                trues.append(y[te_idx])

            yhat = np.concatenate(preds)
            ytrue = np.concatenate(trues)
            rmse = float(np.sqrt(mean_squared_error(ytrue, yhat)))

            if rmse < best["rmse"]:
                best = {"w": float(w), "rmse": rmse}

        return best["w"]

    def model_train(self, X_train, y_train, X_model, cap, cross_val=True, save_cv_fig=True):
        base_model, rf_base, ridge_base = self._build_base_model(X_train, grid_search=False)
        training_model = self.training_model.lower()
        y_train_1d = self.to_1d(y_train)
        years = X_train.index.year.to_numpy()

        if cross_val:
            logo = LeaveOneGroupOut()
            y_true_all = []
            y_pred_all = []
            idx_all = []

            for train_idx, test_idx in logo.split(X_train, y_train_1d, groups=years):
                X_tr = X_train.iloc[train_idx]
                X_te = X_train.iloc[test_idx]

                if X_te.empty:
                    continue

                y_tr = y_train_1d[train_idx]
                y_te = y_train_1d[test_idx]

                if training_model == "hybrid":
                    years_tr = X_tr.index.year.to_numpy()
                    w_rf = self.tune_hybrid_model_weight(
                        X_tr,
                        y_tr,
                        years_tr,
                        rf_base,
                        ridge_base,
                        self.HYBRID_WEIGHT_GRID,
                    )
                    rf = clone(rf_base).fit(X_tr, y_tr)
                    rg = clone(ridge_base).fit(X_tr, y_tr)
                    pred = w_rf * rf.predict(X_te) + (1 - w_rf) * rg.predict(X_te)
                else:
                    model = clone(base_model).fit(X_tr, y_tr)
                    pred = model.predict(X_te)

                pred = self._apply_prediction_constraints(pred, cap)
                y_true_all.append(y_te)
                y_pred_all.append(pred)
                idx_all.append(X_te.index)

            if idx_all:
                idx_all = np.concatenate([idx.to_numpy() for idx in idx_all])
                y_true_all = np.concatenate(y_true_all)
                y_pred_all = np.concatenate(y_pred_all)

                order = np.argsort(idx_all)
                idx_sorted = pd.to_datetime(idx_all[order])
                y_true_sorted = y_true_all[order]
                y_pred_sorted = y_pred_all[order]

                rmse = float(np.sqrt(mean_squared_error(y_true_sorted, y_pred_sorted)))
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
                    fig.savefig(self.path_dict["fitting_result"], bbox_inches="tight")
                    plt.close(fig)

        if training_model == "hybrid":
            w_rf_final = self.tune_hybrid_model_weight(
                X_train,
                y_train_1d,
                years,
                rf_base,
                ridge_base,
                self.HYBRID_FINAL_WEIGHT_GRID,
            )
            print(f"Hybrid model weight: {w_rf_final}")

            rf = clone(rf_base).fit(X_train, y_train_1d)
            rg = clone(ridge_base).fit(X_train, y_train_1d)
            y_pred_train = w_rf_final * rf.predict(X_train) + (1 - w_rf_final) * rg.predict(X_train)
            y_model = None
            if X_model is not None and not X_model.empty:
                y_model = w_rf_final * rf.predict(X_model) + (1 - w_rf_final) * rg.predict(X_model)
        else:
            model = clone(base_model)
            model.fit(X_train, y_train_1d)
            y_pred_train = model.predict(X_train)
            y_model = None
            if X_model is not None and not X_model.empty:
                y_model = model.predict(X_model)

        y_pred_train = self._apply_prediction_constraints(y_pred_train, cap)
        if y_model is not None:
            y_model = self._apply_prediction_constraints(y_model, cap)

        rmse = np.sqrt(mean_squared_error(y_train_1d, y_pred_train))
        mae = mean_absolute_error(y_train_1d, y_pred_train)
        r2 = r2_score(y_train_1d, y_pred_train)
        corr = self.calc_pearson(y_train_1d, y_pred_train)

        fig, ax = plt.subplots(figsize=(10, 10 if y_model is not None else 5), nrows=2 if y_model is not None else 1)
        ax_plt = ax[0] if y_model is not None else ax
        ax_plt.plot(y_train.index, y_pred_train, label="Fitted", color="blue")
        ax_plt.plot(y_train.index, y_train_1d, label="Observed", color="red")
        ax_plt.set_title(
            f"Fitting Results: RMSE={int(rmse)}MWh, MAE={int(mae)}MWh, R2={r2:.3f}, Corr={corr:.3f}"
        )
        ax_plt.legend()
        ax_plt.set_xlabel("Date")
        ax_plt.set_ylabel("Energy (MWh)")

        if y_model is not None:
            ax[1].plot(y_model, label=f"Predicted {X_model.index.year.unique().to_list()}", color="green")
            ax[1].set_title("Modelled for years: " + ", ".join(map(str, X_model.index.year.unique().to_list())))
            ax[1].set_xlabel("Date")
            ax[1].set_ylabel("Energy (MWh)")
            plt.tight_layout()

        plt.savefig(self.path_dict["pred_fig_path"] / f"{self.country_code}_model_training_results.png")
        plt.close()

        if y_model is None:
            return None

        return self._to_target_frame(y_model, X_model.index, y_train)

    def load_process_data(self, freq="W-SUN"):
        disc = pd.read_csv(self.path_dict["disc_file"], index_col=0, parse_dates=True)
        y = pd.read_csv(self.path_dict["data_file"], index_col=0, parse_dates=True)
        disc.index = pd.to_datetime(disc.index, utc=True)
        disc = disc[~disc.index.duplicated(keep="first")]

        y.index = pd.to_datetime(y.index, utc=True)
        if self.hydro_type == "hror":
            if self.country_code not in ["ITSU", "ITCS", "ITCN"]:
                y = y.resample("h").mean()
                print("15min data is converted to 1hour data")
                cap = y.max().values[0]
                if freq == "ME":
                    cap = cap * 30
                elif freq == "W-SUN":
                    cap = cap * 7
            else:
                cap = y.max().values[0]
                if freq == "ME":
                    cap = cap / 7 * 30
        else:
            cap = None

        y = y.resample(freq).sum().dropna(axis=0).shift(freq="24h")
        if self.hydro_type == "hror" and freq in {"W-SUN", "ME"} and len(y.index) >= 2:
            y = y.drop([y.index[0], y.index[-1]])

        disc_resampled = disc.resample(freq).sum().dropna(axis=0).shift(freq="24h")
        disc_lagged = disc_resampled.copy()
        return disc_lagged, y, cap

    def train_test_split(self, X, y, pred_years):
        if self.hydro_type == "hror":
            if self.country_code == "CZ":
                y_train = y[y.index.year >= 2019]
            elif self.country_code in ["BA", "NO1", "NO2", "NO3", "NO4", "NO5"]:
                y_train = y[y.index.year >= 2022]
            elif self.country_code == "ITCN":
                y_train = y[y.index.year >= 2022]
            elif self.country_code == "ITSA":
                y_train = y[y.index.year >= 2016]
            else:
                y_train = y
        else:
            y_train = y

        X_train = X[X.index.year.isin(y_train.index.year.unique())]
        X_model = X[X.index.year.isin(pred_years)].copy()

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
                df.loc[jan1] = df.loc[first_day].to_numpy()
                print(f"Filling missing first day for year {year}")

        return df.sort_index()

    def fill_partial_boundary_weeks(self, df):
        df_w = df.copy()
        idx = pd.DatetimeIndex(df_w.index)

        week_year = pd.Series((idx - pd.Timedelta(days=3)).year, index=df_w.index, name="week_year")
        years = np.array(sorted(week_year.unique()))
        year_set = set(map(int, years))

        for yr in years:
            yr = int(yr)
            mask = week_year == yr
            if not mask.any():
                continue

            first_week = idx[mask].min()
            last_week = idx[mask].max()

            jan_1 = pd.Timestamp(year=yr, month=1, day=1)
            dec_31 = pd.Timestamp(year=yr, month=12, day=31)

            if (yr - 1) not in year_set:
                week1 = first_week - pd.Timedelta(days=6)
                start_in_year = max(week1, jan_1)
                days_present = (first_week - start_in_year).days + 1
                if days_present < 7:
                    df_w.loc[first_week] = df_w.loc[first_week] * (7 / days_present)

                if (yr + 1) not in year_set:
                    week1 = last_week - pd.Timedelta(days=6)
                    end_in_year = min(last_week, dec_31)
                    days_present = (end_in_year - week1).days + 1
                    if days_present < 7:
                        df_w.loc[last_week] = df_w.loc[last_week] * (7 / days_present)

        if len(years):
            all_years = set(range(int(years.min()), int(years.max()) + 1))
            gaps = sorted(all_years - year_set)
        else:
            gaps = []

        print("There are %d years missing: %s" % (len(gaps), gaps))
        return df_w

    def modelled_data_main(self):
        if self.hydro_type == "hror":
            freq = "D"
            if self.country_code in ["HR", "CZ", "LT", "FI"]:
                freq = "ME"
            elif self.country_code in ["PL", "ITSU", "ITCS", "ITCN"]:
                freq = "W-SUN"
        else:
            freq = "W-SUN"

        X, y, cap = self.load_process_data(freq)
        X_train, y_train, X_model = self.train_test_split(X, y, self.pred_years.copy())
        y_model = self.model_train(X_train, y_train, X_model, cap)

        if y_model is None or y_model.empty:
            print("No prediction years available in discharge data. Skipping model output export.")
            return

        y_model.index = pd.to_datetime(y_model.index.strftime("%Y-%m-%d"))
        if freq == "W-SUN":
            y_model.index = y_model.index - pd.Timedelta(days=7)
        elif freq == "ME":
            y_model = y_model.set_index(y_model.index + pd.DateOffset(months=-1))
        else:
            y_model = y_model.set_index(y_model.index + pd.DateOffset(days=-1))

        y_model.to_csv(self.path_dict["pred_data_file"], sep=",")
        print("Model training and prediction completed.")
