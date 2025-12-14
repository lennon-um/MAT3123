import warnings
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import lightgbm as lgb
import optuna
import shap
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# 환경 설정

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["font.family"] = "sans-serif"


class XOM_Absolute_Model:
    def __init__(self, years=10):
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=365 * years)

        self.tickers = {
            "Target": "XOM",
            "Market": "^GSPC",
            "WTI": "CL=F",
            "Brent": "BZ=F",
            "Curve_Short": "USO",
            "Curve_Long": "USL",
            "VIX": "^VIX",
            "OVX": "^OVX",  
        }

    def fetch_data(self):
       
        print(f">> Downloading Data ({self.start_date.date()} ~ )...")

        try:
            raw = yf.download(
                list(self.tickers.values()),
                start=self.start_date,
                end=self.end_date,
                progress=False,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
            )
        except Exception as e:
            print(f"Download Error: {e}")
            return pd.DataFrame()

        df_list = []
        if not isinstance(raw.columns, pd.MultiIndex):
            print("Unexpected yfinance format. Try again or update yfinance.")
            return pd.DataFrame()

        top = raw.columns.get_level_values(0)

        for key, ticker in self.tickers.items():
            if ticker not in top:
                continue
            cols = raw[ticker].columns
            if "Adj Close" in cols:
                s = raw[ticker]["Adj Close"].copy()
            elif "Close" in cols:
                s = raw[ticker]["Close"].copy()
            else:
                continue
            s.name = key
            df_list.append(s)

        if not df_list:
            return pd.DataFrame()

        df = pd.concat(df_list, axis=1).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.asfreq("B")
        df.ffill(inplace=True)

        print(">> Fetching Dividend History...")
        try:
            xom = yf.Ticker("XOM")
            divs = xom.dividends.copy()
            div = divs.reindex(df.index, fill_value=0.0)
            div.name = "Div_Payout"
            df = pd.concat([df, div], axis=1)
        except Exception as e:
            print(f"Dividend fetch failed: {e}")
            df["Div_Payout"] = 0.0

        df.dropna(inplace=True)
        return df

    def engineer_features(self, df):
        """
        """
        data = df.copy()

        # --- 배당수익률 계산 ---
        # 1년 누적 배당 / 현재가 (%)
        data["Div_TTM"] = data["Div_Payout"].rolling(252).sum()
        data["Div_Yield_TTM"] = (data["Div_TTM"] / (data["Target"] + 1e-9)) * 100.0

        # --- 오일 커브---
        data["Oil_Curve_Slope"] = data["Curve_Short"] / (data["Curve_Long"] + 1e-9)
        data["Oil_Curve_Slope_Change20"] = data["Oil_Curve_Slope"].diff(20)

        # --- 유가스프레드---
        data["WTI_Brent_Spread"] = data["WTI"] - data["Brent"]
        data["WTI_Brent_Spread_Change5"] = data["WTI_Brent_Spread"].diff(5)

        # --- 퀀트 맨날 쓰는거 (Target absolute) ---
        data["RSI_14"] = ta.rsi(data["Target"], length=14)

        macd = ta.macd(data["Target"], fast=12, slow=26, signal=9)
        # macd columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        data = pd.concat([data, macd], axis=1)

        sma60 = ta.sma(data["Target"], length=60)
        data["Disparity_MA60"] = (data["Target"] / (sma60 + 1e-9))

        # --- 위험지수 ---
        if "VIX" in data.columns:
            data["VIX_Level"] = data["VIX"]
        if "OVX" in data.columns:
            data["OVX_Level"] = data["OVX"]
            data["OVX_VIX_Ratio"] = data["OVX"] / (data["VIX"] + 1e-9)

        base_feats = [
            "Div_Yield_TTM",
            "Oil_Curve_Slope",
            "Oil_Curve_Slope_Change20",
            "WTI_Brent_Spread",
            "WTI_Brent_Spread_Change5",
            "RSI_14",
            "Disparity_MA60",
            "VIX_Level",
            "OVX_Level",
            "OVX_VIX_Ratio",
        ]
        for c in ["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9"]:
            if c in data.columns:
                base_feats.append(c)

        base_feats = [c for c in base_feats if c in data.columns]

        for col in base_feats:
            for lag in [1, 2, 3, 5, 10, 20]:
                data[f"{col}_Lag{lag}"] = data[col].shift(lag)

        # 다음날 가격 
        data["Target_Return"] = data["Target"].pct_change().shift(-1)

        data.dropna(inplace=True)
        return data

    def run_optimization(self, X_train, y_train):
        def objective(trial):
            param = {
                "objective": "regression",
                "metric": "rmse",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "n_jobs": -1,
                "random_state": 42,
                "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            }

            tscv = TimeSeriesSplit(n_splits=3)
            scores = []

            for tr_idx, va_idx in tscv.split(X_train):
                x_t, x_v = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                y_t, y_v = y_train.iloc[tr_idx], y_train.iloc[va_idx]

                train_ds = lgb.Dataset(x_t, label=y_t)
                val_ds = lgb.Dataset(x_v, label=y_v, reference=train_ds)

                callbacks = [
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0),
                ]

                gbm = lgb.train(
                    param,
                    train_ds,
                    num_boost_round=3000,
                    valid_sets=[val_ds],
                    callbacks=callbacks,
                )
                preds = gbm.predict(x_v, num_iteration=gbm.best_iteration)
                scores.append(np.sqrt(mean_squared_error(y_v, preds)))

            return float(np.mean(scores))

        print(">> Tuning Hyperparameters...")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)
        return study.best_params


def main():
    print("=== XOM Prediction (Absolute Indicators & Dividend) ===")

    modeler = XOM_Absolute_Model(years=10)
    df = modeler.fetch_data()
    if df.empty:
        print("데이터 로드 실패.")
        return

    df_feat = modeler.engineer_features(df)

    target_col = "Target_Return"
    # 가격 복원용으로 실제 가격은 따로 보관
    price_series = df_feat["Target"].copy()

    # 학습 피처 = 타겟/원본가격  제거
    drop_cols = {"Target_Return", "Div_Payout", "Div_TTM"}
    features = [c for c in df_feat.columns if c not in drop_cols and c != "Target"]

    test_size = 252
    train_df = df_feat.iloc[:-test_size].copy()
    test_df = df_feat.iloc[-test_size:].copy()

    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    best_params = modeler.run_optimization(X_train, y_train)

    print(">> Training Final Model...")
    best_params.update(
        {"objective": "regression", "metric": "rmse", "verbosity": -1, "n_jobs": -1, "random_state": 42}
    )

    train_ds = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(best_params, train_ds, num_boost_round=2000)

    # 예측 (return)
    pred_returns = model.predict(X_test)

    # 가격 복원: t-1 실제가격 * (1 + 예측수익률) = t 예측가격
  
    # test 구간의 첫 날은 이전 날 실제가격 사용
  
    prev_prices = df_feat["Target"].shift(1).loc[X_test.index]
    pred_prices = prev_prices.values * (1.0 + pred_returns)
    real_prices = df_feat["Target"].loc[X_test.index].values

    rmse_ret = np.sqrt(mean_squared_error(y_test, pred_returns))
    print(f"\nTest RMSE (Return): {rmse_ret:.6f}")

    # 그래프
  
    plt.figure(figsize=(12, 6))
    plt.plot(X_test.index, real_prices, label="Actual Price", color="black", alpha=0.6)
    plt.plot(X_test.index, pred_prices, label="Predicted Price", color="red", linestyle="--")
    plt.title(f"XOM Price (from predicted returns) | RMSE(Return)={rmse_ret:.6f}")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # SHAP
    print("\n>> Analyzing Feature Importance (SHAP)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Feature Importance (Absolute Indicators)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
