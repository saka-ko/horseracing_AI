import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. データの読み込みと前処理
# ==========================================
file_path = '過去１か月分レース結果.csv'

# 文字コードを自動判別して読み込み
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='shift_jis')

print(f"元データ件数: {len(df)}")

# --- データクリーニング関数 ---
def clean_numeric(x):
    """全角数字などを半角数値に変換。変換不可ならNaN"""
    if pd.isna(x): return np.nan
    # 全角→半角変換
    x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
    try:
        return float(x_str)
    except ValueError:
        return np.nan

# 着順をきれいにする（目的変数）
df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num']) # 着順が不明なデータ（中止など）は削除
df['着順_num'] = df['着順_num'].astype(int)

# 前走着順などもきれいにする（特徴量）
if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# --- 特徴量（AIに入力するデータ）の選定 ---
# ここで「レース前に分かる情報」だけを厳選します
features = [
    '指数',           # ZI値（前日までの能力値なのでOK）
    '前走補正',       # ★修正：今回の「補正」を消して、前回の補正を追加
    '前走着順_num',
    '前走人気',
    '前走単勝オッズ',
    '前走上り3F',
    '前走着差タイム',
    '前PCI',          # 前走のペース判断指標
    '前走芝・ダ',     # 前走の条件
    '前走距離',
    '斤量',
    '馬番',           # 枠順
    '馬体重', 
    '馬体重増減',
    '年齢',
    '間隔',           # レース間隔
    '騎手コード',
    '調教師コード',
    '種牡馬',
]

# 実際にCSVに存在するカラムだけを使うようにフィルタリング
features = [f for f in features if f in df.columns]
print(f"学習に使用する項目数: {len(features)}")

# カテゴリ変数（文字データ）を数値に変換する
# 前走の条件などもカテゴリ変数として処理
categorical_cols = ['場所', '芝・ダ', '馬場状態', '種牡馬', '騎手コード', '調教師コード', '前走芝・ダ']
encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        # 欠損値は 'unknown' として扱う
        df[col] = df[col].fillna('unknown')
        # 数値等の型混在を防ぐため文字列化
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        # 特徴量リストになければ追加
        if col not in features and col in df.columns: # 前走芝・ダなどは既に入っている
             pass 

# 数値データの欠損値を平均値などで埋める
num_features = [f for f in features if f not in categorical_cols]
for col in num_features:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') # 強制的に数値化
        df[col] = df[col].fillna(df[col].mean()) # 平均値埋め

# ==========================================
# 2. AIモデルの作成と学習 (LightGBM)
# ==========================================

# 目的変数を作成
# Case A: 1着になる確率（勝率）
df['target_win'] = (df['着順_num'] == 1).astype(int)
# Case B: 3着以内になる確率（複勝率）
df['target_top3'] = (df['着順_num'] <= 3).astype(int)

# 学習データとテストデータに分割
X = df[features]
y_win = df['target_win']
y_top3 = df['target_top3']

X_train, X_test, y_train_win, y_test_win = train_test_split(X, y_win, test_size=0.2, random_state=42)
_, _, y_train_top3, y_test_top3 = train_test_split(X, y_top3, test_size=0.2, random_state=42)

print("学習開始...")

# --- モデル1: 勝率予測モデル ---
model_win = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model_win.fit(X_train, y_train_win)

# --- モデル2: 複勝率予測モデル ---
model_top3 = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model_top3.fit(X_train, y_train_top3)

# ==========================================
# 3. 結果の確認と予測
# ==========================================

# テストデータで予測
prob_win = model_win.predict_proba(X_test)[:, 1]
prob_top3 = model_top3.predict_proba(X_test)[:, 1]

# 結果をデータフレームにまとめる
results = X_test.copy()
results['実際の着順'] = df.loc[X_test.index, '着順_num']
results['AI勝率予測(%)'] = (prob_win * 100).round(1)
results['AI複勝率予測(%)'] = (prob_top3 * 100).round(1)
results['馬名'] = df.loc[X_test.index, '馬名']

# AIが高評価した馬トップ10（複勝率ベース）
print("\n=== AIが複勝率が高いと予測した馬トップ10 ===")
print(results.sort_values('AI複勝率予測(%)', ascending=False)[['馬名', '実際の着順', 'AI勝率予測(%)', 'AI複勝率予測(%)']].head(10))

# 精度評価 (AUC)
auc_win = roc_auc_score(y_test_win, prob_win)
auc_top3 = roc_auc_score(y_test_top3, prob_top3)

print(f"\nモデル精度(AUC) - 勝率モデル: {auc_win:.4f}")
print(f"モデル精度(AUC) - 複勝率モデル: {auc_top3:.4f}")
print("(※ 0.5がランダム、0.7を超えると優秀、0.8以上は非常に優秀)")

# 重要度表示
print("\n=== 勝率予測で重要だったデータ項目 ===")
importance = pd.DataFrame({'feature': features, 'importance': model_win.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(5))