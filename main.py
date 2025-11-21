import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. データの読み込み
# ==========================================
# ★注意: ここを5年分のCSVファイル名に書き換えてください
file_path = '過去5年分レース結果.csv' 

print(f"データを読み込んでいます... ({file_path})")

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='shift_jis')
        except:
            print("エラー: ファイルの読み込みに失敗しました。エンコーディングを確認してください。")
            raise

print(f"データ読み込み完了: {len(df)}件")

# ==========================================
# 2. 特徴量エンジニアリング
# ==========================================

def clean_numeric(x):
    if pd.isna(x): return np.nan
    x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
    try:
        return float(x_str)
    except ValueError:
        return np.nan

# 基本データの処理
df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num'])
df['着順_num'] = df['着順_num'].astype(int)

if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# --- Factor 1: コース適性 (Course ID) ---
df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)

# --- Factor 2: 黄金コンビ (Jockey-Trainer Combo) ---
df['騎手調教師コンビ'] = df['騎手コード'].astype(str) + "_" + df['調教師コード'].astype(str)

# --- Factor 3: 展開・先行力 (Positioning) ---
# 4コーナーの通過順位を数値化（逃げ・先行馬を判別）
if '前走4角' in df.columns:
    df['前走脚質数値'] = df['前走4角'].apply(clean_numeric).fillna(10) # 空欄は中団(10番手)扱い
else:
    df['前走脚質数値'] = 10

# --- 特徴量の選定 ---
features = [
    '指数',           # ZI値
    '前走補正',       # スピード指数
    '前走着順_num',
    '前走人気',
    '前走単勝オッズ',
    '前走上り3F',
    '前走着差タイム',
    '前PCI',          # ペース配分
    '前走脚質数値',     # ★展開予想用
    '騎手調教師コンビ', # ★相性用
    'コースID',       # ★コース適性用
    '斤量',
    '馬番',           
    '馬体重', 
    '馬体重増減',
    '年齢',
    '間隔',           
    '騎手コード',
    '調教師コード',
    '種牡馬',
    '場所',
    '芝・ダ',
    '距離'
]

# CSVに存在する列だけを使用
features = [f for f in features if f in df.columns]
print(f"学習に使用する特徴量: {len(features)}個")

# --- カテゴリ変数の処理 ---
categorical_cols = ['場所', '芝・ダ', '馬場状態', '種牡馬', '騎手コード', '調教師コード', 
                    '前走芝・ダ', 'コースID', '騎手調教師コンビ']

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        # 欠損値は'unknown'にし、全て文字列として扱う
        df[col] = df[col].fillna('unknown').astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# 数値データの欠損値埋め（前回の修正を適用済み）
num_features = [f for f in features if f not in categorical_cols]
for col in num_features:
    if col in df.columns:
        # 一度数値変換してから変数に入れる
        temp_col = pd.to_numeric(df[col], errors='coerce')
        df[col] = temp_col.fillna(temp_col.mean())

# ==========================================
# 3. モデル学習と評価 (LightGBM)
# ==========================================

df['target_win'] = (df['着順_num'] == 1).astype(int)
df['target_top3'] = (df['着順_num'] <= 3).astype(int)

X = df[features]
y_win = df['target_win']
y_top3 = df['target_top3']

# 5年分だとデータが多いので、test_sizeは0.2のままで十分な検証数が確保できます
X_train, X_test, y_train_win, y_test_win = train_test_split(X, y_win, test_size=0.2, random_state=42)
_, _, y_train_top3, y_test_top3 = train_test_split(X, y_top3, test_size=0.2, random_state=42)

print("\n学習開始... (データ量が増えたため数分かかる場合があります)")

# モデル定義（データが多いので決定木の数を少し増やしても良いですが、まずは標準で）
model_win = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model_win.fit(X_train, y_train_win)

model_top3 = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model_top3.fit(X_train, y_train_top3)

# ==========================================
# 4. 結果の確認
# ==========================================

prob_win = model_win.predict_proba(X_test)[:, 1]
prob_top3 = model_top3.predict_proba(X_test)[:, 1]

auc_win = roc_auc_score(y_test_win, prob_win)
auc_top3 = roc_auc_score(y_test_top3, prob_top3)

print("\n" + "="*40)
print(" 　　　モデル精度レポート (5年データ)")
print("="*40)
print(f"【勝率モデル (AUC)】: {auc_win:.4f}")
print(f"【複勝率モデル (AUC)】: {auc_top3:.4f}")
print("="*40 + "\n")

print("=== 重要度ランキングトップ15 ===")
importance = pd.DataFrame({'feature': features, 'importance': model_win.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(15))