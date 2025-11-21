import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. データの読み込み
# ==========================================
file_path = '過去１か月分レース結果.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='shift_jis')

print(f"データ読み込み完了: {len(df)}件")

# ==========================================
# 2. 特徴量エンジニアリング（ここが進化！）
# ==========================================

# --- 基本的なデータクリーニング ---
def clean_numeric(x):
    if pd.isna(x): return np.nan
    x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
    try:
        return float(x_str)
    except ValueError:
        return np.nan

df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num'])
df['着順_num'] = df['着順_num'].astype(int)

if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# --- ★追加機能: コースIDの作成 ---
# 「場所」+「芝・ダ」+「距離」を組み合わせて、ひとつの「コースID」を作る
# 例: "東京" + "芝" + "2400" -> "東京芝2400"
# これにより、AIは「コースごとの特性」を深く学習できるようになります。
df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)

# --- 特徴量の選定 ---
features = [
    '指数',           
    '前走補正',       
    '前走着順_num',
    '前走人気',
    '前走単勝オッズ',
    '前走上り3F',
    '前走着差タイム',
    '前PCI',          
    '前走芝・ダ',     
    '前走距離',
    '斤量',
    '馬番',           
    '馬体重', 
    '馬体重増減',
    '年齢',
    '間隔',           
    '騎手コード',
    '調教師コード',
    '種牡馬',
    'コースID',       # ★ここに追加！
    '場所',           # 個別の要素も残しておくと、相互作用を学習しやすい
    '芝・ダ',
    '距離'
]

# 存在する列だけ使用
features = [f for f in features if f in df.columns]
print(f"学習に使用する特徴量: {features}")

# --- カテゴリ変数の処理 ---
categorical_cols = ['場所', '芝・ダ', '馬場状態', '種牡馬', '騎手コード', '調教師コード', '前走芝・ダ', 'コースID']
encoders = {}

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna('unknown').astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# 数値データの欠損値埋め
num_features = [f for f in features if f not in categorical_cols]
for col in num_features:
    if col in df.columns:
        # まず数値に変換する（文字はNaNになる）
        temp_col = pd.to_numeric(df[col], errors='coerce')
        # 変換後のデータを使って平均値を計算し、穴埋めする
        df[col] = temp_col.fillna(temp_col.mean())

# ==========================================
# 3. モデル学習と評価 (LightGBM)
# ==========================================

# 目的変数
df['target_win'] = (df['着順_num'] == 1).astype(int)      # 1着
df['target_top3'] = (df['着順_num'] <= 3).astype(int)     # 3着以内

X = df[features]
y_win = df['target_win']
y_top3 = df['target_top3']

# データを分割
X_train, X_test, y_train_win, y_test_win = train_test_split(X, y_win, test_size=0.2, random_state=42)
_, _, y_train_top3, y_test_top3 = train_test_split(X, y_top3, test_size=0.2, random_state=42)

print("\n学習開始...")

# 勝率モデル
model_win = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model_win.fit(X_train, y_train_win)

# 複勝率モデル
model_top3 = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model_top3.fit(X_train, y_train_top3)

# ==========================================
# 4. 結果の確認とベースライン比較
# ==========================================

prob_win = model_win.predict_proba(X_test)[:, 1]
prob_top3 = model_top3.predict_proba(X_test)[:, 1]

# AUCスコア算出
auc_win = roc_auc_score(y_test_win, prob_win)
auc_top3 = roc_auc_score(y_test_top3, prob_top3)

# 前回のベースラインスコア（参考値：あなたの前回の結果を入力）
baseline_auc_win = 0.7462 
baseline_auc_top3 = 0.7275

print("\n" + "="*40)
print(" 　　　モデル精度比較レポート")
print("="*40)
print(f"【勝率モデル (AUC)】")
print(f"  今回: {auc_win:.4f}")
print(f"  前回: {baseline_auc_win:.4f}")
diff_win = auc_win - baseline_auc_win
print(f"  差分: {'+' if diff_win >= 0 else ''}{diff_win:.4f}")

print("-" * 20)

print(f"【複勝率モデル (AUC)】")
print(f"  今回: {auc_top3:.4f}")
print(f"  前回: {baseline_auc_top3:.4f}")
diff_top3 = auc_top3 - baseline_auc_top3
print(f"  差分: {'+' if diff_top3 >= 0 else ''}{diff_top3:.4f}")
print("="*40 + "\n")

# 重要度表示
print("=== 今回の重要度ランキングトップ10 ===")
importance = pd.DataFrame({'feature': features, 'importance': model_win.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(10))