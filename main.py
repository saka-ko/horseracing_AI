import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# 1. データの読み込み
# ==========================================
file_path = 'race_data_5years.csv' 

print(f"データを読み込んでいます... ({file_path})")
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='shift_jis', errors='ignore')

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

df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num'])
df['着順_num'] = df['着順_num'].astype(int)

if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# --- Factor Creation ---
pci_cols = ['前PCI', '前走PCI', '前RPCI', '前走RPCI', '前PCI3', '前走PCI3']
for col in pci_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['前走PCI_val'] = df['前PCI'] if '前PCI' in df.columns else df['前走PCI'] if '前走PCI' in df.columns else 50
df['前走RPCI_val'] = df['前RPCI'] if '前RPCI' in df.columns else df['前走RPCI'] if '前走RPCI' in df.columns else 50

if '前走Ave-3F' in df.columns:
    df['前走Ave3F'] = pd.to_numeric(df['前走Ave-3F'], errors='coerce')
else:
    df['前走Ave3F'] = np.nan

if '前走4角' in df.columns:
    df['前走脚質数値'] = df['前走4角'].apply(clean_numeric).fillna(10)
else:
    df['前走脚質数値'] = 10

df['is_escaper'] = (df['前走脚質数値'] <= 1).astype(int)
race_id_col = 'レースID(新)' if 'レースID(新)' in df.columns else 'レースID'
if race_id_col in df.columns:
    df['同レース逃げ馬数'] = df.groupby(race_id_col)['is_escaper'].transform('sum') - df['is_escaper']
else:
    df['同レース逃げ馬数'] = 0

df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)
df['騎手調教師コンビ'] = df['騎手コード'].astype(str) + "_" + df['調教師コード'].astype(str)

if '騎手コード' in df.columns and '前走騎手コード' in df.columns:
    df['騎手継続フラグ'] = (df['騎手コード'] == df['前走騎手コード']).astype(int)
else:
    df['騎手継続フラグ'] = 0

# --- Features ---
features = [
    '指数', '前走補正', 
    '前走PCI_val', '前走RPCI_val', '前走Ave3F', '同レース逃げ馬数', '前走脚質数値',
    '前走着順_num', '前走人気', '前走単勝オッズ', '前走上り3F', '前走着差タイム',
    '騎手継続フラグ', '騎手調教師コンビ', 'コースID',
    '斤量', '馬番', '馬体重', '馬体重増減', '年齢', '間隔', '種牡馬', '場所', '芝・ダ', '距離'
]
features = [f for f in features if f in df.columns]

# --- Encoding ---
categorical_cols = ['場所', '芝・ダ', '馬場状態', '種牡馬', '騎手コード', '調教師コード', 
                    '前走芝・ダ', 'コースID', '騎手調教師コンビ']

encoders = {} # 出馬表の予測で使うために辞書として保存
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna('unknown').astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

num_features = [f for f in features if f not in categorical_cols]
for col in num_features:
    if col in df.columns:
        temp_col = pd.to_numeric(df[col], errors='coerce')
        df[col] = temp_col.fillna(temp_col.mean())

# ==========================================
# 3. モデル学習 (高速化版)
# ==========================================
df['target_win'] = (df['着順_num'] == 1).astype(int)
X = df[features]
y = df['target_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n学習開始... (設定: 高速モード cv=3)")

# Base Model 
base_model = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=100,
    min_child_samples=50, 
    reg_alpha=0.1,
    n_jobs=-1  # ★CPUフル活用
)

# Calibrated Classifier (cv=3 に変更して高速化)
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# 重要度表示用
base_model.fit(X_train, y_train)

# ==========================================
# 4. 結果分析
# ==========================================
prob_win = calibrated_model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results['馬名'] = df.loc[X_test.index, '馬名']
results['着順'] = df.loc[X_test.index, '着順_num']
results['単勝オッズ'] = pd.to_numeric(df.loc[X_test.index, '単勝オッズ'], errors='coerce').fillna(0)
results['AI勝率予測(%)'] = (prob_win * 100).round(2)
results['期待値'] = (results['AI勝率予測(%)'] / 100) * results['単勝オッズ']

# 簡易診断
def make_comment(row):
    reasons = []
    if '騎手調教師コンビ' in df.columns:
         # コンビ相性が良いなどの判定ロジック（簡易）
         pass
    if row['指数'] > 110: reasons.append("高指数")
    if row['前走PCI_val'] >= 58: reasons.append("瞬発力◎")
    elif row['前走PCI_val'] <= 42: reasons.append("ハイペース向")
    if row['同レース逃げ馬数'] == 0 and row['前走脚質数値'] <= 2: reasons.append("単騎逃げ濃厚")
    if not reasons: return "-"
    return ",".join(reasons)

results['診断'] = results.apply(make_comment, axis=1)

auc = roc_auc_score(y_test, prob_win)
print(f"\nモデル精度(AUC): {auc:.4f}")

print("\n=== 【期待値ランキング】トップ15 (リアル確率版) ===")
display_cols = ['馬名', '着順', '単勝オッズ', 'AI勝率予測(%)', '期待値', '診断']
sorted_results = results.sort_values('期待値', ascending=False)
valid_results = sorted_results[
    (sorted_results['単勝オッズ'] > 0) & 
    (sorted_results['単勝オッズ'] < 300)
]
print(valid_results[display_cols].head(15))

print("\n=== 重要度ランキング ===")
importance = pd.DataFrame({'feature': features, 'importance': base_model.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(10))