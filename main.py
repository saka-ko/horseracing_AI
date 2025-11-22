# ==========================================
# 🧪 ZI & 補正タイム 一点突破モデル
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

# 1. データの読み込み
# ------------------------------------------
train_file = 'race_data_5years.csv'
entry_file = 'entry_table.csv'

print(f"🔄 学習データ({train_file})を読み込んでいます...")
try:
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
    except:
        df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)

# 2. データクリーニング & 特徴量作成
# ------------------------------------------
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# 列名クリーニング
df_train.columns = df_train.columns.str.strip()
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# ターゲット
df_train['着順_num'] = df_train['着順'].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 必須列の確保（列名揺らぎ吸収）
if '前走補正' not in df_train.columns and '補正タイム.1' in df_train.columns:
    df_train['前走補正'] = df_train['補正タイム.1']
if '指数' not in df_train.columns and 'ZI' in df_train.columns:
    df_train['指数'] = df_train['ZI']

# 特徴量：今回は「指数」と「補正」のみ！
# ただし、「絶対値」と「順位(相対評価)」の両方を見せます
features = ['指数', '前走補正', '指数順位', '補正順位']

# 数値化
for f in ['指数', '前走補正']:
    if f in df_train.columns:
        df_train[f] = df_train[f].apply(force_numeric).fillna(0)
    else:
        df_train[f] = 0

# ランク計算
race_id_col = 'レースID(新)' if 'レースID(新)' in df_train.columns else 'レースID'
if race_id_col in df_train.columns:
    df_train['指数順位'] = df_train.groupby(race_id_col)['指数'].rank(ascending=False, method='min')
    df_train['補正順位'] = df_train.groupby(race_id_col)['前走補正'].rank(ascending=False, method='min')
else:
    df_train['指数順位'] = 10; df_train['補正順位'] = 10

# 3. 学習実行
# ------------------------------------------
X = df_train[features]
y = df_train['target']

# データを分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🔥 ZI & 補正タイム特化モデルを学習中...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 重要度の確認
print("\n=== 📊 どっちが重要？ ===")
imp = pd.DataFrame({'feature': features, 'gain': model.booster_.feature_importance(importance_type='gain')})
print(imp.sort_values('gain', ascending=False))

# 4. 最新出馬表での予想
# ------------------------------------------
print(f"\n🚀 出馬表({entry_file})で予想します...")
try:
    df_entry = pd.read_csv(entry_file, encoding='utf-8-sig')
except:
    try:
        df_entry = pd.read_csv(entry_file, encoding='cp932')
    except:
        df_entry = pd.read_csv(entry_file, encoding='shift_jis', errors='replace')

df_entry.columns = df_entry.columns.str.strip()
df_entry = df_entry.loc[:, ~df_entry.columns.duplicated()]
df_pred = df_entry.copy()

# マッピング
rename_map = {
    'ZI': '指数',
    '補正タイム.1': '前走補正', '補正タイム': '前走補正',
    '単勝': '単勝オッズ'
}
for k, v in rename_map.items():
    if k in df_pred.columns and v not in df_pred.columns:
        df_pred[v] = df_pred[k]

# 数値化
for f in ['指数', '前走補正', '単勝オッズ']:
    if f in df_pred.columns:
        df_pred[f] = df_pred[f].apply(force_numeric).fillna(0)
    else:
        df_pred[f] = 0

# ランク計算
race_key = 'レース名' if 'レース名' in df_pred.columns else '開催'
df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['前走補正'].rank(ascending=False, method='min')

# 予測
X_pred = df_pred[features]
probs = model.predict_proba(X_pred)[:, 1]
df_pred['AI勝率(%)'] = (probs * 100).round(1)
df_pred['期待値'] = (df_pred['AI勝率(%)'] / 100) * df_pred['単勝オッズ']

# 結果表示
name_col = '馬名'
if '馬名' not in df_pred.columns:
    cands = [c for c in df_pred.columns if '馬名' in c]
    if cands: name_col = cands[0]

print("\n=== 🎯 シンプルイズベスト推奨馬 (指数＆補正のみ) ===")
out_cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '指数', '前走補正']
valid_list = df_pred[
    (df_pred['単勝オッズ'] >= 1.0) & (df_pred['単勝オッズ'] < 100)
].sort_values('期待値', ascending=False)

print(valid_list[out_cols].head(15))