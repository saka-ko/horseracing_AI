# ==========================================
# 🏇 競馬AI (過去3走Max評価 & 勝率表示版)
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

# ファイル設定
train_file = 'race_5years_zi_hoseitime_kai.csv'
entry_file = 'entry_table.csv'

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# ------------------------------------------------
# 1. 学習データの読み込み & 特徴量エンジニアリング
# ------------------------------------------------
print(f"🔄 学習データ({train_file})を読み込んでいます...")

# 読み込み
try:
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
    except:
        df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)

# 列名クリーニング
df_train.columns = df_train.columns.str.strip()
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# 着順の確保
if '着順' not in df_train.columns and '確定着順' in df_train.columns:
    df_train['着順'] = df_train['確定着順']

df_train['着順_num'] = df_train['着順'].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# --- ★重要: 過去3走の最大補正タイムを計算 ---
print("📊 過去5年分のレース履歴から、各馬の『過去3走MAX補正』を算出中...")

# 日付順に並べる
if '日付(yyyy.mm.dd)' in df_train.columns:
    df_train['date'] = pd.to_datetime(df_train['日付(yyyy.mm.dd)'], errors='coerce')
else:
    # 日付がない場合は並び順を信じるしかないが、通常はあるはず
    df_train['date'] = df_train.index

# 補正タイムを数値化
if '補正' in df_train.columns:
    df_train['補正_val'] = df_train['補正'].apply(force_numeric).fillna(0)
else:
    df_train['補正_val'] = 0

# 馬名と日付でソート
df_train = df_train.sort_values(['馬名', 'date'])

# 過去3走の最大値を取得 (シフトして過去を参照)
# shift(1)で「今回」を含めないようにし、rolling(3)で過去3つを見る
df_train['過去3走MAX補正'] = df_train.groupby('馬名')['補正_val'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).max()
).fillna(0)

# 指数 (ZI)
if '指数' not in df_train.columns:
    if 'ZI' in df_train.columns: df_train['指数'] = df_train['ZI']
    else: df_train['指数'] = 0
df_train['指数'] = df_train['指数'].apply(force_numeric).fillna(0)

# ランク計算
race_id_col = 'レースID(新)' if 'レースID(新)' in df_train.columns else 'レースID'
if race_id_col not in df_train.columns:
    # IDがない場合は日付と場所で代用
    if '日付(yyyy.mm.dd)' in df_train.columns and '場所' in df_train.columns:
         df_train['rid'] = df_train['日付(yyyy.mm.dd)'].astype(str) + df_train['場所']
         race_id_col = 'rid'
    else:
         race_id_col = None

if race_id_col:
    df_train['指数順位'] = df_train.groupby(race_id_col)['指数'].rank(ascending=False, method='min')
    # 過去3走MAXでの順位を計算
    df_train['補正順位'] = df_train.groupby(race_id_col)['過去3走MAX補正'].rank(ascending=False, method='min')
else:
    df_train['指数順位'] = 10; df_train['補正順位'] = 10

# 特徴量リスト
features = ['指数', '過去3走MAX補正', '指数順位', '補正順位']

# 学習実行
print("🔥 過去3走評価モデルを学習中...")
X = df_train[features]
y = df_train['target']

model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 2. 最新オッズでの予想 (出馬表の処理)
# ------------------------------------------------
print(f"🚀 最新の出馬表({entry_file})で予想します...")

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

# --- ★出馬表から過去3走のMAX補正を取得 ---
# 出馬表の列名 (補:1, 補:2, 補:3) を探す
hosei_cols = ['補:1', '補:2', '補:3']
target_hosei_cols = []

# 実際に存在する列だけ使う (補正タイム.1 などの場合も対応)
for c in hosei_cols:
    if c in df_pred.columns: target_hosei_cols.append(c)
if not target_hosei_cols:
    # 別名で探す
    for i in range(1, 4):
        c = f'補正タイム.{i}'
        if c in df_pred.columns: target_hosei_cols.append(c)

print(f"ℹ️ 参照する過去走データ: {target_hosei_cols}")

# 最大値を計算
def get_entry_max_hosei(row):
    vals = []
    for c in target_hosei_cols:
        v = force_numeric(row[c])
        if v > 0: vals.append(v)
    return max(vals) if vals else 0

df_pred['過去3走MAX補正'] = df_pred.apply(get_entry_max_hosei, axis=1)

# その他のマッピング
if 'ZI' in df_pred.columns: df_pred['指数'] = df_pred['ZI'].apply(force_numeric).fillna(0)
else: df_pred['指数'] = 0

# 単勝オッズ
odds_col = '単勝' if '単勝' in df_pred.columns else '単勝オッズ'
if odds_col in df_pred.columns:
    df_pred['単勝オッズ'] = df_pred[odds_col].apply(force_numeric).fillna(0)
else:
    df_pred['単勝オッズ'] = 0

# ランク計算
race_key = 'レース名' if 'レース名' in df_pred.columns else '開催'
if race_key not in df_pred.columns: df_pred['dummy']=1; race_key='dummy'

df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['過去3走MAX補正'].rank(ascending=False, method='min')

# 予測
X_pred = df_pred[features]
probs = calibrated_model.predict_proba(X_pred)[:, 1]
df_pred['AI勝率(%)'] = (probs * 100).round(2)
df_pred['期待値'] = (df_pred['AI勝率(%)'] / 100) * df_pred['単勝オッズ']

# 馬名
name_col = [c for c in df_pred.columns if '馬名' in c]
name_c = name_col[0] if name_col else 'Unknown'

# 診断
def make_comment(row):
    res = []
    if row['指数順位'] == 1: res.append("指数1位")
    if row['補正順位'] == 1: res.append("能力1位")
    elif row['補正順位'] <= 3: res.append("能力上位")
    if row['期待値'] >= 1.2: res.append("★狙い目")
    return ",".join(res) if res else "-"

df_pred['診断'] = df_pred.apply(make_comment, axis=1)

# --- 結果出力 ---
cols_out = ['枠番', '馬番', name_c, '単勝オッズ', 'AI勝率(%)', '期待値', '診断', '指数', '過去3走MAX補正']
disp_cols = [c for c in cols_out if c in df_pred.columns]

print("\n=== 💰 期待値ランキング (回収率重視) ===")
print(df_pred.sort_values('期待値', ascending=False)[disp_cols].head(15))

print("\n=== 🏅 AI勝率ランキング (的中率重視) ===")
print(df_pred.sort_values('AI勝率(%)', ascending=False)[disp_cols].head(15))