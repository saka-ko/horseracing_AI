# ==========================================
# 🏇 競馬AI (ZI & 補正タイム特化型)
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

print(f"🔄 学習データ({train_file})を読み込んでモデルを作成します...")

# 1. 学習データの読み込み
try:
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
    except:
        df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)

# 列名のクリーニング
df_train.columns = df_train.columns.str.strip()
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# ターゲット作成 ('確定着順' を '着順' として扱う)
rank_col = '確定着順' if '確定着順' in df_train.columns else '着順'
if rank_col not in df_train.columns:
    print("⚠️ 着順列が見つかりません。")
    # 簡易的に探す
    cands = [c for c in df_train.columns if '着順' in c]
    if cands: rank_col = cands[0]

df_train['着順_num'] = df_train[rank_col].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 特徴量作成
# 必須列の確認
if '指数' not in df_train.columns: df_train['指数'] = 0
if '前走補正' not in df_train.columns: 
    if '前走補9' in df_train.columns: df_train['前走補正'] = df_train['前走補9']
    else: df_train['前走補正'] = 0

# 数値化 & 欠損埋め
for f in ['指数', '前走補正']:
    df_train[f] = df_train[f].apply(force_numeric).fillna(0)

# ランク計算 (レース内順位)
race_id_col = 'レースID(新)' if 'レースID(新)' in df_train.columns else 'レースID'
if race_id_col not in df_train.columns:
    # IDがない場合、日付と場所で仮ID作成
    if '日付(yyyy.mm.dd)' in df_train.columns and '場所' in df_train.columns:
        df_train['ID'] = df_train['日付(yyyy.mm.dd)'].astype(str) + df_train['場所'].astype(str) + df_train['Ｒ'].astype(str)
        race_id_col = 'ID'
    else:
        race_id_col = None

if race_id_col:
    df_train['指数順位'] = df_train.groupby(race_id_col)['指数'].rank(ascending=False, method='min')
    df_train['補正順位'] = df_train.groupby(race_id_col)['前走補正'].rank(ascending=False, method='min')
else:
    df_train['指数順位'] = 10; df_train['補正順位'] = 10

# ★使用する特徴量はこれだけ！
features = ['指数', '前走補正', '指数順位', '補正順位']

print("🔥 ZI & 補正タイム特化モデルを学習中...")
X = df_train[features]
y = df_train['target']

# モデル学習
base_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 2. 最新オッズでの予想
# ------------------------------------------------
print(f"🚀 最新の出馬表({entry_file})で予想します...")

try:
    df_entry = pd.read_csv(entry_file, encoding='utf-8-sig')
except:
    try:
        df_entry = pd.read_csv(entry_file, encoding='cp932')
    except:
        df_entry = pd.read_csv(entry_file, encoding='shift_jis', errors='replace')

# 列名クリーニング
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
# レース名ごとに順位を出す
race_key = 'レース名' 
if race_key not in df_pred.columns:
    # なければ全て1レースとみなす
    df_pred['dummy'] = 1
    race_key = 'dummy'

df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['前走補正'].rank(ascending=False, method='min')

# 予測実行
X_pred = df_pred[features]
probs = calibrated_model.predict_proba(X_pred)[:, 1]
df_pred['AI勝率(%)'] = (probs * 100).round(1)
df_pred['期待値'] = (df_pred['AI勝率(%)'] / 100) * df_pred['単勝オッズ']

# 馬名取得
name_col = '馬名'
if '馬名' not in df_pred.columns:
    cands = [c for c in df_pred.columns if '馬名' in c]
    if cands: name_col = cands[0]

# 診断コメント
def make_comment(row):
    res = []
    if row['指数順位'] == 1: res.append("指数1位")
    if row['補正順位'] <= 3: res.append("前走Hレベル")
    if row['期待値'] >= 1.0: res.append("★推奨")
    return ",".join(res) if res else "-"

df_pred['診断'] = df_pred.apply(make_comment, axis=1)

# 結果表示
print("\n=== 🎯 シンプルAI (指数＆補正) 推奨馬リスト ===")
cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '診断', '指数', '前走補正']
disp_cols = [c for c in cols if c in df_pred.columns]

# オッズ100倍未満でソート
final_list = df_pred[
    (df_pred['単勝オッズ'] >= 1.0) & 
    (df_pred['単勝オッズ'] < 100.0)
].sort_values('期待値', ascending=False)

print(final_list[disp_cols].head(15))

if len(final_list) > 0:
    top = final_list.iloc[0]
    print(f"\n👑 最終本命: {top[name_col]} (期待値: {top['期待値']:.2f})")