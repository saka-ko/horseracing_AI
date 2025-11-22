# ==========================================
# 🏁 FINAL ANSWER: 学習 & 最新オッズ予想 (出力整形版)
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.preprocessing import LabelEncoder

# ファイル設定
train_file = 'race_5years_zi_hoseitime_kai.csv' 
entry_file = 'entry_table.csv'

print(f"🔄 学習データ({train_file})を読み込んでモデルを構築します...")

# 1. 学習データの読み込み
try:
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
    except:
        df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)

# データクリーニング関数
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

# ターゲット作成
df_train['着順_num'] = df_train['着順'].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 特徴量マッピング (ZI抜き・能力特化)
cols_map = {
    '前走PCI': ['前PCI', '前走PCI', 'PCI'],
    '前走RPCI': ['前RPCI', '前走RPCI', 'RPCI'],
    '前走Ave3F': ['前走Ave-3F', 'Ave-3F', 'Ave-3F.1']
}
for target, sources in cols_map.items():
    if target not in df_train.columns:
        for s in sources:
            if s in df_train.columns:
                df_train[target] = df_train[s]
                break

# コースID
if '場所' not in df_train.columns: df_train['場所'] = 'その他'
if '芝・ダ' not in df_train.columns: df_train['芝・ダ'] = '芝'
if '距離' not in df_train.columns: df_train['距離'] = 1600
df_train['コースID'] = df_train['場所'].astype(str) + df_train['芝・ダ'].astype(str) + df_train['距離'].astype(str)

# 使用する特徴量
features = [
    '前走補正', '前走着順', '前走着差タイム',
    '前走PCI', '前走RPCI', '前走Ave3F', '前走上り3F',
    'コースID'
]

# 数値化 & 欠損埋め
for f in features:
    if f == 'コースID': continue
    if f in df_train.columns:
        df_train[f] = df_train[f].apply(force_numeric).fillna(0)
    else:
        df_train[f] = 0

# エンコーディング
le = LabelEncoder()
df_train['コースID'] = le.fit_transform(df_train['コースID'].astype(str))

# 学習実行
print("🔥 能力特化モデルを学習中...")
X = df_train[features]
y = df_train['target']
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X, y)
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

# マッピング (学習データに合わせる)
rename_map = {
    '補正タイム.1': '前走補正', '補正タイム': '前走補正',
    '着順.1': '前走着順', '着差.1': '前走着差タイム',
    '上り3F.1': '前走上り3F', 
    'PCI.1': '前走PCI', 'PCI': '前走PCI',
    'Ave-3F.1': '前走Ave3F', '単勝': '単勝オッズ'
}
for k, v in rename_map.items():
    if k in df_pred.columns and v not in df_pred.columns:
        df_pred[v] = df_pred[k]

# 出馬表の特徴量作成
# 開催地推定
if '場所' not in df_pred.columns:
    if '開催' in df_pred.columns:
        place_map = {'札':'札幌', '函':'函館', '福':'福島', '新':'新潟', '東':'東京', '中':'中山', '京':'京都', '阪':'阪神', '小':'小倉'}
        df_pred['場所'] = df_pred['開催'].astype(str).apply(lambda x: place_map.get(x[1], 'その他') if len(x)>1 else 'その他')
    else: df_pred['場所'] = 'その他'
if '芝・ダ' not in df_pred.columns: df_pred['芝・ダ'] = '芝' 
if '距離' not in df_pred.columns: df_pred['距離'] = 1600 

df_pred['コースID'] = df_pred['場所'].astype(str) + df_pred['芝・ダ'].astype(str) + df_pred['距離'].astype(str)

# エンコーディング適用 (未知の値対策)
df_pred['コースID'] = df_pred['コースID'].apply(lambda x: x if x in le.classes_ else le.classes_[0])
df_pred['コースID'] = le.transform(df_pred['コースID'])

# 数値化 & 欠損埋め
for f in features:
    if f == 'コースID': continue
    if f in df_pred.columns:
        # 学習データの平均値で埋める(より安全)
        mean_val = df_train[f].mean()
        df_pred[f] = df_pred[f].apply(force_numeric).fillna(mean_val)
    else:
        df_pred[f] = 0

# 予測実行
X_pred = df_pred[features]
probs = model.predict_proba(X_pred)[:, 1]
df_pred['AI勝率(%)'] = (probs * 100).round(1)

# 期待値計算
if '単勝オッズ' in df_pred.columns:
    df_pred['単勝オッズ'] = df_pred['単勝オッズ'].apply(force_numeric).fillna(0)
    df_pred['期待値'] = (df_pred['AI勝率(%)'] / 100) * df_pred['単勝オッズ']
else:
    df_pred['単勝オッズ'] = 0
    df_pred['期待値'] = 0

# 馬名の取得
name_col = '馬名'
if '馬名' not in df_pred.columns:
    cands = [c for c in df_pred.columns if '馬名' in c]
    if cands: name_col = cands[0]

# --- 結果表示 ---
print("\n=== 🎯 最新オッズ反映：推奨馬リスト ===")
cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '前走補正']
disp_cols = [c for c in cols if c in df_pred.columns]

# オッズ1.0倍以上でソート
final_list = df_pred[df_pred['単勝オッズ'] >= 1.0].sort_values('期待値', ascending=False)

print(final_list[disp_cols].head(15).to_markdown(index=False)) # レポート形式で見やすく表示

if len(final_list) > 0:
    top = final_list.iloc[0]
    print(f"\n👑 最終本命: {top[name_col]} (期待値: {top['期待値']:.2f})")