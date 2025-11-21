# ==========================================
# 🏇 競馬AI 最終完全版 (学習→予想を一気に実行)
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ファイル設定
train_file = 'race_data_5years.csv' 
entry_file = 'entry_table.csv'

# ------------------------------------------------
# 1. 学習データの読み込み & クリーニング
# ------------------------------------------------
print(f"🔄 学習データ({train_file})を読み込んでモデルを作成します...")

try:
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
    except:
        df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# 列名のクリーニング
df_train.columns = df_train.columns.str.strip()
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# ターゲット作成
df_train['着順_num'] = df_train['着順'].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 特徴量作成 (ZI抜き・ラップ特化)
# 列名の揺らぎ吸収
if '前走PCI' not in df_train.columns and '前PCI' in df_train.columns: df_train['前走PCI'] = df_train['前PCI']
if '前走RPCI' not in df_train.columns and '前RPCI' in df_train.columns: df_train['前走RPCI'] = df_train['前RPCI']
if '前走Ave3F' not in df_train.columns and '前走Ave-3F' in df_train.columns: df_train['前走Ave3F'] = df_train['前走Ave-3F']

# コースID
if '場所' not in df_train.columns: df_train['場所'] = 'その他'
if '芝・ダ' not in df_train.columns: df_train['芝・ダ'] = '芝'
if '距離' not in df_train.columns: df_train['距離'] = 1600
df_train['コースID'] = df_train['場所'].astype(str) + df_train['芝・ダ'].astype(str) + df_train['距離'].astype(str)

# 使用する特徴量 (ZIは除外)
features = [
    '前走補正',       # スピード
    '前走着順',       # 実績
    '前走着差タイム', # 能力差
    '前走PCI',        # ペース配分
    '前走RPCI',       # レースレベル
    '前走Ave3F',      # 追走力
    '前走上り3F',     # 瞬発力
    'コースID'        # 適性
]

# 学習用データ作成
for f in features:
    if f == 'コースID': continue
    if f in df_train.columns:
        df_train[f] = df_train[f].apply(force_numeric).fillna(0)
    else:
        df_train[f] = 0

# エンコーディング
le = LabelEncoder()
df_train['コースID'] = le.fit_transform(df_train['コースID'].astype(str))

print("🔥 特化型モデルを学習中...")
X = df_train[features]
y = df_train['target']
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 2. 出馬表の読み込み & 予想
# ------------------------------------------------
print(f"🚀 出馬表({entry_file})で予想を実行します...")

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

# マッピング (学習データの項目名に合わせる)
rename_map = {
    '補正タイム.1': '前走補正',
    '補正タイム': '前走補正',
    '着順.1': '前走着順',
    '着差.1': '前走着差タイム',
    '上り3F.1': '前走上り3F',
    'PCI.1': '前走PCI',
    'PCI': '前走PCI',
    'Ave-3F.1': '前走Ave3F',
    '単勝': '単勝オッズ'
}
# 存在する列だけリネーム
for k, v in rename_map.items():
    if k in df_pred.columns and v not in df_pred.columns:
        df_pred[v] = df_pred[k]

# 出馬表用の特徴量作成
# 開催地推定
if '場所' not in df_pred.columns:
    if '開催' in df_pred.columns:
        place_map = {'札':'札幌', '函':'函館', '福':'福島', '新':'新潟', '東':'東京', '中':'中山', '京':'京都', '阪':'阪神', '小':'小倉'}
        df_pred['場所'] = df_pred['開催'].astype(str).apply(lambda x: place_map.get(x[1], 'その他') if len(x)>1 else 'その他')
    else:
        df_pred['場所'] = 'その他'

if '芝・ダ' not in df_pred.columns: df_pred['芝・ダ'] = '芝' 
if '距離' not in df_pred.columns: df_pred['距離'] = 1600 

df_pred['コースID'] = df_pred['場所'].astype(str) + df_pred['芝・ダ'].astype(str) + df_pred['距離'].astype(str)

# エンコーディング適用
df_pred['コースID'] = df_pred['コースID'].apply(lambda x: x if x in le.classes_ else le.classes_[0])
df_pred['コースID'] = le.transform(df_pred['コースID'])

# 数値化 & 欠損埋め
for f in features:
    if f == 'コースID': continue
    if f in df_pred.columns:
        df_pred[f] = df_pred[f].apply(force_numeric).fillna(df_train[f].mean())
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
    # '  馬名' のようなスペース付きがあるか探す
    candidates = [c for c in df_pred.columns if '馬名' in c]
    if candidates: name_col = candidates[0]

# 結果表示
print("\n=== 🎯 能力＆ラップ特化AI 推奨馬リスト ===")
out_cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '前走補正', '前走着差タイム']
# ある列だけ表示
out_cols = [c for c in out_cols if c in df_pred.columns]

# 期待値順にソート (オッズ100倍以上は除外して表示)
valid_list = df_pred[
    (df_pred['単勝オッズ'] >= 1.0) & 
    (df_pred['単勝オッズ'] < 100.0)
].sort_values('期待値', ascending=False)

print(valid_list[out_cols].head(15))

if len(valid_list) > 0:
    top = valid_list.iloc[0]
    print(f"\n👑 AIの本命: {top[name_col]} (期待値: {top['期待値']:.0f})")