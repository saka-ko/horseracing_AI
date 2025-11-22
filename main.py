# ==========================================
# 🏇 競馬AI 最終版 (勝率表示 & 過去3走Max評価)
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.preprocessing import LabelEncoder

# ファイル設定
train_file = 'race_data_5years.csv' 
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
if '着順' not in df_train.columns:
    if '確定着順' in df_train.columns: df_train['着順'] = df_train['確定着順']

df_train['着順_num'] = df_train['着順'].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 特徴量マッピング (学習データは「前走」しかないためそのまま)
cols_map = {
    '前走PCI': ['前PCI', '前走PCI', 'PCI'],
    '前走RPCI': ['前RPCI', '前走RPCI', 'RPCI'],
    '前走Ave3F': ['前走Ave-3F', 'Ave-3F', 'Ave-3F.1'],
    '前走補正': ['補正タイム.1', '前走補9', '補正9']
}
for target, sources in cols_map.items():
    if target not in df_train.columns:
        for s in sources:
            if s in df_train.columns:
                df_train[target] = df_train[s]
                break
    # それでもなければ0埋め
    if target not in df_train.columns: df_train[target] = 0

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
# 2. 最新オッズでの予想 (過去3走評価機能付き)
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

# ★ここが新機能：過去3走からの「最大能力」抽出
# 出馬表にある '補:1'(1走前), '補:2'(2走前), '補:3'(3走前) を使います
# ※列名が '補正タイム.1' などの場合もあるので対応します
hosei_cols = ['補:1', '補:2', '補:3'] # CSVの列名を確認して調整
if '補:1' not in df_pred.columns:
    # '補正タイム'などの名前で入っている場合の予備リスト
    hosei_cols = ['補正タイム.1', '補正タイム.2', '補正タイム.3']

# 過去3走の最大値を計算する関数
def get_max_hosei(row):
    values = []
    for col in hosei_cols:
        if col in row.index:
            val = force_numeric(row[col])
            if val > 0: # 0や欠損は除外
                values.append(val)
    
    if not values: return 0 # データがない場合
    return max(values) # 最大値を返す

print("📊 過去3走の補正タイムから最大パフォーマンスを算出します...")
# 「前走補正」という項目に、あえて「過去3走の最大値」を入れることで
# 「この馬のベストパフォーマンス」をAIに評価させます
df_pred['前走補正'] = df_pred.apply(get_max_hosei, axis=1)

# その他のマッピング
rename_map = {
    '着順.1': '前走着順', '着差.1': '前走着差タイム',
    '上り3F.1': '前走上り3F', 
    'PCI.1': '前走PCI', 'PCI': '前走PCI',
    'Ave-3F.1': '前走Ave3F', '単勝': '単勝オッズ'
}
for k, v in rename_map.items():
    if k in df_pred.columns and v not in df_pred.columns:
        df_pred[v] = df_pred[k]

# 出馬表の特徴量作成
if '場所' not in df_pred.columns:
    if '開催' in df_pred.columns:
        place_map = {'札':'札幌', '函':'函館', '福':'福島', '新':'新潟', '東':'東京', '中':'中山', '京':'京都', '阪':'阪神', '小':'小倉'}
        df_pred['場所'] = df_pred['開催'].astype(str).apply(lambda x: place_map.get(x[1], 'その他') if len(x)>1 else 'その他')
    else: df_pred['場所'] = 'その他'
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
        # 学習データの平均値ではなく、0埋めの方が「データなし」を表現しやすい場合も
        df_pred[f] = df_pred[f].apply(force_numeric).fillna(0)
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
cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '前走補正']
disp_cols = [c for c in cols if c in df_pred.columns]

# 1. 期待値ランキング
print("\n=== 🎯 推奨馬リスト (期待値順) ===")
print("※『前走補正』欄は、過去3走のベスト数値を表示しています")
final_list_ev = df_pred[df_pred['単勝オッズ'] >= 1.0].sort_values('期待値', ascending=False)
print(final_list_ev[disp_cols].head(10))

# 2. 勝率ランキング (NEW!)
print("\n=== 🏆 推奨馬リスト (勝率順) ===")
print("※純粋な強さの評価順です")
final_list_prob = df_pred.sort_values('AI勝率(%)', ascending=False)
print(final_list_prob[disp_cols].head(10))

if len(final_list_ev) > 0:
    top_ev = final_list_ev.iloc[0]
    print(f"\n💰 期待値No.1: {top_ev[name_col]} (期待値 {top_ev['期待値']:.2f})")
if len(final_list_prob) > 0:
    top_prob = final_list_prob.iloc[0]
    print(f"👑 勝率No.1  : {top_prob[name_col]} (勝率 {top_prob['AI勝率(%)']}%)")