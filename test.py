# ==========================================
# 🧪 ZI抜き・特化モデル + 一点買いシミュレーション
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. データの読み込み
file_path = 'race_data_5years.csv'

print(f"データを読み込んでいます... ({file_path})")
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df = pd.read_csv(file_path, encoding='cp932', low_memory=False)
    except:
        df = pd.read_csv(file_path, encoding='shift_jis', errors='ignore', low_memory=False)

# 2. データクリーニング
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# ターゲット
df['着順_num'] = df['着順'].apply(force_numeric)
df = df.dropna(subset=['着順_num'])
df['target'] = (df['着順_num'] == 1).astype(int)

# レースIDの確保 (グループ化に必須)
race_id_col = 'レースID(新)' if 'レースID(新)' in df.columns else 'レースID'
# IDがない場合は日付と開催場所で仮IDを作る
if race_id_col not in df.columns:
    df['レースID'] = df['日付'].astype(str) + df['場所'].astype(str) + df['R'].astype(str)
    race_id_col = 'レースID'

# 特徴量（ZI抜き・能力＆ラップ特化）
features = [
    '前走補正', '前走着順_num', '前走着差タイム',
    '前走PCI_val', '前走RPCI_val', '前走Ave3F', '前走上り3F',
    'コースID'
]

# 列名の揺らぎ吸収 & 数値化
cols_map = {
    '前走着順': '前走着順_num', '着順.1': '前走着順_num',
    '前走着差': '前走着差タイム', '着差.1': '前走着差タイム',
    '前走補正': '前走補正', '補正タイム.1': '前走補正',
    '前PCI': '前走PCI_val', '前走PCI': '前走PCI_val', 'PCI.1': '前走PCI_val',
    '前RPCI': '前走RPCI_val', '前走RPCI': '前走RPCI_val', 'レースPCI.1': '前走RPCI_val',
    '前走Ave-3F': '前走Ave3F', 'Ave-3F.1': '前走Ave3F',
    '前走上り3F': '前走上り3F', '上り3F.1': '前走上り3F'
}

df_model = pd.DataFrame()
df_model['target'] = df['target']
df_model['Odds'] = df['単勝オッズ'].apply(force_numeric).fillna(0)
df_model['RaceID'] = df[race_id_col] # レースIDを保持

# コースID
if '場所' not in df.columns and '開催' in df.columns:
    place_map = {'札':'札幌', '函':'函館', '福':'福島', '新':'新潟', '東':'東京', '中':'中山', '京':'京都', '阪':'阪神', '小':'小倉'}
    df['場所'] = df['開催'].astype(str).apply(lambda x: place_map.get(x[1], 'その他') if len(x)>1 else 'その他')
if '場所' not in df.columns: df['場所'] = 'その他'
if '芝・ダ' not in df.columns: df['芝・ダ'] = '芝'
if '距離' not in df.columns: df['距離'] = 1600

df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)
le = LabelEncoder()
df_model['コースID'] = le.fit_transform(df['コースID'].astype(str))

for f in features:
    if f == 'コースID': continue
    # マッピング
    found = False
    for k, v in cols_map.items():
        if v == f and k in df.columns:
            df_model[f] = df[k].apply(force_numeric)
            found = True
            break
    if not found and f in df.columns:
        df_model[f] = df[f].apply(force_numeric)
    # 欠損埋め
    if f in df_model.columns:
        df_model[f] = df_model[f].fillna(df_model[f].mean())
    else:
        df_model[f] = 0

# 3. 学習 & 予測
X = df_model[features]
y = df_model['target']
ids = df_model['RaceID'] # IDも分割対象にする

# データを分割 (レースIDも一緒に分ける)
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, ids, test_size=0.2, random_state=42
)

print("🔥 特化型モデル(ZI抜き)を学習中...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 予測
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\n✅ モデル精度(AUC): {auc:.4f}")

# 4. 💰 レース内No.1戦略 シミュレーション
# ------------------------------------------
print("\n=== 💰 「レース内 期待値No.1」単勝一点買い シミュレーション ===")

# 結果をまとめる
sim_df = pd.DataFrame({
    'RaceID': ids_test,
    'target': y_test,
    'prob': probs,
    'odds': df_model.loc[X_test.index, 'Odds']
})
sim_df['ev'] = sim_df['prob'] * sim_df['odds'] # 期待値

# 各レースで「期待値」が最大の馬を取得
# idxmax()を使って、各グループ内でevが最大の行のインデックスを取得
idx_max = sim_df.groupby('RaceID')['ev'].idxmax()
top_picks = sim_df.loc[idx_max]

# シミュレーション実行関数
def simulate_strategy(picks_df, min_ev=0.0, min_odds=1.0):
    # 条件でフィルタリング
    # 1. 期待値が min_ev 以上 (低すぎる期待値の1位は買わない)
    # 2. オッズが min_odds 以上 (1.0倍などは買わない)
    bets = picks_df[
        (picks_df['ev'] >= min_ev) & 
        (picks_df['odds'] >= min_odds)
    ]
    
    cnt = len(bets)
    if cnt == 0: return 0, 0, 0, 0
    
    hits = len(bets[bets['target'] == 1])
    return_amount = bets[bets['target'] == 1]['odds'].sum() * 100
    invest_amount = cnt * 100
    
    rate = (return_amount / invest_amount) * 100
    profit = return_amount - invest_amount
    return cnt, hits, rate, profit

# いろいろな条件で試す
conditions = [
    (0.0, 1.0, "条件なし (全レース購入)"),
    (1.0, 1.0, "期待値1.0以上 (ボーダー超えのみ)"),
    (1.2, 1.0, "期待値1.2以上 (厳選)"),
    (1.0, 5.0, "期待値1.0以上 & 単勝5倍以上 (穴狙い)"),
    (1.2, 10.0, "期待値1.2以上 & 単勝10倍以上 (大穴厳選)")
]

print(f"{'条 件':<25} | {'購入数':<6} | {'的中率':<6} | {'回収率':<6} | {'収支'}")
print("-" * 70)

for min_ev, min_odds, label in conditions:
    cnt, hits, rate, profit = simulate_strategy(top_picks, min_ev, min_odds)
    print(f"{label:<25} | {cnt:>6} | {hits/cnt*100:>5.1f}% | {rate:>5.1f}% | {profit:>+8.0f}円")

# 最も良かった条件の詳細
best_bets = top_picks[top_picks['ev'] >= 1.0] # デフォルトは1.0以上
print("\n--- 参考: 期待値1.0以上の分布 ---")
print(f"平均オッズ: {best_bets['odds'].mean():.1f}倍")
print(f"平均勝率  : {best_bets['prob'].mean()*100:.1f}%")