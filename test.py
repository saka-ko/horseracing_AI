# ==========================================
# 🧪 能力＆展開特化モデル バックテスト (5年データ版)
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1. データの読み込み
# ------------------------------------------
file_path = 'race_data_5years.csv' # ★ここを5年分のファイル名に

print(f"データを読み込んでいます... ({file_path})")
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df = pd.read_csv(file_path, encoding='cp932', low_memory=False)
    except:
        df = pd.read_csv(file_path, encoding='shift_jis', errors='ignore', low_memory=False)

# 2. データクリーニング & 特徴量作成
# ------------------------------------------
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

# 特徴量（特化型）
# 列名の揺らぎを吸収
if '前走PCI' not in df.columns and '前PCI' in df.columns: df['前走PCI'] = df['前PCI']
if '前走RPCI' not in df.columns and '前RPCI' in df.columns: df['前走RPCI'] = df['前RPCI']
if '前走Ave3F' not in df.columns and '前走Ave-3F' in df.columns: df['前走Ave3F'] = df['前走Ave-3F']

features = [
    '指数', '前走補正', 
    '前走着順', '前走着差タイム',
    '前走PCI', '前走RPCI', '前走Ave3F', '前走上り3F'
]

# 数値化 & 欠損埋め
df_model = pd.DataFrame()
df_model['target'] = df['target']
df_model['Odds'] = df['単勝オッズ'].apply(force_numeric).fillna(0)

for f in features:
    if f in df.columns:
        df_model[f] = df[f].apply(force_numeric).fillna(df[f].apply(force_numeric).mean())
    else:
        df_model[f] = 0 # ない場合は0

# 3. 学習 & 予測
# ------------------------------------------
X = df_model[features]
y = df_model['target']
odds = df_model['Odds']

# 時系列を意識して後半20%をテストにするのが理想ですが、今回はランダムで
X_train, X_test, y_train, y_test, odds_train, odds_test = train_test_split(
    X, y, odds, test_size=0.2, random_state=42
)

print("🔥 特化型モデルを学習中...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 予測
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\n✅ モデル精度(AUC): {auc:.4f}")

# 重要度
print("\n=== 📊 重要度ランキング (何を見て判断したか) ===")
imp = pd.DataFrame({'feature': features, 'gain': model.booster_.feature_importance(importance_type='gain')})
print(imp.sort_values('gain', ascending=False))

# 4. 💰 黄金の買い方 シミュレーション
# ------------------------------------------
print("\n=== 💰 回収率100%超え条件の探索 ===")
sim_df = pd.DataFrame({'target': y_test, 'prob': probs, 'odds': odds_test})
sim_df['ev'] = sim_df['prob'] * sim_df['odds']

# 条件総当たり
best_conds = []
for min_odds in [5.0, 10.0, 15.0, 20.0]: # 穴狙い
    for max_odds in [30.0, 50.0, 100.0]:
        if min_odds >= max_odds: continue
        for min_ev in [0.8, 1.0, 1.2, 1.5]:
            
            # 該当馬を抽出
            bets = sim_df[
                (sim_df['odds'] >= min_odds) & 
                (sim_df['odds'] < max_odds) & 
                (sim_df['ev'] >= min_ev)
            ]
            
            cnt = len(bets)
            if cnt < 50: continue # サンプル不足は除外
            
            hits = len(bets[bets['target'] == 1])
            ret = bets[bets['target'] == 1]['odds'].sum()
            rate = ret / cnt * 100
            
            if rate > 90: # 90%超えを表示
                best_conds.append({
                    '条件': f"オッズ{min_odds}-{max_odds}倍 & 期待値{min_ev}↑",
                    '件数': cnt,
                    '的中率': f"{hits/cnt*100:.1f}%",
                    '回収率': f"{rate:.1f}%"
                })

if best_conds:
    res_df = pd.DataFrame(best_conds)
    print(res_df.sort_values('回収率', ascending=False).head(10))
else:
    print("条件付きでも90%超えは見つかりませんでした。")