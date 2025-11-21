# ==========================================
# 🧪 ZI抜き・能力特化モデル 検証用コード
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# 1. データの読み込み
# ------------------------------------------
file_path = 'race_data_5years.csv' # 5年分データ

print(f"データを読み込んでいます... ({file_path})")
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df = pd.read_csv(file_path, encoding='cp932', low_memory=False)
    except:
        df = pd.read_csv(file_path, encoding='shift_jis', errors='ignore', low_memory=False)

# 2. データクリーニング
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

# 列名の揺らぎ吸収 & 数値化
cols_map = {
    '前走着順': '前走着順_num',
    '着順.1': '前走着順_num',
    '前走着差タイム': '前走着差',
    '着差.1': '前走着差',
    '前走補正': '前走補正',
    '補正タイム.1': '前走補正',
    '前PCI': '前走PCI',
    '前走PCI': '前走PCI',
    '前走RPCI': '前走RPCI',
    '前走Ave-3F': '前走Ave3F',
    '前走上り3F': '前走上り3F'
}

# データの整理
df_model = pd.DataFrame()
df_model['target'] = df['target']
df_model['Odds'] = df['単勝オッズ'].apply(force_numeric).fillna(0)

# コースID作成
if '場所' not in df.columns and '開催' in df.columns:
    place_map = {'札':'札幌', '函':'函館', '福':'福島', '新':'新潟', '東':'東京', '中':'中山', '京':'京都', '阪':'阪神', '小':'小倉'}
    df['場所'] = df['開催'].astype(str).apply(lambda x: place_map.get(x[1], 'その他') if len(x)>1 else 'その他')
if '場所' not in df.columns: df['場所'] = 'その他'
if '芝・ダ' not in df.columns: df['芝・ダ'] = '芝'
if '距離' not in df.columns: df['距離'] = 1600

df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)
le = LabelEncoder()
df_model['コースID'] = le.fit_transform(df['コースID'].astype(str))

# 特徴量の取り込み (ZIは除外)
features = [
    '前走補正',       # スピード指数
    '前走着順_num',   # 着順
    '前走着差',       # タイム差
    '前走PCI',        # ラップバランス
    '前走RPCI',       # レースペース
    '前走Ave3F',      # スピード
    '前走上り3F',     # 末脚
    'コースID'        # 適性
]

for f in features:
    # マッピング対応
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
        df_model[f] = 0 # なければ0

# 3. 学習 & 予測
# ------------------------------------------
X = df_model[features]
y = df_model['target']
odds = df_model['Odds']

X_train, X_test, y_train, y_test, odds_train, odds_test = train_test_split(
    X, y, odds, test_size=0.2, random_state=42
)

print("🔥 ZI抜き・ラップ特化モデルを学習中...")
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# 予測
probs = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
print(f"\n✅ モデル精度(AUC): {auc:.4f}")

# 重要度
print("\n=== 📊 重要度ランキング (ZIなし) ===")
imp = pd.DataFrame({'feature': features, 'gain': model.booster_.feature_importance(importance_type='gain')})
print(imp.sort_values('gain', ascending=False))

# 4. 💰 黄金の買い方 シミュレーション
# ------------------------------------------
print("\n=== 💰 回収率100%超え条件の探索 ===")
sim_df = pd.DataFrame({'target': y_test, 'prob': probs, 'odds': odds_test})
sim_df['ev'] = sim_df['prob'] * sim_df['odds']

best_conds = []
# 条件総当たり (少し範囲を広げます)
for min_odds in [5.0, 10.0, 15.0, 20.0, 30.0]:
    for max_odds in [50.0, 100.0, 150.0]:
        if min_odds >= max_odds: continue
        for min_ev in [0.8, 1.0, 1.2, 1.5, 2.0]:
            
            bets = sim_df[
                (sim_df['odds'] >= min_odds) & 
                (sim_df['odds'] < max_odds) & 
                (sim_df['ev'] >= min_ev)
            ]
            
            cnt = len(bets)
            if cnt < 30: continue # サンプル少なすぎは除外
            
            hits = len(bets[bets['target'] == 1])
            ret = bets[bets['target'] == 1]['odds'].sum()
            rate = ret / cnt * 100
            
            if rate > 100:
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
    print("条件付きでも100%超えは見つかりませんでした。")