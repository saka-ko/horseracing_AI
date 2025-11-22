# ==========================================
# 🏇 競馬AI (ZI & 断層理論) - 決定版
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------
# 0. 設定
# ------------------------------------------------
train_file = 'race_5years_zi_hoseitime_kai.csv' 
entry_file = 'entry_table.csv'      

if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
    entry_file = sys.argv[1]

# ------------------------------------------------
# 1. 学習データの読み込み
# ------------------------------------------------
print(f"🔄 学習データ({train_file})を読み込んでいます...")

try:
    df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
except:
    df_train = pd.read_csv(train_file, encoding='utf-8', low_memory=False)

df_train.columns = df_train.columns.str.strip()

# --- 列名マッピング ---
col_map = {}
aliases = {
    '着順': ['確定着順', '着順'],
    'ZI': ['指数', 'ZI', 'ZI値'],
    'オッズ': ['単勝オッズ', '単勝', '確定単勝オッズ'],
    'レースID': ['レースID(新)', 'レースID(旧)', 'レースID'],
    '前走補正': ['前走補9', '前走補正', '前走タイム'] 
}

for key, candidates in aliases.items():
    for cand in candidates:
        if cand in df_train.columns:
            col_map[key] = cand
            break

if '着順' not in col_map or 'ZI' not in col_map:
    print(f"❌ エラー: 必要な列が見つかりません。現在の列名: {list(df_train.columns)}")
    sys.exit(1)

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        import re
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# データ整形
df_train['target'] = (df_train[col_map['着順']].apply(force_numeric) == 1).astype(int)
df_train['指数'] = df_train[col_map['ZI']].apply(force_numeric).fillna(0)
df_train['単勝オッズ'] = df_train[col_map['オッズ']].apply(force_numeric).fillna(0)

if '前走補正' in col_map:
    df_train['前走補正'] = df_train[col_map['前走補正']].apply(force_numeric).fillna(0)
else:
    df_train['前走補正'] = 0

# レースID修正（馬番カット）
rid_col = col_map['レースID']
df_train['rid_str'] = df_train[rid_col].astype(str)
if len(df_train) / df_train['rid_str'].nunique() < 5.0:
    df_train['rid_group'] = df_train['rid_str'].str[:-2]
else:
    df_train['rid_group'] = df_train['rid_str']

# ランク計算
df_train['指数順位'] = df_train.groupby('rid_group')['指数'].rank(ascending=False, method='min')
df_train['補正順位'] = df_train.groupby('rid_group')['前走補正'].rank(ascending=False, method='min')

features = ['指数', '前走補正', '指数順位', '補正順位']
X = df_train[features]
y = df_train['target']

# ------------------------------------------------
# 2. モデル検証（正しい分割方法）
# ------------------------------------------------
print("\n📊 モデル検証中（レース単位で正しく分割）...")

# ★重要: レースID単位で分割する（バラバラ殺人を防ぐ）
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, y, groups=df_train['rid_group']))

X_train = X.iloc[train_idx]
y_train = y.iloc[train_idx]
X_val = X.iloc[val_idx]
y_val = y.iloc[val_idx]

# 検証用DF
df_val_sim = df_train.iloc[val_idx].copy()

# 学習
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# 予測
probs = calibrated_model.predict_proba(X_val)[:, 1]
df_val_sim['prob'] = probs
df_val_sim['expected_value'] = df_val_sim['prob'] * df_val_sim['単勝オッズ']

# --- 🦁 オッズ断層の計算 ---
df_val_sim = df_val_sim.sort_values(by=['rid_group', '単勝オッズ'])
df_val_sim['next_odds'] = df_val_sim.groupby('rid_group')['単勝オッズ'].shift(-1)
df_val_sim['gap_next'] = df_val_sim['next_odds'] / df_val_sim['単勝オッズ']
df_val_sim['gap_next'] = df_val_sim['gap_next'].fillna(1.0)

# === 🧪 シミュレーション条件 ===
# 1. ベースライン: ZI指数1位を買い続ける
cond_zi = df_val_sim['指数順位'] == 1

# 2. AI本命: AIの確率が最も高い馬を買う (これが真の実力)
# 各レースでprobが最大の行にフラグを立てる
idx_max_prob = df_val_sim.groupby('rid_group')['prob'].idxmax()
cond_ai_top = df_val_sim.index.isin(idx_max_prob)

# 3. AI期待値 + 確率フィルター (大穴事故防止)
# 期待値100円以上、かつ勝率が10%以上ある現実的な馬
cond_ai_ev = (df_val_sim['expected_value'] >= 1.0) & (df_val_sim['prob'] >= 0.10)

# 4. 断層理論 (勝率10%以上 + 期待値100円 + 後ろに断層)
cond_gap = (df_val_sim['expected_value'] >= 1.0) & \
           (df_val_sim['prob'] >= 0.10) & \
           (df_val_sim['gap_next'] >= 1.5)

def report_sim(name, condition):
    picks = df_val_sim[condition]
    if len(picks) == 0:
        print(f"  [{name}] 該当なし")
        return
    hits = picks[picks['target'] == 1]
    accuracy = len(hits) / len(picks) * 100
    return_rate = hits['単勝オッズ'].sum() / len(picks) * 100
    avg_odds = picks['単勝オッズ'].mean()
    print(f"  [{name}]")
    print(f"    購入: {len(picks)}R / 平均オッズ: {avg_odds:.1f}倍")
    print(f"    🎯 的中率: {accuracy:.2f}%")
    print(f"    💰 回収率: {return_rate:.2f}%")

print(f"--- 🏁 検証結果 (テスト期間のシミュレーション) ---")
report_sim("ベースライン: ZI順位1位ベタ買い", cond_zi)
print("-" * 40)
report_sim("プランA: AI自信度No.1 (本命・対抗)", cond_ai_top)
report_sim("プランB: AI期待値狙い (勝率10%以上限定)", cond_ai_ev)
print("-" * 40)
report_sim("プランC: AI + オッズ断層 (勝負レース)", cond_gap)
print(f"--------------------------------------------------")

# 本番再学習
print("🔄 本番用に全データで再学習しています...")
calibrated_model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 3. 最新オッズでの予想 (断層診断機能付き)
# ------------------------------------------------
print(f"\n🚀 出馬表({entry_file})で予想します...")
# (以下、予想パートは変更なしのためそのまま使えます)
if not os.path.exists(entry_file):
    print(f"❌ エラー: 予想用ファイル({entry_file})が見つかりません。")
    sys.exit(1)

try:
    df_entry = pd.read_csv(entry_file, encoding='utf-8-sig')
except:
    try:
        df_entry = pd.read_csv(entry_file, encoding='cp932')
    except:
        df_entry = pd.read_csv(entry_file, encoding='shift_jis', errors='replace')

df_entry.columns = df_entry.columns.str.strip()
df_pred = df_entry.copy()

# 過去3走から最大補正タイムを取得
hosei_cols = []
for i in range(1, 4):
    c1 = f'補:{i}'; c2 = f'補正タイム.{i}'
    if c1 in df_pred.columns: hosei_cols.append(c1)
    elif c2 in df_pred.columns: hosei_cols.append(c2)
if '補正タイム' in df_pred.columns: hosei_cols.append('補正タイム')

def get_max_hosei(row):
    vals = []
    for c in hosei_cols:
        v = force_numeric(row[c])
        if v > 0: vals.append(v)
    return max(vals) if vals else 0

df_pred['前走補正'] = df_pred.apply(get_max_hosei, axis=1)

if 'ZI' in df_pred.columns: df_pred['指数'] = df_pred['ZI'].apply(force_numeric).fillna(0)
else: df_pred['指数'] = 0

odds_col_entry = None
for c in ['単勝', '単勝オッズ', '予想単勝オッズ']:
    if c in df_pred.columns: odds_col_entry = c; break
df_pred['単勝オッズ'] = df_pred[odds_col_entry].apply(force_numeric).fillna(0) if odds_col_entry else 0

race_key = 'レース名' if 'レース名' in df_pred.columns else 'dummy'
if race_key == 'dummy': df_pred['dummy'] = 1

df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['前走補正'].rank(ascending=False, method='min')

X_pred = df_pred[features]
raw_probs = calibrated_model.predict_proba(X_pred)[:, 1]

total_prob = raw_probs.sum()
normalized_probs = raw_probs / total_prob if total_prob > 0 else raw_probs

df_pred['AI勝率(%)'] = (normalized_probs * 100).round(2)
df_pred['期待値'] = (normalized_probs * df_pred['単勝オッズ'])

# 馬名取得
name_col = '馬名' if '馬名' in df_pred.columns else df_pred.columns[0]

# 診断コメント
def make_comment(row):
    res = []
    if row['指数順位'] == 1: res.append("指数1位")
    if row['補正順位'] == 1: res.append("能力1位")
    if row['期待値'] >= 1.0: res.append("★推奨")
    return ",".join(res) if res else "-"
df_pred['診断'] = df_pred.apply(make_comment, axis=1)

# オッズ断層分析
def analyze_odds_gap(df_race):
    df_sorted = df_race[df_race['単勝オッズ'] > 0].sort_values('単勝オッズ')
    if len(df_sorted) < 6: return "⚠️ データ不足", []

    odds = df_sorted['単勝オッズ'].values
    gaps = odds[1:] / odds[:-1]
    
    diagnosis = []
    target_horse_indices = [] 

    if gaps[0] >= 2.5: diagnosis.append(f"🦁 1番人気鉄板(断層{gaps[0]:.1f})")
    elif gaps[0] < 1.5: diagnosis.append(f"⚠️ 1番人気危険(断層{gaps[0]:.1f})")

    middle_gaps = gaps[1:5] 
    if len(middle_gaps) > 0:
        max_gap_idx = np.argmax(middle_gaps) + 1 
        max_gap_val = middle_gaps[np.argmax(middle_gaps)]
        if max_gap_val >= 2.0:
            target_pop = max_gap_idx + 1
            target_name = df_sorted.iloc[max_gap_idx][name_col]
            diagnosis.append(f"💰 {target_pop}番人気({target_name})狙い目(後続と断層{max_gap_val:.1f})")
            target_horse_indices.append(max_gap_idx)
    
    if all(g < 1.5 for g in gaps[:5]):
        diagnosis.append("💤 混戦スルー推奨")

    return " / ".join(diagnosis), df_sorted.iloc[target_horse_indices][name_col].tolist()

cols_out = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '診断', '指数', '前走補正']
disp_cols = [c for c in cols_out if c in df_pred.columns]

print("\n=== 📊 オッズ断層分析 (マーケット心理) ===")
gap_msg, gap_targets = analyze_odds_gap(df_pred)
print(f"💬 {gap_msg}")
if gap_targets:
    print(f"👉 断層理論の注目馬: {', '.join(gap_targets)}")

print("\n=== 💰 期待値ランキング ===")
print(df_pred[df_pred['単勝オッズ'] >= 1.0].sort_values('期待値', ascending=False)[disp_cols].head(15))