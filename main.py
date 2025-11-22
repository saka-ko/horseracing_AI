# ==========================================
# 🏇 競馬AI (ZI & 断層 & 特徴量強化版)
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------
# 0. 設定
# ------------------------------------------------
train_file = 'race_5years_zi_hoseitime_kai.csv' 
entry_file = 'entry_table.csv'      

if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
    entry_file = sys.argv[1]

# ------------------------------------------------
# 1. 学習データの読み込み & クリーニング
# ------------------------------------------------
print(f"🔄 学習データ({train_file})を読み込んでいます...")

try:
    df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
except:
    df_train = pd.read_csv(train_file, encoding='utf-8', low_memory=False)

df_train.columns = df_train.columns.str.strip()

# 列名マッピング
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
    print(f"❌ エラー: 必要な列が見つかりません。")
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

df_train['target'] = (df_train[col_map['着順']].apply(force_numeric) == 1).astype(int)
df_train['指数'] = df_train[col_map['ZI']].apply(force_numeric).fillna(0)
df_train['単勝オッズ'] = df_train[col_map['オッズ']].apply(force_numeric).fillna(0)

if '前走補正' in col_map:
    df_train['前走補正'] = df_train[col_map['前走補正']].apply(force_numeric).fillna(0)
else:
    df_train['前走補正'] = 0

# ZI=0の除外 (品質向上のカギ)
df_train = df_train[df_train['指数'] > 0]

# レースID修正
rid_col = col_map['レースID']
df_train['rid_str'] = df_train[rid_col].astype(str)
df_train['rid_group'] = df_train['rid_str'].str[:-2]

# ランク計算
df_train['指数順位'] = df_train.groupby('rid_group')['指数'].rank(ascending=False, method='min')
df_train['補正順位'] = df_train.groupby('rid_group')['前走補正'].rank(ascending=False, method='min')

# --- ★特徴量エンジニアリング (ここが進化) ---
# カテゴリ変数のエンコーディング用辞書
encoders = {}

def get_encoder(col_name, df):
    le = LabelEncoder()
    # 欠損値を埋めて文字列化
    filled = df[col_name].fillna('Unknown').astype(str)
    le.fit(filled)
    return le

cat_cols = ['場所', '馬場状態', '天気']
for col in cat_cols:
    if col in df_train.columns:
        encoders[col] = get_encoder(col, df_train)
        df_train[col + '_enc'] = encoders[col].transform(df_train[col].fillna('Unknown').astype(str))
    else:
        df_train[col + '_enc'] = 0

# 数値系特徴量
if '斤量' in df_train.columns:
    df_train['斤量'] = df_train['斤量'].apply(force_numeric).fillna(55)
else:
    df_train['斤量'] = 55

if '馬体重' in df_train.columns:
    df_train['馬体重'] = df_train['馬体重'].apply(force_numeric).fillna(480)
else:
    df_train['馬体重'] = 480

if '馬体重増減' in df_train.columns:
    def parse_weight_change(x):
        if pd.isna(x): return 0
        try:
            x = str(x).replace(' ', '') # 空白除去
            return float(x)
        except: return 0
    df_train['馬体重増減_num'] = df_train['馬体重増減'].apply(parse_weight_change)
else:
    df_train['馬体重増減_num'] = 0

# 学習に使用する全特徴量
features = [
    '指数', '前走補正', '指数順位', '補正順位',
    '場所_enc', '馬場状態_enc', '天気_enc',
    '斤量', '馬体重', '馬体重増減_num'
]

X = df_train[features]
y = df_train['target']

# ------------------------------------------------
# 2. モデル検証
# ------------------------------------------------
print("\n📊 進化したAIモデルを検証中...")

if df_train['rid_group'].nunique() > 1:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=df_train['rid_group']))
    
    X_train = X.iloc[train_idx]; y_train = y.iloc[train_idx]
    X_val = X.iloc[val_idx]; y_val = y.iloc[val_idx]
    df_val_sim = df_train.iloc[val_idx].copy()
    
    model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated_model.fit(X_train, y_train)
    
    probs = calibrated_model.predict_proba(X_val)[:, 1]
    df_val_sim['prob'] = probs
    df_val_sim['expected_value'] = df_val_sim['prob'] * df_val_sim['単勝オッズ']
    
    # 断層計算
    df_val_sim = df_val_sim.sort_values(by=['rid_group', '単勝オッズ'])
    df_val_sim['next_odds'] = df_val_sim.groupby('rid_group')['単勝オッズ'].shift(-1)
    df_val_sim['gap_next'] = df_val_sim['next_odds'] / df_val_sim['単勝オッズ']
    df_val_sim['gap_next'] = df_val_sim['gap_next'].fillna(1.0)
    
    # 検証条件
    cond_zi = df_val_sim['指数順位'] == 1
    # cond_ai_top = df_val_sim.groupby('rid_group')['prob'].transform(max) == df_val_sim['prob']
    idx_max_prob = df_val_sim.groupby('rid_group')['prob'].idxmax()
    cond_ai_top = df_val_sim.index.isin(idx_max_prob)
    
    cond_gap = (df_val_sim['expected_value'] >= 1.0) & \
               (df_val_sim['prob'] >= 0.10) & \
               (df_val_sim['gap_next'] >= 1.5)
    
    def report_sim(name, condition):
        picks = df_val_sim[condition]
        if len(picks) == 0: return
        hits = picks[picks['target'] == 1]
        acc = len(hits) / len(picks) * 100
        rec = hits['単勝オッズ'].sum() / len(picks) * 100
        print(f"  [{name}] Acc: {acc:.2f}% / Rec: {rec:.2f}%")
    
    print("--- 🏁 検証結果 (ZI有効データのみ) ---")
    report_sim("ベースライン(ZI 1位)", cond_zi)
    report_sim("プランA(AI本命)", cond_ai_top)
    report_sim("プランB(AI+断層)", cond_gap)
    print("-" * 40)
    
    # 再学習
    print("🔄 本番用に再学習中...")
    calibrated_model.fit(X, y)
else:
    print("データ不足のため学習スキップ")
    sys.exit()

# ------------------------------------------------
# 3. 予想パート (特徴量追加版)
# ------------------------------------------------
print(f"\n🚀 出馬表({entry_file})で予想します...")
if not os.path.exists(entry_file): 
    print(f"❌ エラー: ファイルが見つかりません: {entry_file}")
    sys.exit(1)

try:
    # まずUTF-8で試す
    df_entry = pd.read_csv(entry_file, encoding='utf-8-sig')
except:
    try:
        # だめならCP932（Shift-JIS拡張）で試す
        df_entry = pd.read_csv(entry_file, encoding='cp932')
    except:
        # それでもだめならShift-JISで試す
        df_entry = pd.read_csv(entry_file, encoding='shift_jis')

df_entry.columns = df_entry.columns.str.strip()
df_pred = df_entry.copy()

# --- 特徴量作成（学習時と同じ処理）---
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

race_key = 'レース名' if 'レース名' in df_pred.columns else 'dummy'
if race_key == 'dummy': df_pred['dummy'] = 1
df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['前走補正'].rank(ascending=False, method='min')

# カテゴリ・数値変換
for col in cat_cols:
    # 学習時に見たことがないカテゴリは 'Unknown' 扱いにする対応が必要だが、
    # LabelEncoderは未知の値に弱いため、簡易的に fit 時のクラスを使うか 0 にする
    if col in df_pred.columns:
        # 安全策: mapを使って変換し、なければ0
        mapping = dict(zip(encoders[col].classes_, encoders[col].transform(encoders[col].classes_)))
        df_pred[col + '_enc'] = df_pred[col].astype(str).map(mapping).fillna(0)
    else:
        df_pred[col + '_enc'] = 0

if '斤量' in df_pred.columns: df_pred['斤量'] = df_pred['斤量'].apply(force_numeric).fillna(55)
else: df_pred['斤量'] = 55

if '馬体重' in df_pred.columns: df_pred['馬体重'] = df_pred['馬体重'].apply(force_numeric).fillna(480)
else: df_pred['馬体重'] = 480

if '馬体重増減' in df_pred.columns:
    def parse_weight(x):
        try: return float(str(x).replace(' ',''))
        except: return 0
    df_pred['馬体重増減_num'] = df_pred['馬体重増減'].apply(parse_weight)
else: df_pred['馬体重増減_num'] = 0

# 予測
X_pred = df_pred[features]
raw_probs = calibrated_model.predict_proba(X_pred)[:, 1]

# オッズ処理
odds_col_entry = None
for c in ['単勝', '単勝オッズ', '予想単勝オッズ']:
    if c in df_pred.columns: odds_col_entry = c; break
df_pred['単勝オッズ'] = df_pred[odds_col_entry].apply(force_numeric).fillna(0) if odds_col_entry else 0

total_prob = raw_probs.sum()
norm_probs = raw_probs / total_prob if total_prob > 0 else raw_probs
df_pred['AI勝率(%)'] = (norm_probs * 100).round(2)
df_pred['期待値'] = (norm_probs * df_pred['単勝オッズ'])

# オッズ断層 & 出力
def analyze_odds_gap(df_race):
    df_sorted = df_race[df_race['単勝オッズ'] > 0].sort_values('単勝オッズ')
    if len(df_sorted) < 6: return "", []
    odds = df_sorted['単勝オッズ'].values
    gaps = odds[1:] / odds[:-1]
    diag = []; targets = []
    if gaps[0] >= 2.5: diag.append(f"🦁 1人気鉄板(断層{gaps[0]:.1f})")
    elif gaps[0] < 1.5: diag.append(f"⚠️ 1人気危険(断層{gaps[0]:.1f})")
    
    mid_gaps = gaps[1:5]
    if len(mid_gaps) > 0:
        idx = np.argmax(mid_gaps)
        val = mid_gaps[idx]
        if val >= 2.0:
            t_idx = idx + 1
            name = df_sorted.iloc[t_idx]['馬名'] if '馬名' in df_sorted.columns else ''
            diag.append(f"💰 {t_idx+1}人気({name})狙い(断層{val:.1f})")
            targets.append(name)
            
    if all(g < 1.5 for g in gaps[:5]): diag.append("💤 混戦")
    return " / ".join(diag), targets

gap_msg, gap_targets = analyze_odds_gap(df_pred)
name_col = '馬名' if '馬名' in df_pred.columns else df_pred.columns[0]

def make_cmt(row):
    res = []
    if row['指数順位']==1: res.append("ZI1位")
    if row['期待値']>=1.0: res.append("★推奨")
    if row[name_col] in gap_targets: res.append("💰断層理論")
    return ",".join(res) if res else "-"
df_pred['診断'] = df_pred.apply(make_cmt, axis=1)

cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '診断', '指数']
disp = [c for c in cols if c in df_pred.columns]

print("\n=== 💰 最終予想 (ZI x 断層 x AI補正) ===")
print(df_pred[df_pred['単勝オッズ']>=1.0].sort_values('期待値', ascending=False)[disp].head(15))
print(f"\n💬 断層診断: {gap_msg}")