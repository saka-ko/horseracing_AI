import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# 1. データの読み込み
# ==========================================
# ★ 5年分のファイル名を指定
file_path = 'race_data_5years.csv' 

print(f"データを読み込んでいます... ({file_path})")
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='shift_jis', errors='ignore')

print(f"データ読み込み完了: {len(df)}件")

# ==========================================
# 2. 特徴量エンジニアリング
# ==========================================
def clean_numeric(x):
    if pd.isna(x): return np.nan
    x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
    try:
        return float(x_str)
    except ValueError:
        return np.nan

df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num'])
df['着順_num'] = df['着順_num'].astype(int)

if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# --- 展開・PCI ---
pci_cols = ['前PCI', '前走PCI', '前RPCI', '前走RPCI', '前PCI3', '前走PCI3']
for col in pci_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df['前走PCI_val'] = df['前PCI'] if '前PCI' in df.columns else df['前走PCI'] if '前走PCI' in df.columns else 50
df['前走RPCI_val'] = df['前RPCI'] if '前RPCI' in df.columns else df['前走RPCI'] if '前走RPCI' in df.columns else 50

if '前走Ave-3F' in df.columns:
    df['前走Ave3F'] = pd.to_numeric(df['前走Ave-3F'], errors='coerce')
else:
    df['前走Ave3F'] = np.nan

if '前走4角' in df.columns:
    df['前走脚質数値'] = df['前走4角'].apply(clean_numeric).fillna(10)
else:
    df['前走脚質数値'] = 10

df['is_escaper'] = (df['前走脚質数値'] <= 1).astype(int)
race_id_col = 'レースID(新)' if 'レースID(新)' in df.columns else 'レースID'
if race_id_col in df.columns:
    df['同レース逃げ馬数'] = df.groupby(race_id_col)['is_escaper'].transform('sum') - df['is_escaper']
else:
    df['同レース逃げ馬数'] = 0

df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)
df['騎手調教師コンビ'] = df['騎手コード'].astype(str) + "_" + df['調教師コード'].astype(str)

if '騎手コード' in df.columns and '前走騎手コード' in df.columns:
    df['騎手継続フラグ'] = (df['騎手コード'] == df['前走騎手コード']).astype(int)
else:
    df['騎手継続フラグ'] = 0

# --- Features ---
features = [
    '指数', '前走補正', 
    '前走PCI_val', '前走RPCI_val', '前走Ave3F', '同レース逃げ馬数', '前走脚質数値',
    '前走着順_num', '前走人気', '前走単勝オッズ', '前走上り3F', '前走着差タイム',
    '騎手継続フラグ', '騎手調教師コンビ', 'コースID',
    '斤量', '馬番', '馬体重', '馬体重増減', '年齢', '間隔', '種牡馬', '場所', '芝・ダ', '距離'
]
features = [f for f in features if f in df.columns]

# --- Encoding ---
categorical_cols = ['場所', '芝・ダ', '馬場状態', '種牡馬', '騎手コード', '調教師コード', 
                    '前走芝・ダ', 'コースID', '騎手調教師コンビ']

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna('unknown').astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

num_features = [f for f in features if f not in categorical_cols]
for col in num_features:
    if col in df.columns:
        temp_col = pd.to_numeric(df[col], errors='coerce')
        df[col] = temp_col.fillna(temp_col.mean())

# ==========================================
# 3. モデル学習 (高速化版)
# ==========================================
df['target_win'] = (df['着順_num'] == 1).astype(int)
X = df[features]
y = df['target_win']

# シミュレーション用にレースIDも保持しておく
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, df[race_id_col], test_size=0.2, random_state=42
)

print("\n学習開始... (設定: 高速モード cv=3)")

base_model = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=100,
    min_child_samples=50, 
    reg_alpha=0.1,
    n_jobs=-1
)

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# ==========================================
# 4. 💰 回収率バックテスト
# ==========================================
prob_win = calibrated_model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results['レースID'] = ids_test
results['馬名'] = df.loc[X_test.index, '馬名']
results['着順'] = df.loc[X_test.index, '着順_num']
results['単勝オッズ'] = pd.to_numeric(df.loc[X_test.index, '単勝オッズ'], errors='coerce').fillna(0)
results['AI勝率予測(%)'] = (prob_win * 100)
results['期待値'] = (results['AI勝率予測(%)'] / 100) * results['単勝オッズ']

# 100倍以上の大穴はノイズとして除外する（現実的な運用のため）
results = results[results['単勝オッズ'] < 100]

print("\n" + "="*50)
print(" 💰 回収率シミュレーション結果 (単勝ベタ買い)")
print("="*50)

# --- パターンA: 期待値が「○以上」なら全部買う ---
print("\n【パターンA】期待値によるフィルタリング")
print(f"{'条件(期待値)':<10} | {'購入件数':<8} | {'的中率':<8} | {'回収率':<8} | {'収支(1点100円)':<10}")
print("-" * 65)

for threshold in [0.8, 1.0, 1.2, 1.5, 2.0, 3.0]:
    # 条件に合う馬を抽出
    bet_df = results[results['期待値'] >= threshold]
    
    if len(bet_df) == 0:
        continue
        
    bet_count = len(bet_df)
    hits = bet_df[bet_df['着順'] == 1]
    hit_count = len(hits)
    
    investment = bet_count * 100
    return_amount = hits['単勝オッズ'].sum() * 100
    recovery_rate = (return_amount / investment) * 100
    profit = return_amount - investment
    
    print(f"{threshold:>6.1f}以上 | {bet_count:>8} | {hit_count/bet_count*100:>7.1f}% | {recovery_rate:>7.1f}% | {profit:>+10.0f}円")

# --- パターンB: 各レースで「一番期待値が高い馬」だけ買う ---
print("\n【パターンB】各レース 期待値No.1の馬だけ購入")
# レースごとに期待値最大の行を取得
top_picks = results.loc[results.groupby('レースID')['期待値'].idxmax()]

# さらに「そのNo.1の馬の期待値が1.0を超えている場合のみ」買う条件を追加
top_picks_filtered = top_picks[top_picks['期待値'] >= 1.0]

bet_count_b = len(top_picks_filtered)
hits_b = top_picks_filtered[top_picks_filtered['着順'] == 1]
hit_count_b = len(hits_b)
investment_b = bet_count_b * 100
return_amount_b = hits_b['単勝オッズ'].sum() * 100
recovery_rate_b = (return_amount_b / investment_b) * 100 if bet_count_b > 0 else 0
profit_b = return_amount_b - investment_b

print(f"条件: レース内1位 & 期待値1.0以上")
print(f"購入件数: {bet_count_b}件")
print(f"的中率  : {hit_count_b / bet_count_b * 100:.1f}%")
print(f"回収率  : {recovery_rate_b:.1f}%")
print(f"収支    : {profit_b:+,.0f}円")
print("="*50)