import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# ==========================================
# 1. データの読み込み
# ==========================================
file_path = 'race_data_5years.csv' 

print(f"データを読み込んでいます... ({file_path})")
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except:
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except:
        df = pd.read_csv(file_path, encoding='shift_jis', errors='ignore')

print(f"データ読み込み完了: {len(df)}件")

# ==========================================
# 2. 特徴量エンジニアリング (相対評価の追加)
# ==========================================

def clean_numeric(x):
    if pd.isna(x): return np.nan
    x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
    try:
        return float(x_str)
    except ValueError:
        return np.nan

# 数値化
df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num'])
df['着順_num'] = df['着順_num'].astype(int)

if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# 数値列の処理
num_cols = ['指数', '前走補正', '前PCI', '前走PCI', '前走上り3F', '前走着差タイム']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        df[col] = np.nan # ない場合は欠損値

# --------------------------------------------------------
# ★ New: 「レース内偏差値」を計算する魔法の関数
# --------------------------------------------------------
# 「そのレースの中で、その馬がどれくらい強いか」を数値化します
def calculate_deviation(series):
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return 50.0 # 差がない場合は偏差値50
    return 50.0 + 10.0 * (series - mean) / std

race_id_col = 'レースID(新)' if 'レースID(新)' in df.columns else 'レースID'

if race_id_col in df.columns:
    print("相対評価(偏差値)を計算中... これが効きます！")
    # 指数の偏差値（メンバー内でどれだけ抜けているか）
    df['指数_偏差値'] = df.groupby(race_id_col)['指数'].transform(calculate_deviation).fillna(50)
    
    # 前走補正の偏差値（スピードの相対評価）
    df['補正_偏差値'] = df.groupby(race_id_col)['前走補正'].transform(calculate_deviation).fillna(50)
    
    # 上がり3Fの偏差値（このメンバーの中でキレるかどうか。※タイムは小さい方が良いので正負逆転）
    # 速いほうが偏差値高くなるように -1 をかける
    df['上り_偏差値'] = df.groupby(race_id_col)['前走上り3F'].transform(
        lambda x: calculate_deviation(-x)
    ).fillna(50)
else:
    print("※レースIDが見つからないため、相対評価計算をスキップしました")
    df['指数_偏差値'] = 50
    df['補正_偏差値'] = 50
    df['上り_偏差値'] = 50

# --- その他のファクター ---
# PCI統一
df['前走PCI_val'] = df['前PCI'] if '前PCI' in df.columns else df['前走PCI'] if '前走PCI' in df.columns else 50
# ID作成
df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)
df['騎手調教師コンビ'] = df['騎手コード'].astype(str) + "_" + df['調教師コード'].astype(str)
if '騎手コード' in df.columns and '前走騎手コード' in df.columns:
    df['騎手継続フラグ'] = (df['騎手コード'] == df['前走騎手コード']).astype(int)
else:
    df['騎手継続フラグ'] = 0

# --- Features ---
features = [
    '指数_偏差値',    # ★最強の新規追加
    '補正_偏差値',    # ★最強の新規追加
    '上り_偏差値',    # ★最強の新規追加
    '指数', '前走補正', 
    '前走PCI_val', 
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
# 3. モデル学習 (偏差値入り)
# ==========================================
df['target_win'] = (df['着順_num'] == 1).astype(int)
X = df[features]
y = df['target_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n学習開始... (相対評価を学習中)")

base_model = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=120, # 少し増やす
    min_child_samples=30,
    num_leaves=40,
    n_jobs=-1
)

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# 重要度確認用
base_model.fit(X_train, y_train)

# ==========================================
# 4. グリッドサーチ (必勝法探し)
# ==========================================
prob_win = calibrated_model.predict_proba(X_test)[:, 1]
results = X_test.copy()
results['馬名'] = df.loc[X_test.index, '馬名']
results['着順'] = df.loc[X_test.index, '着順_num']
results['単勝オッズ'] = pd.to_numeric(df.loc[X_test.index, '単勝オッズ'], errors='coerce').fillna(0)
results['AI勝率予測(%)'] = (prob_win * 100)
results['期待値'] = (results['AI勝率予測(%)'] / 100) * results['単勝オッズ']

print("\n🚀 新モデルで最適な買い条件を探索中...")

best_strategies = []
min_odds_list = [5.0, 10.0, 15.0]
max_odds_list = [20.0, 30.0, 50.0, 100.0]
min_exp_list = [0.8, 1.0, 1.2, 1.5]

for min_odds in min_odds_list:
    for max_odds in max_odds_list:
        if min_odds >= max_odds: continue
        for min_exp in min_exp_list:
            target = results[
                (results['単勝オッズ'] >= min_odds) & 
                (results['単勝オッズ'] < max_odds) &
                (results['期待値'] >= min_exp)
            ]
            count = len(target)
            if count < 50: continue 
            
            invest = count * 100
            ret = target[target['着順'] == 1]['単勝オッズ'].sum() * 100
            rate = (ret / invest) * 100
            profit = ret - invest
            
            if rate >= 95: # ハードルを少し下げて傾向を見る
                best_strategies.append({
                    'オッズ': f"{min_odds}-{max_odds}",
                    '期待値': f"{min_exp}以上",
                    '件数': count,
                    '回収率': f"{rate:.1f}%",
                    '収支': profit
                })

if len(best_strategies) > 0:
    strategy_df = pd.DataFrame(best_strategies)
    print("\n=== 🏆 回収率ランキング (偏差値導入後) ===")
    print(strategy_df.sort_values('収支', ascending=False).head(15))
else:
    print("\n条件は見つかりませんでしたが、重要度ランキングを確認してください↓")

print("\n=== 重要度ランキング ===")
importance = pd.DataFrame({'feature': features, 'importance': base_model.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(10))