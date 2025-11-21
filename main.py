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
# 2. 特徴量エンジニアリング (順位ランク重視)
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

# 数値列の処理
num_cols = ['指数', '前走補正', '前PCI', '前走PCI', '前走上り3F']
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# --------------------------------------------------------
# ★ New: 「レース内順位」を計算 (偏差値よりシンプルで強力)
# --------------------------------------------------------
race_id_col = 'レースID(新)' if 'レースID(新)' in df.columns else 'レースID'

if race_id_col in df.columns:
    print("レース内順位(ランキング)を計算中... AIが相対評価を理解します！")
    
    # 指数順位 (値が大きいほうが1位)
    df['指数順位'] = df.groupby(race_id_col)['指数'].rank(ascending=False, method='min')
    
    # 補正順位 (値が大きいほうが1位)
    df['補正順位'] = df.groupby(race_id_col)['前走補正'].rank(ascending=False, method='min')
    
    # 上がり3F順位 (値が小さいほうが1位)
    df['上り順位'] = df.groupby(race_id_col)['前走上り3F'].rank(ascending=True, method='min')
    
    # 自分の指数と、レース内1位の指数との差（トップとの差）
    df['指数トップ差'] = df.groupby(race_id_col)['指数'].transform('max') - df['指数']

else:
    print("※レースIDが見つからないため、順位計算をスキップ")
    df['指数順位'] = 10
    df['補正順位'] = 10
    df['上り順位'] = 10
    df['指数トップ差'] = 0

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
    '指数順位',      # ★1位なら強い
    '補正順位',      # ★1位なら速い
    '上り順位',      # ★1位ならキレる
    '指数トップ差',   # ★トップとどれくらい差があるか
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
# 3. モデル学習 (Rank重視)
# ==========================================
df['target_win'] = (df['着順_num'] == 1).astype(int)
X = df[features]
y = df['target_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n学習開始... (レース内順位を重視)")

# パラメータ調整: num_leavesを増やして、より複雑な条件（順位×コンビなど）を捉える
base_model = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=150, 
    min_child_samples=20, 
    num_leaves=50,
    n_jobs=-1
)

calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X_train, y_train)

# 重要度確認
base_model.fit(X_train, y_train)

# ==========================================
# 4. 結果分析 & 今週末の予想準備
# ==========================================
prob_win = calibrated_model.predict_proba(X_test)[:, 1]
results = X_test.copy()
results['馬名'] = df.loc[X_test.index, '馬名']
results['着順'] = df.loc[X_test.index, '着順_num']
results['単勝オッズ'] = pd.to_numeric(df.loc[X_test.index, '単勝オッズ'], errors='coerce').fillna(0)
results['AI勝率予測(%)'] = (prob_win * 100)
results['期待値'] = (results['AI勝率予測(%)'] / 100) * results['単勝オッズ']

print("\n=== 重要度ランキング (順位は機能したか？) ===")
importance = pd.DataFrame({'feature': features, 'importance': base_model.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(10))

# -----------------------------------------------------------------
# ★ 今週末の出馬表 (entry_table.csv) があれば自動で予想
# -----------------------------------------------------------------
entry_file_path = 'entry_table.csv'
import os

if os.path.exists(entry_file_path):
    print(f"\n🚀 続けて今週末のレースを予想します... ({entry_file_path})")
    try:
        df_entry = pd.read_csv(entry_file_path, encoding='utf-8-sig')
    except:
        try:
            df_entry = pd.read_csv(entry_file_path, encoding='cp932')
        except:
            df_entry = pd.read_csv(entry_file_path, encoding='shift_jis', errors='ignore')

    # --- 出馬表の前処理 ---
    if '前走着順' in df_entry.columns:
        df_entry['前走着順_num'] = df_entry['前走着順'].apply(clean_numeric)
    else:
        df_entry['前走着順_num'] = np.nan

    # 順位計算 (出馬表の中で計算)
    race_id_col_entry = 'レースID(新)' if 'レースID(新)' in df_entry.columns else 'レースID'
    if race_id_col_entry in df_entry.columns:
        # 指数は数値化しておく
        if '指数' in df_entry.columns:
            df_entry['指数'] = pd.to_numeric(df_entry['指数'], errors='coerce')
        if '前走補正' in df_entry.columns:
            df_entry['前走補正'] = pd.to_numeric(df_entry['前走補正'], errors='coerce')
        if '前走上り3F' in df_entry.columns:
            df_entry['前走上り3F'] = pd.to_numeric(df_entry['前走上り3F'], errors='coerce')

        df_entry['指数順位'] = df_entry.groupby(race_id_col_entry)['指数'].rank(ascending=False, method='min')
        df_entry['補正順位'] = df_entry.groupby(race_id_col_entry)['前走補正'].rank(ascending=False, method='min')
        df_entry['上り順位'] = df_entry.groupby(race_id_col_entry)['前走上り3F'].rank(ascending=True, method='min')
        df_entry['指数トップ差'] = df_entry.groupby(race_id_col_entry)['指数'].transform('max') - df_entry['指数']
    else:
        df_entry['指数順位'] = 10
        df_entry['補正順位'] = 10
        df_entry['上り順位'] = 10
        df_entry['指数トップ差'] = 0

    # その他特徴量作成 (学習時と同様)
    pci_cols_e = ['前PCI', '前走PCI', '前RPCI', '前走RPCI', '前PCI3', '前走PCI3']
    for col in pci_cols_e:
        if col in df_entry.columns:
            df_entry[col] = pd.to_numeric(df_entry[col], errors='coerce')
    
    df_entry['前走PCI_val'] = df_entry['前PCI'] if '前PCI' in df_entry.columns else df_entry['前走PCI'] if '前走PCI' in df_entry.columns else 50
    df_entry['コースID'] = df_entry['場所'].astype(str) + df_entry['芝・ダ'].astype(str) + df_entry['距離'].astype(str)
    df_entry['騎手調教師コンビ'] = df_entry['騎手コード'].astype(str) + "_" + df_entry['調教師コード'].astype(str)
    
    if '騎手コード' in df_entry.columns and '前走騎手コード' in df_entry.columns:
        df_entry['騎手継続フラグ'] = (df_entry['騎手コード'] == df_entry['前走騎手コード']).astype(int)
    else:
        df_entry['騎手継続フラグ'] = 0

    # エンコーディング適用
    for col in categorical_cols:
        if col in df_entry.columns and col in encoders:
            le = encoders[col]
            df_entry[col] = df_entry[col].fillna('unknown').astype(str)
            known_classes = set(le.classes_)
            df_entry[col] = df_entry[col].apply(lambda x: x if x in known_classes else 'unknown')
            if 'unknown' in known_classes:
                df_entry[col] = le.transform(df_entry[col])
            else:
                df_entry[col] = le.transform([le.classes_[0]] * len(df_entry))

    # 欠損処理
    for col in num_features:
        if col in df_entry.columns:
            temp_col = pd.to_numeric(df_entry[col], errors='coerce')
            df_entry[col] = temp_col.fillna(0)
        else:
            df_entry[col] = 0

    # 予測
    X_entry = df_entry[features]
    prob_entry = calibrated_model.predict_proba(X_entry)[:, 1]
    df_entry['AI勝率予測(%)'] = (prob_entry * 100).round(2)

    # 期待値
    if '単勝オッズ' in df_entry.columns: # 予想オッズがあれば
        df_entry['単勝オッズ'] = pd.to_numeric(df_entry['単勝オッズ'], errors='coerce').fillna(0)
        df_entry['期待値'] = (df_entry['AI勝率予測(%)'] / 100) * df_entry['単勝オッズ']
    elif '予想単勝オッズ' in df_entry.columns:
        df_entry['単勝オッズ'] = pd.to_numeric(df_entry['予想単勝オッズ'], errors='coerce').fillna(0)
        df_entry['期待値'] = (df_entry['AI勝率予測(%)'] / 100) * df_entry['単勝オッズ']
    else:
        df_entry['期待値'] = 0

    # 診断コメント
    def make_comment(row):
        reasons = []
        if row['指数順位'] == 1: reasons.append("指数1位")
        if row['上り順位'] == 1: reasons.append("上り1位")
        if row['補正順位'] <= 2: reasons.append("補正上位")
        return ",".join(reasons)
    
    df_entry['診断'] = df_entry.apply(make_comment, axis=1)

    print("\n=== 🎯 今週末の推奨馬リスト (期待値順) ===")
    disp_cols = ['レース名', '馬番', '馬名', '単勝オッズ', 'AI勝率予測(%)', '期待値', '診断']
    disp_cols = [c for c in disp_cols if c in df_entry.columns]
    
    # 単勝50倍未満で、期待値が高い順
    valid_entries = df_entry[
        (df_entry['単勝オッズ'] > 0) & (df_entry['単勝オッズ'] < 50)
    ].sort_values('期待値', ascending=False)
    
    print(valid_entries[disp_cols].head(20))
else:
    print("\n⚠️ 'entry_table.csv' が見つかりません。アップロードしてください。")