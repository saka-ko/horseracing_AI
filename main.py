import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

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

# 基本処理
df['着順_num'] = df['着順'].apply(clean_numeric)
df = df.dropna(subset=['着順_num'])
df['着順_num'] = df['着順_num'].astype(int)

if '前走着順' in df.columns:
    df['前走着順_num'] = df['前走着順'].apply(clean_numeric)
else:
    df['前走着順_num'] = np.nan

# ----------------------------------------------------
# ★ Factor 1: 展開の「質」を読む (PCI & Pace)
# ----------------------------------------------------
# PCI (Pace Change Index): 50が平均。
# 50超 = スローからの瞬発力勝負（末脚タイプ）
# 50未満 = ハイペースの消耗戦（持続力タイプ）

# 前走のPCI情報を数値化
pci_cols = ['前PCI', '前走PCI', '前RPCI', '前走RPCI', '前PCI3', '前走PCI3', '前好走']
for col in pci_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 列名の揺らぎを吸収（TARGETの設定によって名前が違うことがあるため）
df['前走PCI_val'] = df['前PCI'] if '前PCI' in df.columns else df['前走PCI'] if '前走PCI' in df.columns else 50
df['前走RPCI_val'] = df['前RPCI'] if '前RPCI' in df.columns else df['前走RPCI'] if '前走RPCI' in df.columns else 50

# ロングスパート適性を見るための「Ave-3F（上がり3ハロン以外の平均速度）」
# これが速い馬は、道中緩まないペースに強い（ハイペース・ロングスパート適性）
if '前走Ave-3F' in df.columns:
    df['前走Ave3F'] = pd.to_numeric(df['前走Ave-3F'], errors='coerce')
else:
    df['前走Ave3F'] = np.nan

# ----------------------------------------------------
# ★ Factor 2: 展開の「形」を読む (Escapers)
# ----------------------------------------------------
if '前走4角' in df.columns:
    df['前走脚質数値'] = df['前走4角'].apply(clean_numeric).fillna(10)
else:
    df['前走脚質数値'] = 10

# 逃げ馬フラグ
df['is_escaper'] = (df['前走脚質数値'] <= 1).astype(int) # 1番手のみを純粋な逃げと定義

# 同レースの逃げ馬数をカウント
race_id_col = 'レースID(新)' if 'レースID(新)' in df.columns else 'レースID'
if race_id_col in df.columns:
    df['同レース逃げ馬数'] = df.groupby(race_id_col)['is_escaper'].transform('sum') - df['is_escaper']
else:
    df['同レース逃げ馬数'] = 0

# ----------------------------------------------------
# ★ Factor 3: その他の重要ファクター
# ----------------------------------------------------
df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)
df['騎手調教師コンビ'] = df['騎手コード'].astype(str) + "_" + df['調教師コード'].astype(str)

if '騎手コード' in df.columns and '前走騎手コード' in df.columns:
    df['騎手継続フラグ'] = (df['騎手コード'] == df['前走騎手コード']).astype(int)
else:
    df['騎手継続フラグ'] = 0

# --- 特徴量リスト ---
features = [
    # 能力・指数
    '指数',           # ZI
    '前走補正',       # スピード指数
    
    # 展開・ラップ適性（ここを強化！）
    '前走PCI_val',    # その馬が前回どんなペース配分で走ったか（瞬発力or持久力）
    '前走RPCI_val',   # 前走のレース自体のペース（ハイペースだったかスローだったか）
    '前走Ave3F',      # 道中の厳しさへの対応力（ロングスパート適性）
    '同レース逃げ馬数', # 今回のペース予測
    '前走脚質数値',     # 位置取り

    # 前走成績
    '前走着順_num',
    '前走人気',
    '前走単勝オッズ',
    '前走上り3F',     # 末脚の絶対値
    '前走着差タイム',

    # 相性・属性
    '騎手継続フラグ',
    '騎手調教師コンビ',
    'コースID',
    
    # 基本データ
    '斤量', '馬番', '馬体重', '馬体重増減', '年齢', '間隔',
    '種牡馬', '場所', '芝・ダ', '距離'
]

features = [f for f in features if f in df.columns]
print(f"学習に使用する特徴量: {len(features)}個")
print(f"追加された展開系指標: {[f for f in features if 'PCI' in f or 'Ave' in f]}")

# --- カテゴリ変数処理 ---
categorical_cols = ['場所', '芝・ダ', '馬場状態', '種牡馬', '騎手コード', '調教師コード', 
                    '前走芝・ダ', 'コースID', '騎手調教師コンビ']

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].fillna('unknown').astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# 数値欠損処理
num_features = [f for f in features if f not in categorical_cols]
for col in num_features:
    if col in df.columns:
        temp_col = pd.to_numeric(df[col], errors='coerce')
        df[col] = temp_col.fillna(temp_col.mean())

# ==========================================
# 3. モデル学習 (LightGBM)
# ==========================================
df['target_win'] = (df['着順_num'] == 1).astype(int)
X = df[features]
y = df['target_win']

# データ量が多いので、学習データ80%、テスト20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n学習開始... (展開の質を学習中)")
# num_leaves(葉の数)を少し増やして、より複雑な条件（コース×ペース）を学習できるようにする
model = lgb.LGBMClassifier(random_state=42, n_estimators=150, num_leaves=63) 
model.fit(X_train, y_train)

# ==========================================
# 4. 期待値シミュレーション
# ==========================================
prob_win = model.predict_proba(X_test)[:, 1]

results = X_test.copy()
results['馬名'] = df.loc[X_test.index, '馬名']
results['着順'] = df.loc[X_test.index, '着順_num']
results['単勝オッズ'] = pd.to_numeric(df.loc[X_test.index, '単勝オッズ'], errors='coerce').fillna(0)
results['AI勝率予測(%)'] = (prob_win * 100).round(2)
results['期待値'] = (results['AI勝率予測(%)'] / 100) * results['単勝オッズ']

auc = roc_auc_score(y_test, prob_win)
print(f"\nモデル精度(AUC): {auc:.4f}")

# --- 期待値ランキング ---
print("\n=== 【期待値ランキング】トップ10 (AI厳選) ===")
display_cols = ['馬名', '着順', '単勝オッズ', 'AI勝率予測(%)', '期待値']
sorted_results = results.sort_values('期待値', ascending=False)
print(sorted_results[sorted_results['単勝オッズ'] > 0][display_cols].head(10))

# --- 重要度ランキング ---
print("\n=== 重要度ランキング (AIが重視したファクター) ===")
importance = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print(importance.sort_values('importance', ascending=False).head(15))