# ==========================================
# 🏇 あなた専用モデル：能力＆ラップ特化型AI
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# 1. ファイル読み込み
train_file = 'race_data_5years.csv'
entry_file = 'entry_table.csv'

print(f"🔄 学習データ({train_file})を読み込んでいます...")
try:
    df_train = pd.read_csv(train_file, encoding='utf-8-sig', low_memory=False)
except:
    try:
        df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
    except:
        df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)

# 2. 特徴量の厳選 (あなたが見ているファクターのみ)
# ------------------------------------------------
features = [
    # --- 能力評価 ---
    '指数',           # ZI値（総合能力）
    '前走補正',       # スピード指数（タイムレベル）
    '前走着順_num',   # 着順
    '前走着差タイム', # 負け方（0.1秒差なら逆転可能など）
    
    # --- ラップ・展開 ---
    '前走PCI_val',    # ペース配分（瞬発力or持久力）
    '前走RPCI_val',   # レース全体のペース
    '前走Ave3F',      # 前半〜中盤のスピード
    '前走上り3F',     # 末脚の絶対値
    
    # --- 舞台設定 ---
    'コースID'        # コース適性（必須）
]
# ------------------------------------------------

# データクリーニング関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        # 全角→半角、記号削除
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except:
        return np.nan

# 前処理プロセス
def preprocess_data(df, is_train=True):
    # 列名クリーニング
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    
    # コースID作成
    if '場所' not in df.columns:
        if '開催' in df.columns:
            place_map = {'札':'札幌', '函':'函館', '福':'福島', '新':'新潟', '東':'東京', '中':'中山', '京':'京都', '阪':'阪神', '小':'小倉'}
            df['場所'] = df['開催'].astype(str).apply(lambda x: place_map.get(x[1], 'その他') if len(x)>1 else 'その他')
        else:
            df['場所'] = 'その他'
    
    if '芝・ダ' not in df.columns: df['芝・ダ'] = '芝'
    if '距離' not in df.columns: df['距離'] = 1600
    
    df['コースID'] = df['場所'].astype(str) + df['芝・ダ'].astype(str) + df['距離'].astype(str)

    # 数値項目のマッピングと変換
    # 学習データと出馬表データの列名の違いを吸収
    if not is_train:
        rename_map = {
            'ZI': '指数',
            '補正タイム.1': '前走補正', # 出馬表の過去走
            '補正タイム': '前走補正',   # 念のため
            '着順.1': '前走着順',
            '着差.1': '前走着差タイム',
            '上り3F.1': '前走上り3F',
            'PCI.1': '前走PCI',
            'Ave-3F.1': '前走Ave-3F',
            'PCI': '前走PCI',         # 出馬表によってはここに入る
            '単勝': '単勝オッズ'
        }
        # 存在する列だけリネーム
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

    # 各特徴量の数値化 (最重要)
    # 1. 着順
    if '着順' in df.columns:
        # 学習用ターゲット
        if is_train:
            df['着順_num'] = df['着順'].apply(force_numeric)
            df = df.dropna(subset=['着順_num'])
            df['着順_num'] = df['着順_num'].astype(int)
    
    # 2. 前走着順
    if '前走着順' in df.columns: df['前走着順_num'] = df['前走着順'].apply(force_numeric)
    else: df['前走着順_num'] = np.nan

    # 3. タイム差
    if '前走着差タイム' in df.columns: df['前走着差タイム'] = df['前走着差タイム'].apply(force_numeric)
    else: df['前走着差タイム'] = 0

    # 4. PCI (ラップ)
    if '前走PCI' in df.columns: df['前走PCI_val'] = df['前走PCI'].apply(force_numeric)
    elif 'PCI.1' in df.columns: df['前走PCI_val'] = df['PCI.1'].apply(force_numeric)
    else: df['前走PCI_val'] = 50 # 平均

    # 5. RPCI
    if '前走RPCI' in df.columns: df['前走RPCI_val'] = df['前走RPCI'].apply(force_numeric)
    elif 'レースPCI.1' in df.columns: df['前走RPCI_val'] = df['レースPCI.1'].apply(force_numeric)
    else: df['前走RPCI_val'] = 50

    # 6. Ave-3F
    if '前走Ave-3F' in df.columns: df['前走Ave3F'] = df['前走Ave-3F'].apply(force_numeric)
    else: df['前走Ave3F'] = np.nan

    # 7. その他 (指数, 補正, 上がり)
    for col in ['指数', '前走補正', '前走上り3F']:
        if col in df.columns:
            df[col] = df[col].apply(force_numeric)
        else:
            df[col] = np.nan

    return df

# データ処理実行
df_train = preprocess_data(df_train, is_train=True)

# コースIDエンコーディング
le_course = LabelEncoder()
df_train['コースID'] = df_train['コースID'].astype(str)
le_course.fit(df_train['コースID'])
df_train['コースID'] = le_course.transform(df_train['コースID'])

# 欠損値埋め
for f in features:
    if f in df_train.columns:
        df_train[f] = df_train[f].fillna(df_train[f].mean())
    else:
        df_train[f] = 0

print("🔥 あなた専用モデル(能力重視)を学習中...")
X = df_train[features]
y = (df_train['着順_num'] == 1).astype(int)

model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 3. 出馬表で予想
# ------------------------------------------------
print(f"🚀 出馬表({entry_file})を読み込んで予想します...")
try:
    df_entry = pd.read_csv(entry_file, encoding='utf-8-sig')
except:
    try:
        df_entry = pd.read_csv(entry_file, encoding='cp932')
    except:
        df_entry = pd.read_csv(entry_file, encoding='shift_jis', errors='replace')

# 前処理
df_entry = preprocess_data(df_entry, is_train=False)

# コースID変換 (未知のコースは'0'扱い)
df_entry['コースID'] = df_entry['コースID'].astype(str).apply(lambda x: x if x in le_course.classes_ else le_course.classes_[0])
df_entry['コースID'] = le_course.transform(df_entry['コースID'])

# 欠損埋め
for f in features:
    if f in df_entry.columns:
        df_entry[f] = df_entry[f].fillna(df_train[f].mean()) # 学習データの平均で埋める
    else:
        df_entry[f] = 0

# 予測
X_entry = df_entry[features]
probs = model.predict_proba(X_entry)[:, 1]
df_entry['AI勝率(%)'] = (probs * 100).round(1)

# 期待値 (オッズがあれば)
if '単勝オッズ' in df_entry.columns:
    df_entry['単勝オッズ'] = df_entry['単勝オッズ'].apply(force_numeric).fillna(0)
    df_entry['期待値'] = (df_entry['AI勝率(%)'] / 100) * df_entry['単勝オッズ']
else:
    df_entry['単勝オッズ'] = 0
    df_entry['期待値'] = 0

# 馬名の取得
name_col = [c for c in df_entry.columns if '馬名' in c][0] if [c for c in df_entry.columns if '馬名' in c] else '馬名'
if name_col not in df_entry.columns: df_entry[name_col] = 'Unknown'

# 結果表示
print("\n=== 🎯 能力＆ラップ特化型AI 推奨馬 ===")
out_cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '指数', '前走補正', '前走着差タイム']
# 存在する列だけ表示
out_cols = [c for c in out_cols if c in df_entry.columns]

print(df_entry.sort_values('AI勝率(%)', ascending=False)[out_cols].head(15))

# 重要度確認
print("\n=== このAIが重視したファクター ===")
imp = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
print(imp.sort_values('importance', ascending=False))