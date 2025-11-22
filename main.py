# ==========================================
# 🏇 競馬AI (ZI & 補正タイム特化型) - CLI対応版
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------
# 0. 設定とコマンドライン引数の取得
# ------------------------------------------------
train_file = 'race_data_5years.csv' # 学習用データ（固定）
entry_file = 'entry_table.csv'      # デフォルトの予想用ファイル

# コマンドライン引数がある場合は、それを予想ファイルとして使う
# 使い方: python main.py [ファイル名.csv]
if len(sys.argv) > 1:
    # Colabなどのシステム引数(-f など)を除外する簡易チェック
    if sys.argv[1].endswith('.csv'):
        entry_file = sys.argv[1]

print(f"📂 学習データ: {train_file}")
print(f"📂 予想データ: {entry_file}")

# ------------------------------------------------
# 1. 学習データの読み込み & クリーニング
# ------------------------------------------------
print(f"🔄 学習データを読み込んでいます...")

# 読み込みトライアル
df_train = None
encodings = ['utf-8-sig', 'cp932', 'shift_jis', 'utf-8'] 

for enc in encodings:
    try:
        df = pd.read_csv(train_file, encoding=enc, low_memory=False)
        df.columns = df.columns.str.strip()
        # 必須列があるかチェック
        if any('着順' in col for col in df.columns) or any('ZI' in col for col in df.columns):
            df_train = df
            break
    except:
        continue

if df_train is None:
    print(f"❌ エラー: 学習データ({train_file})が見つかりません。")
    sys.exit(1) # 終了

# 重複列の削除
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# 列名救済措置
rank_col = None
if '着順' in df_train.columns: rank_col = '着順'
elif '確定着順' in df_train.columns: rank_col = '確定着順'

if not rank_col:
    # 着順を含み、数字っぽい列を探す
    cands = [c for c in df_train.columns if '着順' in c]
    if cands: rank_col = cands[0]
    else:
        print("❌ 学習データに『着順』列が見つかりません。")
        sys.exit(1)

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        import re
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# ターゲット作成
df_train['着順_num'] = df_train[rank_col].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 特徴量作成
# 学習時は「前走」のデータだけを使う
if '前走補正' not in df_train.columns:
    if '前走補9' in df_train.columns: df_train['前走補正'] = df_train['前走補9']
    elif '補9' in df_train.columns: df_train['前走補正'] = df_train['補9']
    elif '補正タイム.1' in df_train.columns: df_train['前走補正'] = df_train['補正タイム.1']
    else: df_train['前走補正'] = 0

if '指数' not in df_train.columns:
    if 'ZI' in df_train.columns: df_train['指数'] = df_train['ZI']
    else: df_train['指数'] = 0

# 数値化 & 欠損埋め
for f in ['指数', '前走補正']:
    df_train[f] = df_train[f].apply(force_numeric).fillna(0)

# ランク計算 (レース内順位)
race_id_col = 'レースID(新)' if 'レースID(新)' in df_train.columns else 'レースID'
if race_id_col in df_train.columns:
    df_train['指数順位'] = df_train.groupby(race_id_col)['指数'].rank(ascending=False, method='min')
    df_train['補正順位'] = df_train.groupby(race_id_col)['前走補正'].rank(ascending=False, method='min')
else:
    # IDがない場合、日付と場所で仮ID作成
    if '日付(yyyy.mm.dd)' in df_train.columns and '場所' in df_train.columns:
         df_train['rid'] = df_train['日付(yyyy.mm.dd)'].astype(str) + df_train['場所']
         df_train['指数順位'] = df_train.groupby('rid')['指数'].rank(ascending=False, method='min')
         df_train['補正順位'] = df_train.groupby('rid')['前走補正'].rank(ascending=False, method='min')
    else:
         df_train['指数順位'] = 10; df_train['補正順位'] = 10

# 使用する特徴量
features = ['指数', '前走補正', '指数順位', '補正順位']

print("🔥 ZI & 補正タイム特化モデルを学習中...")
X = df_train[features]
y = df_train['target']

# モデル学習
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated_model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 2. 最新オッズでの予想 (過去3走評価)
# ------------------------------------------------
print(f"🚀 出馬表({entry_file})で予想します...")

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

# 列名クリーニング
df_entry.columns = df_entry.columns.str.strip()
df_entry = df_entry.loc[:, ~df_entry.columns.duplicated()]
df_pred = df_entry.copy()

# --- ★重要: 過去3走から最大補正タイムを取得 ---
# 列名を探す (補:1, 補:2, 補:3 または 補正タイム.1, 補正タイム.2...)
hosei_cols = []
for i in range(1, 4):
    c1 = f'補:{i}'
    c2 = f'補正タイム.{i}'
    if c1 in df_pred.columns: hosei_cols.append(c1)
    elif c2 in df_pred.columns: hosei_cols.append(c2)

# 「補正タイム」自体も候補に入れる（TARGETの仕様により1走前の場合がある）
if '補正タイム' in df_pred.columns:
    hosei_cols.append('補正タイム')

# 重複除去
hosei_cols = list(set(hosei_cols))
# print(f"ℹ️ 参照する過去走データ列: {hosei_cols}")

def get_max_hosei(row):
    vals = []
    for c in hosei_cols:
        v = force_numeric(row[c])
        if v > 0: vals.append(v)
    return max(vals) if vals else 0

# 最大値を「前走補正」として扱う
df_pred['前走補正'] = df_pred.apply(get_max_hosei, axis=1)

# 指数 (ZI)
if 'ZI' in df_pred.columns: df_pred['指数'] = df_pred['ZI'].apply(force_numeric).fillna(0)
else: df_pred['指数'] = 0

# 単勝オッズ
odds_col = None
for c in ['単勝', '単勝オッズ', '予想単勝オッズ']:
    if c in df_pred.columns:
        odds_col = c
        break
if odds_col:
    df_pred['単勝オッズ'] = df_pred[odds_col].apply(force_numeric).fillna(0)
else:
    df_pred['単勝オッズ'] = 0

# ランク計算
# レース名がない場合、すべて同じレースとみなして順位をつける
race_key = 'レース名' 
if race_key not in df_pred.columns:
    df_pred['dummy'] = 1
    race_key = 'dummy'

df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['前走補正'].rank(ascending=False, method='min')

# 予測実行
X_pred = df_pred[features]
raw_probs = calibrated_model.predict_proba(X_pred)[:, 1]

# ★確率の正規化（合計を100%にする）
total_prob = raw_probs.sum()
if total_prob > 0:
    normalized_probs = raw_probs / total_prob
else:
    normalized_probs = raw_probs

df_pred['AI勝率(%)'] = (normalized_probs * 100).round(2)
df_pred['期待値'] = (normalized_probs * df_pred['単勝オッズ'])

# 馬名取得
name_col = '馬名'
if '馬名' not in df_pred.columns:
    cands = [c for c in df_pred.columns if '馬名' in c]
    if cands: name_col = cands[0]

# 診断コメント
def make_comment(row):
    res = []
    if row['指数順位'] == 1: res.append("指数1位")
    if row['補正順位'] == 1: res.append("能力1位")
    elif row['補正順位'] <= 3: res.append("能力上位")
    if row['期待値'] >= 1.0: res.append("★推奨")
    return ",".join(res) if res else "-"

df_pred['診断'] = df_pred.apply(make_comment, axis=1)

# --- 結果出力 ---
cols_out = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '診断', '指数', '前走補正']
disp_cols = [c for c in cols_out if c in df_pred.columns]

# 1. 期待値ランキング
print("\n=== 💰 期待値ランキング (回収率重視) ===")
print("※『前走補正』欄は、過去3走のベスト数値を表示しています")
final_list_ev = df_pred[df_pred['単勝オッズ'] >= 1.0].sort_values('期待値', ascending=False)
print(final_list_ev[disp_cols].head(15))

# 2. 勝率ランキング
print("\n=== 🏅 AI勝率ランキング (的中率重視) ===")
final_list_prob = df_pred.sort_values('AI勝率(%)', ascending=False)
print(final_list_prob[disp_cols].head(15))

if len(final_list_ev) > 0:
    top_ev = final_list_ev.iloc[0]
    print(f"\n💰 期待値No.1: {top_ev[name_col]} (期待値 {top_ev['期待値']:.2f})")
if len(final_list_prob) > 0:
    top_prob = final_list_prob.iloc[0]
    print(f"👑 勝率No.1  : {top_prob[name_col]} (勝率 {top_prob['AI勝率(%)']}%)")