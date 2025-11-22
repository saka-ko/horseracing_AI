# ==========================================
# 🏇 競馬AI (ZI & 補正タイム特化型) - 完全版
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# ------------------------------------------------
# 0. 設定
# ------------------------------------------------
train_file = 'race_5years_zi_hoseitime_kai.csv' # いただいたファイル
entry_file = 'entry_table.csv'      # 予想用ファイル

# コマンドライン引数対応
if len(sys.argv) > 1 and sys.argv[1].endswith('.csv'):
    entry_file = sys.argv[1]

# ------------------------------------------------
# 1. 学習データの読み込み（改良版）
# ------------------------------------------------
print(f"🔄 学習データ({train_file})を読み込んでいます...")

try:
    df_train = pd.read_csv(train_file, encoding='cp932', low_memory=False)
except:
    df_train = pd.read_csv(train_file, encoding='utf-8', low_memory=False)

df_train.columns = df_train.columns.str.strip()

# --- ★列名の自動マッピング ---
col_map = {}
# 必須列のエイリアス（別名）定義
aliases = {
    '着順': ['確定着順', '着順'],
    'ZI': ['指数', 'ZI', 'ZI値'],
    'オッズ': ['単勝オッズ', '単勝', '確定単勝オッズ'],
    'レースID': ['レースID(新)', 'レースID(旧)', 'レースID'],
    # 重要: ここで「前走」のデータだけを選ぶ
    '前走補正': ['前走補9', '前走補正', '前走タイム'] 
}

for key, candidates in aliases.items():
    for cand in candidates:
        if cand in df_train.columns:
            col_map[key] = cand
            break

# 必須チェック
if '着順' not in col_map or 'ZI' not in col_map:
    print(f"❌ エラー: 必要な列が見つかりません。現在の列名: {list(df_train.columns)}")
    sys.exit(1)

print("✅ データを正しく認識しました！")

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        import re
        # 全角→半角, 数字以外削除
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# データ整形
df_train['target'] = (df_train[col_map['着順']].apply(force_numeric) == 1).astype(int)
df_train['指数'] = df_train[col_map['ZI']].apply(force_numeric).fillna(0)
df_train['単勝オッズ'] = df_train[col_map['オッズ']].apply(force_numeric).fillna(0)

# 補正タイム（前走データのみ使用）
if '前走補正' in col_map:
    df_train['前走補正'] = df_train[col_map['前走補正']].apply(force_numeric).fillna(0)
else:
    # なければ0で埋める（エラーにしない）
    df_train['前走補正'] = 0

# --- 🚨 レースIDの修正（18桁問題対策） ---
# レースIDが長すぎる（馬番込み）場合は、末尾2桁をカットしてグルーピングする
rid_col = col_map['レースID']
df_train['rid_str'] = df_train[rid_col].astype(str)
# 簡易判定: 平均頭数が5頭以下ならIDが細かすぎると判断
if len(df_train) / df_train['rid_str'].nunique() < 5.0:
    print("ℹ️ レースIDを補正します（馬番を除去してグループ化）")
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
# 2. モデル検証 & 学習
# ------------------------------------------------
print("\n📊 モデルの実力を検証中（データを8:2に分割）...")

# 検証用データ分割
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
val_indices = X_val.index
val_odds = df_train.loc[val_indices, '単勝オッズ']
val_rids = df_train.loc[val_indices, 'rid_group']

# 学習
model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
calibrated.fit(X_train, y_train)

# 検証シミュレーション
probs_val = calibrated.predict_proba(X_val)[:, 1]
df_sim = pd.DataFrame({'rid': val_rids, 'target': y_val, 'prob': probs_val, 'odds': val_odds})

# 各レースで「AI推奨1位」の馬のみ購入
bets = df_sim.sort_values('prob', ascending=False).groupby('rid').head(1)
hits = bets[bets['target'] == 1]

accuracy = (len(hits) / len(bets)) * 100
recovery = (hits['odds'].sum() / len(bets)) * 100

print(f"--- 🏁 検証結果 (テストデータ {len(bets)}レース) ---")
print(f"🎯 的中率: {accuracy:.2f}%")
print(f"💰 回収率: {recovery:.2f}%")
print(f"--------------------------------------------------")

# 本番用再学習
print("🔄 本番用に全データで再学習しています...")
calibrated.fit(X, y)
print("✅ 学習完了！次のステップ（予想）へ進めます。")


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