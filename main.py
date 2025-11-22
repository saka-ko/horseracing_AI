# ==========================================
# 🏇 競馬AI (ZI & 補正タイム特化型) - 修復版
# ==========================================
import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV

# ファイル設定
train_file = 'race_data_5years.csv'
entry_file = 'entry_table.csv'

# ------------------------------------------------
# 1. 学習データの読み込み & クリーニング
# ------------------------------------------------
print(f"🔄 学習データ({train_file})を読み込んでいます...")

# 読み込みトライアル (文字コード対策)
df_train = None
encodings = ['cp932', 'shift_jis', 'utf-8-sig', 'utf-8'] # TARGETはcp932が多い

for enc in encodings:
    try:
        df = pd.read_csv(train_file, encoding=enc, low_memory=False)
        # 読めたら列名チェック
        if any('着順' in col for col in df.columns) or any('ZI' in col for col in df.columns):
            df_train = df
            print(f"✅ {enc} で読み込み成功")
            break
    except:
        continue

if df_train is None:
    # 最終手段：エラー無視で読み込む
    df_train = pd.read_csv(train_file, encoding='shift_jis', errors='ignore', low_memory=False)
    print("⚠️ 強制読み込みを行いました（文字化けの可能性があります）")

# 列名のクリーニング
df_train.columns = df_train.columns.str.strip()
df_train = df_train.loc[:, ~df_train.columns.duplicated()]

# ------------------------------------------------------
# 🚑 列名救済措置 (ここが修正ポイント！)
# ------------------------------------------------------
# 「着順」が見つからない場合、「確定着順」などを探してリネームする
rank_col = None
if '着順' in df_train.columns:
    rank_col = '着順'
else:
    # '着順'を含む列を探す
    cands = [c for c in df_train.columns if '着順' in c]
    if cands:
        rank_col = cands[0]
        print(f"ℹ️ '{rank_col}' を '着順' として扱います")
        df_train.rename(columns={rank_col: '着順'}, inplace=True)

if '着順' not in df_train.columns:
    print("❌ エラー: 学習データに『着順』列が見つかりません。")
    print("現在の列名:", df_train.columns.tolist()[:10]) # 先頭10個だけ表示
    raise ValueError("学習データを確認してください。")

# 数値化関数
def force_numeric(x):
    if pd.isna(x): return np.nan
    try:
        x_str = str(x).translate(str.maketrans({chr(0xFF10 + i): chr(0x30 + i) for i in range(10)}))
        clean_str = re.sub(r'[^\d.-]', '', x_str)
        return float(clean_str)
    except: return np.nan

# ターゲット作成
df_train['着順_num'] = df_train['着順'].apply(force_numeric)
df_train = df_train.dropna(subset=['着順_num'])
df_train['target'] = (df_train['着順_num'] == 1).astype(int)

# 特徴量作成（シンプル特化）
# 必須列の確保
if '前走補正' not in df_train.columns:
    if '補正タイム.1' in df_train.columns: df_train['前走補正'] = df_train['補正タイム.1']
    elif '補正9' in df_train.columns: df_train['前走補正'] = df_train['補正9'] # TARGETの別名

if '指数' not in df_train.columns and 'ZI' in df_train.columns:
    df_train['指数'] = df_train['ZI']

# 数値化 & 欠損埋め
for f in ['指数', '前走補正']:
    if f in df_train.columns:
        df_train[f] = df_train[f].apply(force_numeric).fillna(0)
    else:
        df_train[f] = 0

# ランク計算 (相対評価)
race_id_col = 'レースID(新)' if 'レースID(新)' in df_train.columns else 'レースID'
if race_id_col in df_train.columns:
    df_train['指数順位'] = df_train.groupby(race_id_col)['指数'].rank(ascending=False, method='min')
    df_train['補正順位'] = df_train.groupby(race_id_col)['前走補正'].rank(ascending=False, method='min')
else:
    df_train['指数順位'] = 10; df_train['補正順位'] = 10

# 使用特徴量
features = ['指数', '前走補正', '指数順位', '補正順位']

print("🔥 ZI & 補正タイム特化モデルを学習中...")
X = df_train[features]
y = df_train['target']

# モデル学習
base_model = lgb.LGBMClassifier(random_state=42, n_estimators=100)
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrated_model.fit(X, y)
print("✅ 学習完了！")

# ------------------------------------------------
# 2. 最新オッズでの予想
# ------------------------------------------------
print(f"🚀 最新の出馬表({entry_file})で予想します...")

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

# マッピング
rename_map = {
    'ZI': '指数',
    '補正タイム.1': '前走補正', '補正タイム': '前走補正',
    '単勝': '単勝オッズ'
}
for k, v in rename_map.items():
    if k in df_pred.columns and v not in df_pred.columns:
        df_pred[v] = df_pred[k]

# 数値化
for f in ['指数', '前走補正', '単勝オッズ']:
    if f in df_pred.columns:
        df_pred[f] = df_pred[f].apply(force_numeric).fillna(0)
    else:
        df_pred[f] = 0

# ランク計算
race_key = 'レース名' if 'レース名' in df_pred.columns else '開催'
df_pred['指数順位'] = df_pred.groupby(race_key)['指数'].rank(ascending=False, method='min')
df_pred['補正順位'] = df_pred.groupby(race_key)['前走補正'].rank(ascending=False, method='min')

# 予測実行
X_pred = df_pred[features]
probs = calibrated_model.predict_proba(X_pred)[:, 1]
df_pred['AI勝率(%)'] = (probs * 100).round(1)
df_pred['期待値'] = (df_pred['AI勝率(%)'] / 100) * df_pred['単勝オッズ']

# 馬名取得
name_col = '馬名'
if '馬名' not in df_pred.columns:
    cands = [c for c in df_pred.columns if '馬名' in c]
    if cands: name_col = cands[0]

# 診断コメント
def make_comment(row):
    res = []
    if row['指数順位'] == 1: res.append("指数1位")
    if row['補正順位'] <= 3: res.append("補正上位")
    if row['期待値'] >= 1.0: res.append("★推奨")
    return ",".join(res) if res else "-"

df_pred['診断'] = df_pred.apply(make_comment, axis=1)

# 結果表示
print("\n=== 🎯 シンプルAI (指数＆補正) 推奨馬リスト ===")
cols = ['枠番', '馬番', name_col, '単勝オッズ', 'AI勝率(%)', '期待値', '診断', '指数', '前走補正']
disp_cols = [c for c in cols if c in df_pred.columns]

# オッズ100倍未満でソート
final_list = df_pred[
    (df_pred['単勝オッズ'] >= 1.0) & 
    (df_pred['単勝オッズ'] < 100.0)
].sort_values('期待値', ascending=False)

print(final_list[disp_cols].head(15))

if len(final_list) > 0:
    top = final_list.iloc[0]
    print(f"\n👑 最終本命: {top[name_col]} (期待値: {top['期待値']:.2f})")