# -*- coding: utf-8 -*-
import requests
import time
import csv
from flask import Flask, render_template, request
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

now = datetime.now()

# データを取得する期間の設定（昨日から365日前まで）
end_date_str = (now - timedelta(days=1)).strftime('%Y-%m-%d')
start_date_str = (now - timedelta(days=365)).strftime('%Y-%m-%d')

# 時間単位で外気温・外湿度を取得する
weather_url = f"https://archive-api.open-meteo.com/v1/archive?latitude=35.1351&longitude=136.9781&start_date={start_date_str}&end_date={end_date_str}&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FTokyo"

LATEST_API_URL = 'https://airoco.necolico.jp/data-api/latest'
ID = 'CgETViZ2'
SUB_KEY = '6b8aa7133ece423c836c38af01c59880'
ROOM_LIST = [
    "Ｒ３ー３０１",
    "Ｒ３ー４０１",
    "Ｒ３ー４０３",
    "Ｒ３ーB1Ｆ_ＥＨ",
    "Ｒ３ー１Ｆ_ＥＨ",
    "Ｒ３ー３Ｆ_ＥＨ",
    "Ｒ３ー４Ｆ_ＥＨ"
]

app = Flask(__name__)

# グローバル変数として学習済みモデルを保持
trained_model = None

# 屋外の気温と湿度を取得しCSVファイルに保存
def get_outdoor_temphumid():
    print("屋外の1年分の室温・温度データを取得中...")
    response = requests.get(weather_url)
    data = response.json()

    df = pd.DataFrame({
        'date': pd.to_datetime(data['hourly']['time']),
        'outdoor_temp': data['hourly']['temperature_2m'],
        'outdoor_humid': data['hourly']['relative_humidity_2m']
    })
    
    df.to_csv('annual_outdoor_data.csv', index=False)
    print("屋外のCSVファイルに保存しました。")
    return 'annual_outdoor_data.csv'

# 室内の気温と湿度を取得しCSVファイルに保存
def get_room_temphumid(room_name):
    room_records = []
    print("ネコリコAPIから1年分の室温・湿度データを取得中")
    
    # 過去365日分を1日ずつ遡って取得するループ
    for i in range(365):
        target_date = now - timedelta(days=i)
        tt = int(target_date.timestamp())
        
        url = f'https://airoco.necolico.jp/data-api/day-csv?id={ID}&subscription-key={SUB_KEY}&startDate={tt}'
        
        try:
            res = requests.get(url)
            if res.status_code != 200:
                continue
                
            raw = res.text.strip().splitlines()
            reader = csv.reader(raw)
            
            for row in reader:
                if not row or row[0] == "date" or "日時" in row[0]:
                    continue
                    
                if row[1] == room_name:
                    try:
                        dt = datetime.strptime(row[0], "%Y/%m/%d %H:%M:%S")
                        dt_clipped = dt.replace(second=0) 
                        
                        room_records.append({
                            'date': dt_clipped,
                            'room_temp': float(row[4]),
                            'room_humid': float(row[5])
                        })
                    except ValueError:
                        continue
        except Exception as e:
            continue

        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/365日] データの取得が完了...")

    df_room = pd.DataFrame(room_records)

    if df_room.empty:
        print("エラー: 室内データが取得できませんでした。")
        exit()

    # 1時間ごとの平均値にデータを丸める
    df_room['date'] = df_room['date'].dt.round('h')
    df_room_hourly = df_room.groupby('date').mean().reset_index()
    df_room_hourly = df_room_hourly.sort_values('date').reset_index(drop=True)

    output_filename = 'annual_room_data.csv'
    df_room_hourly.to_csv(output_filename, index=False)
    print(f"【成功】1年分の室温・湿度データを '{output_filename}' に保存しました。")
    return 'annual_room_data.csv'

# 2つのCSVファイルを1つにマージ
def load_and_merge_data(room_csv, outdoor_csv):
    print("1. CSVデータを読み込んで時間単位で結合しています...")
    df_room = pd.read_csv(room_csv)
    df_outdoor = pd.read_csv(outdoor_csv)
    
    df_room['date'] = pd.to_datetime(df_room['date'])
    df_outdoor['date'] = pd.to_datetime(df_outdoor['date'])
    
    # 横に結合
    merged_df = pd.merge(df_room, df_outdoor, on='date', how='inner')
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    return merged_df

# 特徴量とターゲットの作成
def create_features_and_targets(df):
    print("2. 機械学習用のデータ加工を行っています...")
    df['target_room_temp'] = df['room_temp'].shift(-24)
    df = df.dropna()
    
    X = df[['room_temp', 'room_humid', 'outdoor_temp', 'outdoor_humid']]
    Y = df['target_room_temp']
    return X, Y

# モデルの学習と時系列交差検証
def train_and_evaluate_model(X, Y):
    print("3. 時系列交差検証による学習を開始します...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        print(f"  パターン {fold+1} の平均予測誤差: {mae:.2f}℃")
        
    print("4. 最終モデルを全データで学習中...")
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X, Y)
    return final_model

# 不快指数を計算する
def get_discomfort(temp, humid):
    if temp is None or humid is None: return 0
    return 0.81 * temp + 0.01 * humid * (0.99 * temp - 14.3) + 46.3

@app.route("/", methods=["GET", "POST"])
def main():
    global trained_model

    # 部屋選択
    if request.method == "POST":
        room_name = request.form.get("room_name")
    else:
        room_name = ROOM_LIST[0]

    # リアルタイムデータ取得
    current_room_temp = None
    current_room_humid = None
    try:
        latest_res = requests.get(f"{LATEST_API_URL}?id={ID}&subscription-key={SUB_KEY}").json()
        for device in latest_res.get('devices', []):
            if device.get('name') == room_name:
                current_room_temp = float(device['temperature'])
                current_room_humid = float(device['humidity'])
    except:
        pass

    # バックアップ
    if current_room_temp is None:
        df_room_backup = pd.read_csv('annual_room_data.csv')
        current_room_temp = df_room_backup['room_temp'].iloc[-1]
        current_room_humid = df_room_backup['room_humid'].iloc[-1]

    discomfort = get_discomfort(current_room_temp, current_room_humid)

    # AI予測
    tomorrow_pred_text = "予測不可"
    if trained_model is not None:
        df_out = pd.read_csv('annual_outdoor_data.csv')
        current_outdoor_temp = df_out['outdoor_temp'].iloc[-1]
        current_outdoor_humid = df_out['outdoor_humid'].iloc[-1]

        current_data = [[current_room_temp, current_room_humid, current_outdoor_temp, current_outdoor_humid]]
        prediction = trained_model.predict(current_data)
        tomorrow_pred_text = f"{prediction[0]:.1f} ℃"

    return render_template(
        "index.html",
        rooms=ROOM_LIST,
        room_name=room_name,
        current_room_temp=f"{current_room_temp:.1f}",
        current_room_humid=f"{current_room_humid:.1f}",
        discomfort=f"{discomfort:.1f}",
        tomorrow_pred=tomorrow_pred_text
    )



if __name__ == "__main__":
    print("--- システム初期化・AI学習フェーズ ---")
    outdoor_file = get_outdoor_temphumid()
    room_file = get_room_temphumid(ROOM_LIST[0])
    
    # 取得したCSVファイルをマージして学習
    merged_df = load_and_merge_data(room_file, outdoor_file)
    X_data, y_data = create_features_and_targets(merged_df)
    trained_model = train_and_evaluate_model(X_data, y_data)
    
    print("--- 初期化完了。Flaskサーバーを起動します ---")
    app.run(debug=True, port=5000, use_reloader=False)