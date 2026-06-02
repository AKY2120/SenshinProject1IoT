# -*- coding: utf-8 -*-
import os
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
]

app = Flask(__name__)

# 各部屋の学習済みモデルを辞書型で保持
trained_models = {}

# 屋外の気温と湿度を取得しCSVファイルに保存
def get_outdoor_temphumid():
    output_filename = 'annual_outdoor_data.csv'
    
    # すでにファイルがあればAPIを叩かずに再利用
    if os.path.exists(output_filename):
        print("屋外データはローカルに存在するため、キャッシュを読み込みます。")
        return output_filename

    print("屋外の1年分の室温・温度データを取得中...")
    response = requests.get(weather_url)
    data = response.json()

    df = pd.DataFrame({
        'date': pd.to_datetime(data['hourly']['time']),
        'outdoor_temp': data['hourly']['temperature_2m'],
        'outdoor_humid': data['hourly']['relative_humidity_2m']
    })
    
    df.to_csv(output_filename, index=False)
    print("屋外のCSVファイルに保存しました。")
    return output_filename

# 部屋ごとの気温と湿度を取得しCSVファイルに保存
def get_room_temphumid(room_name):
    output_filename = f'annual_{room_name}_room_data.csv'
    
    # すでにファイルがあればAPIを叩かずに再利用
    if os.path.exists(output_filename):
        return output_filename

    room_records = []
    print(f"ネコリコAPIから{room_name}の1年分の室温・湿度データを取得中...")
    
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
        
        # APIサーバーへの負荷軽減
        time.sleep(0.05)

    df_room = pd.DataFrame(room_records)

    if df_room.empty:
        print(f"エラー: {room_name} の室内データが取得できませんでした。")
        return None

    # 1時間ごとの平均値にデータを丸める
    df_room['date'] = df_room['date'].dt.round('h')
    df_room_hourly = df_room.groupby('date').mean().reset_index()
    df_room_hourly = df_room_hourly.sort_values('date').reset_index(drop=True)

    df_room_hourly.to_csv(output_filename, index=False)
    print(f"1年分の室温・湿度データを '{output_filename}' に保存しました。")
    return output_filename

# 2つのCSVファイルを1つにマージ
def load_and_merge_data(room_csv, outdoor_csv):
    df_room = pd.read_csv(room_csv)
    df_outdoor = pd.read_csv(outdoor_csv)
    
    df_room['date'] = pd.to_datetime(df_room['date'])
    df_outdoor['date'] = pd.to_datetime(df_outdoor['date'])
    
    merged_df = pd.merge(df_room, df_outdoor, on='date', how='inner')
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    return merged_df

# 特徴量とターゲットの作成
def create_features_and_targets(df):
    df = df.copy() 
    df['target_room_temp'] = df['room_temp'].shift(-24)
    df = df.dropna()
    
    X = df[['room_temp', 'room_humid', 'outdoor_temp', 'outdoor_humid']]
    Y = df['target_room_temp']
    return X, Y

# モデルの学習と時系列交差検証
def train_and_evaluate_model(room_name, X, Y):
    print(f"{room_name}の時系列交差検証による学習を開始...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(Y_test, predictions)
        print(f"  パターン {fold+1} の平均予測誤差: {mae:.2f}℃")
        
    print(f"{room_name}の最終モデルを全データで学習中...")
    final_model = RandomForestRegressor(n_estimators=100, random_state=42)
    final_model.fit(X, Y)
    return final_model

# 不快指数を計算する
def get_discomfort(temp, humid):
    if temp is None or humid is None: return 0
    return 0.81 * temp + 0.01 * humid * (0.99 * temp - 14.3) + 46.3

@app.route("/api/data", methods=["POST"])
def api_data():
    room_name = request.form.get("room_name")

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

    # バックアップ読み込みのファイル名を動的に
    if current_room_temp is None:
        backup_file = f'annual_{room_name}_room_data.csv'
        if os.path.exists(backup_file):
            df_room_backup = pd.read_csv(backup_file)
            current_room_temp = df_room_backup['room_temp'].iloc[-1]
            current_room_humid = df_room_backup['room_humid'].iloc[-1]
        else:
            current_room_temp, current_room_humid = 22.0, 50.0  

    discomfort = get_discomfort(current_room_temp, current_room_humid)

    # 部屋に対応するモデルを呼び出して予測
    tomorrow_pred = "予測不可"
    model = trained_models.get(room_name)
    if model is not None:
        df_out = pd.read_csv('annual_outdoor_data.csv')
        current_outdoor_temp = df_out['outdoor_temp'].iloc[-1]
        current_outdoor_humid = df_out['outdoor_humid'].iloc[-1]

        current_data = [[current_room_temp, current_room_humid, current_outdoor_temp, current_outdoor_humid]]
        prediction = model.predict(current_data)
        tomorrow_pred = f"{prediction[0]:.1f} ℃"

    return {
        "temp": f"{current_room_temp:.1f}",
        "humid": f"{current_room_humid:.1f}",
        "discomfort": f"{discomfort:.1f}",
        "pred": tomorrow_pred
    }

@app.route("/", methods=["GET", "POST"])
def main():
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

    # バックアップ読み込みのファイル名を動的に
    if current_room_temp is None:
        backup_file = f'annual_{room_name}_room_data.csv'
        if os.path.exists(backup_file):
            df_room_backup = pd.read_csv(backup_file)
            current_room_temp = df_room_backup['room_temp'].iloc[-1]
            current_room_humid = df_room_backup['room_humid'].iloc[-1]
        else:
            current_room_temp, current_room_humid = 22.0, 50.0

    discomfort = get_discomfort(current_room_temp, current_room_humid)

    # 部屋に対応するモデルを呼び出して予測
    tomorrow_pred_text = "予測不可"
    model = trained_models.get(room_name)
    if model is not None:
        df_out = pd.read_csv('annual_outdoor_data.csv')
        current_outdoor_temp = df_out['outdoor_temp'].iloc[-1]
        current_outdoor_humid = df_out['outdoor_humid'].iloc[-1]

        current_data = [[current_room_temp, current_room_humid, current_outdoor_temp, current_outdoor_humid]]
        prediction = model.predict(current_data)
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
    print("システム初期化・AI学習フェーズ")
    outdoor_file = get_outdoor_temphumid()
    
    # すべての部屋でループ処理し、個別モデルとして保存
    for r_name in ROOM_LIST:
        room_file = get_room_temphumid(r_name)
        
        if room_file and os.path.exists(room_file):
            merged_df = load_and_merge_data(room_file, outdoor_file)
            X_data, y_data = create_features_and_targets(merged_df)
            
            # 各部屋専用のモデルを作成
            trained_models[r_name] = train_and_evaluate_model(r_name, X_data, y_data)
        else:
            print(f"エラー: {r_name} のデータが不足しているため、モデルを生成できませんでした。")
    
    print("初期化完了。Flaskサーバーを起動します")
    app.run(debug=True, port=5000, use_reloader=False)