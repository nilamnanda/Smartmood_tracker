import csv
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

DATA_FILE = 'mood_activity_data.csv'
MIN_DAYS_FOR_GRAPH = 3

# Aturan permainan
def show_game_rules():
    print("""
    ==========================
    SELAMAT DATANG DI MOOD TRACKER!
    Sebelum mulai, simak aturan mainnya dulu ya:
    1. Kamu wajib input mood dan aktivitas tiap hari.
    2. Minimal input selama 3 hari supaya bisa lihat grafik.
    3. Aku bakal coba tebak mood kamu berdasarkan aktivitas sebelumnya.
    4. Aku kasih saran aktivitas biar mood kamu makin oke.
    ==========================
    """)
    input("Tekan Enter kalau sudah paham dan siap lanjut...")

# Load data CSV
def load_data():
    data = []
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    return data

# Simpan data CSV
def save_data(data):
    fieldnames = ['username', 'date', 'mood', 'activity']
    with open(DATA_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

# Input data mood & activity per user per date
def input_data(username):
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"\nHalo {username}, input mood dan aktivitasmu untuk hari ini ({today}):")
    mood = input("Mood kamu hari ini (contoh: happy, sad, stressed, excited): ").strip().lower()
    activity = input("Aktivitas yang kamu lakukan hari ini (contoh: belajar, nonton, olahraga): ").strip().lower()
    return {'username': username, 'date': today, 'mood': mood, 'activity': activity}

# Filter data per user
def filter_data_by_user(data, username):
    return [d for d in data if d['username'] == username]

# Visualisasi grafik mood harian
def plot_mood_graph(data, username):
    if len(data) < MIN_DAYS_FOR_GRAPH:
        print(f"Maaf {username}, kamu harus input minimal {MIN_DAYS_FOR_GRAPH} hari dulu baru bisa lihat grafik mood.")
        return
    
    # Urutkan data berdasarkan tanggal
    data_sorted = sorted(data, key=lambda x: x['date'])
    
    # Mood encoding untuk grafik
    moods = [d['mood'] for d in data_sorted]
    dates = [d['date'] for d in data_sorted]
    
    le = LabelEncoder()
    mood_encoded = le.fit_transform(moods)
    
    plt.figure(figsize=(10,5))
    plt.plot(dates, mood_encoded, marker='o', linestyle='-', color='purple')
    plt.title(f'Grafik Mood Harian - {username}')
    plt.xlabel('Tanggal')
    plt.ylabel('Mood (encoded)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Model prediksi mood berdasarkan aktivitas
def train_predict_model(data, username):
    user_data = filter_data_by_user(data, username)
    if len(user_data) < MIN_DAYS_FOR_GRAPH:
        print(f"Maaf {username}, data belum cukup untuk prediksi mood (minimal {MIN_DAYS_FOR_GRAPH} hari).")
        return None
    
    # Prepare training data
    X_raw = [d['activity'] for d in user_data]
    y_raw = [d['mood'] for d in user_data]
    
    le_activity = LabelEncoder()
    le_mood = LabelEncoder()
    
    X = le_activity.fit_transform(X_raw).reshape(-1,1)
    y = le_mood.fit_transform(y_raw)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    
    # Predict mood for today's activity
    today_activity = input("Masukkan aktivitasmu hari ini untuk prediksi mood: ").strip().lower()
    X_test = le_activity.transform([today_activity]).reshape(-1,1) if today_activity in le_activity.classes_ else None
    
    if X_test is None:
        print("Aktivitas baru, tidak ada prediksi mood untuk aktivitas ini.")
        return None
    
    pred_encoded = model.predict(X_test)[0]
    pred_mood = le_mood.inverse_transform([pred_encoded])[0]
    print(f"Prediksi mood kamu berdasarkan aktivitas '{today_activity}': {pred_mood}")
    return pred_mood

# Rekomendasi aktivitas berdasarkan mood historis
def recommend_activity(data, username):
    user_data = filter_data_by_user(data, username)
    if not user_data:
        print("Belum ada data untuk rekomendasi aktivitas.")
        return
    
    mood_activity_map = defaultdict(lambda: defaultdict(int))
    for d in user_data:
        mood_activity_map[d['mood']][d['activity']] += 1
    
    # Cari mood hari ini
    today = datetime.now().strftime('%Y-%m-%d')
    today_entry = next((d for d in user_data if d['date'] == today), None)
    if not today_entry:
        print("Input dulu mood hari ini supaya bisa dapat rekomendasi.")
        return
    
    today_mood = today_entry['mood']
    
    # Rekomendasi aktivitas yang sering dilakukan saat mood lain yang lebih positif
    positive_moods = ['happy', 'excited', 'relaxed', 'calm']
    recommended = []
    for mood in positive_moods:
        if mood != today_mood and mood in mood_activity_map:
            # Ambil aktivitas paling sering untuk mood positif selain mood hari ini
            top_activity = max(mood_activity_map[mood], key=mood_activity_map[mood].get)
            if top_activity not in recommended:
                recommended.append(top_activity)
    
    if recommended:
        print(f"Rekomendasi aktivitas untuk memperbaiki mood '{today_mood}': {', '.join(recommended)}")
    else:
        print("Belum cukup data aktivitas untuk rekomendasi.")

# Main program loop
def main():
    show_game_rules()
    
    print("=== MOOD TRACKER MULTI USER ===")
    username = input("Masukkan username kamu: ").strip().lower()
    
    data = load_data()
    
    # Cek apakah user sudah input untuk hari ini
    user_data = filter_data_by_user(data, username)
    today = datetime.now().strftime('%Y-%m-%d')
    today_input = next((d for d in user_data if d['date'] == today), None)
    
    if today_input:
        print(f"Kamu sudah input data untuk hari ini ({today}).")
    else:
        new_entry = input_data(username)
        data.append(new_entry)
        save_data(data)
        print("Data berhasil disimpan!")
    
    # Menu pilihan
    while True:
        print("\nMenu:")
        print("1. Lihat grafik mood harian")
        print("2. Prediksi mood hari ini berdasarkan aktivitas")
        print("3. Rekomendasi aktivitas")
        print("4. Input ulang data hari ini")
        print("5. Keluar")
        choice = input("Pilih menu (1-5): ").strip()
        
        if choice == '1':
            user_data = filter_data_by_user(data, username)
            plot_mood_graph(user_data, username)
        elif choice == '2':
            train_predict_model(data, username)
        elif choice == '3':
            recommend_activity(data, username)
        elif choice == '4':
            # Input ulang hari ini
            data = [d for d in data if not (d['username'] == username and d['date'] == today)]
            new_entry = input_data(username)
            data.append(new_entry)
            save_data(data)
            print("Data hari ini berhasil diperbarui!")
        elif choice == '5':
            print("Makasih sudah pakai Mood Tracker. Semoga harimu menyenangkan!")
            break
        else:
            print("Pilihan tidak valid, coba lagi ya.")

if __name__ == "__main__":
    main()
