import wfdb
import time
import requests
import threading
import numpy as np
from scipy.signal import resample, butter, filtfilt, find_peaks

# ==============================================================================
# 1. KONFIGURASI FIREBASE & PATH DATA
# ==============================================================================
FIREBASE_URL = "https://smart-ekg-tmj-6b-default-rtdb.asia-southeast1.firebasedatabase.app/monitoring/live_data.json"

# Menggunakan record pilihan untuk simulasi
RECORD_NAME = '207'
DATA_PATH = fr'C:\Users\Ghea Geltra\OneDrive\Documents\PBL SEM6 EKG\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\{RECORD_NAME}'

print("======================================================")
print(fr"🚀 INJEKTOR DYNAMIC BPM & FILTER (375-PTS CHUNK) ")
print(fr"PASIEN {RECORD_NAME}")
print("======================================================")

# ==============================================================================
# 2. MEMBACA & MENYIAPKAN DATA SECARA UTUH
# ==============================================================================
print("Membaca rekaman EKG asli (Full Record)...")
try:
    record = wfdb.rdrecord(DATA_PATH)
except FileNotFoundError:
    print(f"❌ ERROR: File data untuk pasien {RECORD_NAME} tidak ditemukan di path:")
    print(DATA_PATH)
    exit()

signal = record.p_signal[:, 0]

print("Melakukan resampling ke 125 Hz...")
new_len = int(len(signal) * 125 / record.fs)
signal_125hz = resample(signal, new_len)

print("Filtering dengan Butterworth Bandpass (0.5 - 40Hz)...")
TARGET_FS = 125
nyq = 0.5 * TARGET_FS
b, a = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
signal_filtered = filtfilt(b, a, signal_125hz)

print("Menyesuaikan amplitudo visual untuk Web HTML...")
signal_scaled = (signal_filtered * 150) + 715


# ==============================================================================
# 3. ALGORITMA PENGHITUNG BPM DINAMIS (R-PEAK DETECTION)
# ==============================================================================
print("Menghitung BPM dinamis dari R-to-R Interval...")
peaks, _ = find_peaks(signal_filtered, distance=35, height=np.max(signal_filtered)*0.3)

bpm_track = np.zeros(len(signal_scaled))
current_bpm = 75 

last_peak = 0
for peak in peaks:
    if last_peak != 0:
        rr_interval_samples = peak - last_peak
        rr_interval_sec = rr_interval_samples / 125.0 
        current_bpm = int(60.0 / rr_interval_sec)
        
        if current_bpm > 220: current_bpm = 220
        if current_bpm < 30: current_bpm = 30
        
    bpm_track[last_peak:peak] = current_bpm
    last_peak = peak

bpm_track[last_peak:] = current_bpm
if len(peaks) > 0:
    bpm_track[0:peaks[0]] = current_bpm


# ==============================================================================
# 4. FUNGSI TEMBAK FIREBASE (BACKGROUND THREAD)
# ==============================================================================
def send_to_firebase_background(payload):
    try:
        requests.put(FIREBASE_URL, json=payload, timeout=3)
    except Exception as e:
        pass 


# ==============================================================================
# 5. PROSES INJEKSI (CHUNK 375 POINTS)
# ==============================================================================
print("\nMulai menyuntikkan data ke Firebase (Paket 375 Data)...")
print("Buka halaman web dashboard-mu sekarang! Tekan Ctrl+C di terminal ini untuk stop.\n")

# PERUBAHAN UTAMA: Ukuran paket menjadi 375
chunk_size = 375

# Waktu tunda dinamis: 375 data pada 125Hz = 3 detik per paket
TARGET_CYCLE_SEC = 3.0 

try:
    start_time_global = time.time()
    
    for i in range(0, len(signal_scaled), chunk_size):
        start_loop_time = time.time()
        
        chunk = signal_scaled[i : i+chunk_size]
        if len(chunk) < chunk_size:
            break 
            
        current_chunk_bpm = int(np.mean(bpm_track[i : i+chunk_size]))
        graph_value = [float(val) for val in chunk]
        
        payload = { "bpm": current_chunk_bpm, "graph_value": graph_value }

        threading.Thread(target=send_to_firebase_background, args=(payload,)).start()
        
        # Hitung waktu (Setiap paket mewakili 3 detik)
        elapsed_seconds = (i // chunk_size) * TARGET_CYCLE_SEC
        current_mm = int(elapsed_seconds // 60)
        current_ss = int(elapsed_seconds % 60)
        
        print(f"[{current_mm:02d}:{current_ss:02d}] 🚀 Mengirim 375 sampel (1 Inference Window) | BPM: {current_chunk_bpm}")

        time_elapsed_in_loop = time.time() - start_loop_time
        dynamic_sleep = TARGET_CYCLE_SEC - time_elapsed_in_loop
        
        if dynamic_sleep > 0:
            time.sleep(dynamic_sleep) 
            
    print(f"\n✅ Injeksi 100% selesai! Waktu nyata: {(time.time() - start_time_global)/60:.1f} menit.")
    
except KeyboardInterrupt:
    print("\n⏹️ Injeksi dihentikan manual oleh pengguna (Ctrl+C).")
except Exception as e:
    print(f"\n⚠️ Terjadi kesalahan sistem: {e}")