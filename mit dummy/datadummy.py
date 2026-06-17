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

# Record 230 ini sangat cocok untuk mengetes logika BRADIKARDIA di Web HTML-mu
RECORD_NAME = '233'
DATA_PATH = fr'C:\Users\Ghea Geltra\OneDrive\Documents\PBL SEM6 EKG\mit-bih-arrhythmia-database-1.0.0\mit-bih-arrhythmia-database-1.0.0\{RECORD_NAME}'

print("======================================================")
print(fr"🚀 INJEKTOR DYNAMIC BPM & FILTER: PASIEN {RECORD_NAME}")
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
# Mencari puncak gelombang R. Jarak minimal 35 sampel (0.28s) agar tidak dobel hitung.
peaks, _ = find_peaks(signal_filtered, distance=35, height=np.max(signal_filtered)*0.3)

# Membuat array kosong sepanjang data sinyal untuk menyimpan jejak BPM
bpm_track = np.zeros(len(signal_scaled))
current_bpm = 75 # Nilai default awal

last_peak = 0
for peak in peaks:
    if last_peak != 0:
        # Hitung jarak antar puncak (dalam satuan sampel)
        rr_interval_samples = peak - last_peak
        
        # Konversi ke waktu (detik) lalu ke BPM
        rr_interval_sec = rr_interval_samples / 125.0 
        current_bpm = int(60.0 / rr_interval_sec)
        
        # Batasi anomali ekstrem akibat noise
        if current_bpm > 220: current_bpm = 220
        if current_bpm < 30: current_bpm = 30
        
    # Terapkan nilai BPM ini dari puncak sebelumnya ke puncak saat ini
    bpm_track[last_peak:peak] = current_bpm
    last_peak = peak

# Isi sisa ujung sinyal dengan BPM terakhir
bpm_track[last_peak:] = current_bpm
# Isi celah di awal sinyal dengan BPM yang pertama kali terdeteksi
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
# 5. PROSES INJEKSI DENGAN WAKTU MUTLAK
# ==============================================================================
print("\nMulai menyuntikkan data ke Firebase secara Asynchronous (Anti-Lag)...")
print("Buka halaman web dashboard-mu sekarang! Tekan Ctrl+C di terminal ini untuk stop.\n")

chunk_size = 40
TARGET_CYCLE_SEC = 0.32 

try:
    start_time_global = time.time()
    
    for i in range(0, len(signal_scaled), chunk_size):
        start_loop_time = time.time()
        
        chunk = signal_scaled[i : i+chunk_size]
        if len(chunk) < chunk_size:
            break 
            
        # Ambil rata-rata BPM spesifik untuk 40 sampel potongan ini saja
        current_chunk_bpm = int(np.mean(bpm_track[i : i+chunk_size]))
        
        graph_value = [float(val) for val in chunk]
        
        # Masukkan BPM dinamis ke dalam payload!
        payload = { "bpm": current_chunk_bpm, "graph_value": graph_value }

        threading.Thread(target=send_to_firebase_background, args=(payload,)).start()
        
        elapsed_seconds = (i // chunk_size) * 0.32
        current_mm = int(elapsed_seconds // 60)
        current_ss = int(elapsed_seconds % 60)
        
        # Print BPM di terminal agar kamu bisa pantau perubahannya
        print(f"[{current_mm:02d}:{current_ss:02d}] 🚀 Mengirim 40 sampel | BPM: {current_chunk_bpm}")

        time_elapsed_in_loop = time.time() - start_loop_time
        dynamic_sleep = TARGET_CYCLE_SEC - time_elapsed_in_loop
        
        if dynamic_sleep > 0:
            time.sleep(dynamic_sleep) 
            
    print(f"\n✅ Injeksi 100% selesai! Waktu nyata: {(time.time() - start_time_global)/60:.1f} menit.")
    
except KeyboardInterrupt:
    print("\n⏹️ Injeksi dihentikan manual oleh pengguna (Ctrl+C).")
except Exception as e:
    print(f"\n⚠️ Terjadi kesalahan sistem: {e}")