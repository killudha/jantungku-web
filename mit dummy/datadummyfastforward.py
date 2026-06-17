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
RECORD_NAME = '230' 
DATA_PATH = fr'C:\Users\Dhanny\Documents\CODING\ekg\mit-bih-arrhythmia-database-1.0.0\{RECORD_NAME}'

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
# 5. PROSES INJEKSI DENGAN FITUR FAST-FORWARD (DATA ASLI 100%)
# ==============================================================================
print("\nMulai menyuntikkan data ke Firebase secara Asynchronous (Anti-Lag)...")

chunk_size = 40
TARGET_CYCLE_SEC = 0.32 

# ---------------------------------------------------------
# MAU MULAI DARI MENIT KE BERAPA? (Ubah angka ini untuk mencari Bradikardia)
START_MINUTE = 25
# ---------------------------------------------------------

# Konversi menit ke index array (1 menit = 60 detik * 125 Hz = 7500 index)
start_index = int(START_MINUTE * 60 * 125)

print(f"⏩ FAST FORWARD: Memulai data langsung dari menit ke-{START_MINUTE}...")

try:
    start_time_global = time.time()
    
    # PERHATIKAN: Loop sekarang dimulai dari 'start_index', bukan dari 0
    for i in range(start_index, len(signal_scaled), chunk_size):
        start_loop_time = time.time()
        
        chunk = signal_scaled[i : i+chunk_size]
        if len(chunk) < chunk_size:
            break 
            
        # Ambil rata-rata BPM spesifik (ASLI MURNI) untuk 40 sampel potongan ini saja
        current_chunk_bpm = int(np.mean(bpm_track[i : i+chunk_size]))
        
        graph_value = [float(val) for val in chunk]
        
        # Payload ASLI
        payload = { "bpm": current_chunk_bpm, "graph_value": graph_value }

        threading.Thread(target=send_to_firebase_background, args=(payload,)).start()
        
        # Hitung waktu rekaman asli di MIT-BIH (bukan waktu nyata laptop)
        mit_bih_seconds = (i // 125)
        mit_bih_mm = int(mit_bih_seconds // 60)
        mit_bih_ss = int(mit_bih_seconds % 60)
        
        print(f"[Waktu Pasien {mit_bih_mm:02d}:{mit_bih_ss:02d}] 🚀 Kirim 40 sampel | BPM ASLI: {current_chunk_bpm}")

        time_elapsed_in_loop = time.time() - start_loop_time
        dynamic_sleep = TARGET_CYCLE_SEC - time_elapsed_in_loop
        
        if dynamic_sleep > 0:
            time.sleep(dynamic_sleep) 
            
    print(f"\n✅ Injeksi 100% selesai!")
    
except KeyboardInterrupt:
    print("\n⏹️ Injeksi dihentikan manual oleh pengguna (Ctrl+C).")
except Exception as e:
    print(f"\n⚠️ Terjadi kesalahan sistem: {e}")