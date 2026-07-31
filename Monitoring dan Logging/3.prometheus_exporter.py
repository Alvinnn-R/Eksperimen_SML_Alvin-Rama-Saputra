"""
prometheus_exporter.py - Prometheus Exporter untuk Monitoring Model ML
Proyek Akhir MSML - Alvin Rama Saputra
"""
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time, random, threading

# Metriks 1: Total request prediksi
REQUEST_COUNT = Counter(
    'model_prediction_requests_total',
    'Total jumlah request prediksi yang diterima'
)

# Metriks 2: Latensi prediksi
REQUEST_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Waktu latensi prediksi model dalam detik',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Metriks 3: Error prediksi
ERROR_COUNT = Counter(
    'model_prediction_errors_total',
    'Total jumlah error saat prediksi'
)

# Metriks 4: Hasil prediksi per label
PREDICTION_RESULT = Counter(
    'model_prediction_result_total',
    'Total prediksi berdasarkan hasil',
    ['result']
)

# Metriks 5: Confidence score
MODEL_CONFIDENCE = Histogram(
    'model_prediction_confidence',
    'Confidence score prediksi model',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Metriks 6: Active requests
ACTIVE_REQUESTS = Gauge(
    'model_active_requests',
    'Jumlah request yang sedang diproses'
)

# Metriks 7: Status model
MODEL_STATUS = Gauge(
    'model_status',
    'Status model (1 = running, 0 = stopped)'
)

def simulate_prediction():
    """Simulasi prediksi model untuk menghasilkan metriks"""
    while True:
        REQUEST_COUNT.inc()
        ACTIVE_REQUESTS.inc()
        MODEL_STATUS.set(1)
        
        latency = random.uniform(0.01, 0.5)
        REQUEST_LATENCY.observe(latency)
        time.sleep(latency)
        
        if random.random() < 0.05:
            ERROR_COUNT.inc()
        else:
            confidence = random.uniform(0.5, 0.99)
            MODEL_CONFIDENCE.observe(confidence)
            if random.random() > 0.4:
                PREDICTION_RESULT.labels(result='diabetes').inc()
            else:
                PREDICTION_RESULT.labels(result='normal').inc()
        
        ACTIVE_REQUESTS.dec()
        time.sleep(random.uniform(1, 3))

if __name__ == '__main__':
    start_http_server(8001)
    print("=" * 50)
    print("Prometheus Exporter berjalan pada port 8001")
    print("Akses metriks di: http://localhost:8001/metrics")
    print("=" * 50)
    
    sim_thread = threading.Thread(target=simulate_prediction, daemon=True)
    sim_thread.start()
    
    print("\nSimulasi prediksi berjalan...")
    print("Tekan Ctrl+C untuk berhenti.\n")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExporter dihentikan.")
