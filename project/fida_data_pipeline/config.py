CONFIG = {
    'simulation': {
        'network': 'ieee14',
        'sampling_rate': 10,  # Hz
        'duration_hours': 24,
        'noise_level': 0.01
    },
    'communication': {
        'protocol': 'iec60870-5-104',
        'latency_ms': 100,
        'packet_loss_rate': 0.01
    },
    'bdd_detection': {
        'method': 'wls',  # 加权最小二乘
        'threshold': 0.05,
        'max_iterations': 10
    },
    'visualization': {
        'dashboard_port': 8050,
        'update_interval_ms': 1000,
        'buffer_size': 1000
    }
}