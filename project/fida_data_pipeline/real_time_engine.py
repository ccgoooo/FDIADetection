from multiprocessing import Process, Queue, Event
import threading
import queue

class RealTimeDataEngine:
    """
    实时数据流处理引擎
    生产者-消费者模式
    """
    def __init__(self, buffer_size=1000):
        self.data_queue = Queue(maxsize=buffer_size)
        self.stop_event = Event()
        self.producers = []
        self.consumers = []
        self.metrics = {
            'produced': 0,
            'consumed': 0,
            'dropped': 0,
            'latency': []
        }
    
    def start_producer(self, generator_func, interval_ms=100):
        """启动生产者线程"""
        def producer_loop():
            while not self.stop_event.is_set():
                try:
                    # 生成数据
                    data = generator_func()
                    timestamp = time.time()
                    
                    # 放入队列（非阻塞）
                    try:
                        self.data_queue.put_nowait((timestamp, data))
                        self.metrics['produced'] += 1
                    except queue.Full:
                        # 队列满，丢弃数据
                        self.metrics['dropped'] += 1
                    
                    # 控制采样间隔
                    time.sleep(interval_ms / 1000)
                except Exception as e:
                    print(f"Producer error: {e}")
        
        thread = threading.Thread(target=producer_loop)
        thread.daemon = True
        thread.start()
        self.producers.append(thread)
    
    def start_consumer(self, consumer_func):
        """启动消费者线程"""
        def consumer_loop():
            while not self.stop_event.is_set() or not self.data_queue.empty():
                try:
                    # 获取数据（带超时）
                    timestamp, data = self.data_queue.get(timeout=0.5)
                    
                    # 处理数据
                    start_time = time.time()
                    result = consumer_func(data, timestamp)
                    latency = (time.time() - start_time) * 1000  # ms
                    
                    # 记录指标
                    self.metrics['consumed'] += 1
                    self.metrics['latency'].append(latency)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Consumer error: {e}")
        
        thread = threading.Thread(target=consumer_loop)
        thread.daemon = True
        thread.start()
        self.consumers.append(thread)