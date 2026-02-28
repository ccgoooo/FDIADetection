# dashboard.py 简化版本
import dash
from dash import dcc, html
import plotly.graph_objs as go
from collections import deque
import numpy as np
import datetime

class RealtimeDashboard:
    """简化版实时仪表板"""
    
    def __init__(self, max_length=100):
        self.app = dash.Dash(__name__)
        self.data = {
            'times': deque(maxlen=max_length),
            'values': deque(maxlen=max_length),
            'residuals': deque(maxlen=max_length),
            'alarms': deque(maxlen=max_length)
        }
        self.setup_layout()
    
    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("FDIA检测监控"),
            dcc.Graph(id='live-graph'),
            dcc.Interval(id='interval', interval=1000)
        ])
        
        @self.app.callback(
            dash.dependencies.Output('live-graph', 'figure'),
            [dash.dependencies.Input('interval', 'n_intervals')]
        )
        def update_graph(n):
            fig = go.Figure()
            
            if len(self.data['times']) > 0:
                # 测量值
                fig.add_trace(go.Scatter(
                    x=list(self.data['times']),
                    y=list(self.data['values']),
                    mode='lines',
                    name='测量值'
                ))
                
                # 标记攻击点
                attack_times = [t for t, a in zip(self.data['times'], self.data['alarms']) if a]
                attack_values = [v for v, a in zip(self.data['values'], self.data['alarms']) if a]
                if attack_times:
                    fig.add_trace(go.Scatter(
                        x=attack_times,
                        y=attack_values,
                        mode='markers',
                        marker=dict(color='red', size=10),
                        name='攻击'
                    ))
            
            return fig
    
    def update_data(self, data):
        """更新数据"""
        self.data['times'].append(data.get('timestamp', datetime.datetime.now()))
        
        # 修复：检查measurement是否为None
        measurement = data.get('measurement')
        if measurement is not None:
            # 如果measurement是数组，取第一个值
            if hasattr(measurement, '__len__') and len(measurement) > 0:
                value = measurement[0]
            else:
                value = measurement
        else:
            value = 0
        
        self.data['values'].append(value)
        self.data['alarms'].append(data.get('bdd_result', {}).get('is_attack', False))
    def run(self, debug=False, port=8050):
        self.app.run_server(debug=debug, port=port)