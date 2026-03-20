# dashboard.py 最小工作版本
import dash
from dash import dcc, html
import plotly.graph_objs as go
from collections import deque
import datetime

class RealtimeDashboard:
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
                # 测量值（取第一个特征，通常为电压）
                fig.add_trace(go.Scatter(
                    x=list(self.data['times']),
                    y=list(self.data['values']),
                    mode='lines',
                    name='测量值',
                    line=dict(color='blue')
                ))
                
                # 残差（如果有）
                if len(self.data['residuals']) > 0:
                    fig.add_trace(go.Scatter(
                        x=list(self.data['times']),
                        y=list(self.data['residuals']),
                        mode='lines',
                        name='残差',
                        line=dict(color='red'),
                        yaxis='y2'
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
            
            fig.update_layout(
                title='FDIA实时监控',
                xaxis_title='时间',
                yaxis=dict(title='测量值', side='left'),
                yaxis2=dict(title='残差', side='right', overlaying='y'),
                hovermode='x unified'
            )
            return fig
    
    def update_data(self, data):
        self.data['times'].append(data.get('timestamp', datetime.datetime.now()))
        
        # 测量值（取第一个特征）
        measurement = data.get('measurement')
        if measurement is not None:
            if hasattr(measurement, '__len__') and len(measurement) > 0:
                value = measurement[0]
            else:
                value = measurement
        else:
            value = 0
        self.data['values'].append(value)
        
        # 更新残差
        bdd_result = data.get('bdd_result', {})
        residual = bdd_result.get('residual_norm', 0)
        self.data['residuals'].append(residual)
        
        # 更新报警
        self.data['alarms'].append(bdd_result.get('is_attack', False))
    
    def run(self, debug=False, port=8050):
        try:
            print(f"尝试启动仪表板于 http://localhost:{port}")
            self.app.run(debug=debug, port=port, use_reloader=False)
        except Exception as e:
            print(f"仪表板启动失败: {e}")
            import traceback
            traceback.print_exc()