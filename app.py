import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from engine import get_rolling_metrics

app = dash.Dash(__name__)
server = app.server

indices = {
    '^FCHI': 'CAC 40', '^GSPC': 'S&P 500', 'GC=F': 'GOLD (OR)', 
    'CL=F': 'CRUDE OIL', '^IXIC': 'NASDAQ', 'BTC-USD': 'BITCOIN'
}

app.layout = html.Div(className='main-container', children=[
    html.H1("Risk Engine Terminal", className='gold-title'),
    
    html.Div(className='card dropdown-container card-hover-sable', children=[
        html.Div("SELECT MARKET INTELLIGENCE", className='kpi-label', style={'textAlign': 'center', 'marginBottom': '10px'}),
        dcc.Dropdown(
            id='ticker-select',
            options=[{'label': v, 'value': k} for k, v in indices.items()],
            value='^FCHI',
            clearable=False,
            className='custom-dropdown'
        ),
    ]),

    html.Div(className='card card-hover-sable', children=[
        html.Div("CALCULATION WINDOW (DAYS)", className='kpi-label', style={'textAlign': 'center', 'marginBottom': '20px'}),
        dcc.Slider(
            id='window-slider',
            min=125, max=1000, step=25, value=504,
            marks={125: '6m', 252: '1y', 504: '2y', 756: '3y', 1000: '4y'},
        ),
    ]),

    html.Div(id='kpi-display', style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '30px'}),

    html.Div(className='card card-elevation-only', style={'backgroundColor': '#D6C5B0'}, children=[
        dcc.Graph(id='main-chart', config={'displayModeBar': False}),
    ]),

    html.Div(id='metrics-grid', style={
        'display': 'grid', 
        'gridTemplateColumns': 'repeat(4, 1fr)', 
        'gap': '20px',
        'marginBottom': '40px'
    }),
    
    html.Div(className='card section-analyses card-elevation-only', children=[
        html.H2("BASEL IV COMPLIANCE", className='kpi-label', style={'color': '#D6C5B0'}),
        html.P("VaR 99%: Daily prediction with rolling window analytics."),
        html.P("Expected Shortfall 97.5%: Conditional mean of tail losses."),
        html.P("Backtesting: Statistical validation vs real-world performance.")
    ])
])

@app.callback(
    [Output('main-chart', 'figure'),
     Output('kpi-display', 'children'),
     Output('metrics-grid', 'children')],
    [Input('ticker-select', 'value'),
     Input('window-slider', 'value')]
)
def update_dashboard(ticker, window_size):
    df = get_rolling_metrics(ticker, window=window_size)
    if df.empty: return go.Figure(), [], []

    exc_emp = len(df[df['Actual_Loss'] > df['VaR_Emp']])
    m_min, m_max = df['Actual_Loss'].min()*100, df['Actual_Loss'].max()*100
    m_mean, m_std = df['Actual_Loss'].mean()*100, df['Actual_Loss'].std()*100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Actual_Loss'], name='Daily Loss', line=dict(color='#3E2723', width=1), opacity=0.3))
    fig.add_trace(go.Scatter(x=df.index, y=df['VaR_Emp'], name='VaR 99%', line=dict(color='#E65100', width=4)))
    fig.add_trace(go.Scatter(x=df.index, y=df['ES_975'], name='ES 97.5%', line=dict(color='#3E2723', width=2, dash='dot')))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Corpus", color="#3E2723"),
        xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(62,39,35,0.1)'),
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center')
    )

    kpi_cards = [
        html.Div(className='card card-hover-marron', style={'backgroundColor': '#BDB3A2'}, children=[
            html.Div("EXCEPTIONS (N)", className='kpi-label'), 
            html.Div(exc_emp, className='kpi-value')
        ]),
        html.Div(className='card card-hover-marron', style={'backgroundColor': '#BDB3A2'}, children=[
            html.Div("MODEL QUALITY", className='kpi-label'), 
            html.Div("STABLE" if exc_emp <= 5 else "WEAK", className='kpi-value', 
                     style={'color': '#3E2723'}) # Changé en MARRON pour WEAK
        ])
    ]
    
    metrics_cards = [
        html.Div(className='metric-card', children=[html.Div("MIN LOSS", className='kpi-label', style={'fontSize': '0.75em'}), html.B(f"{m_min:.2f}%")]),
        html.Div(className='metric-card', children=[html.Div("MAX LOSS", className='kpi-label', style={'fontSize': '0.75em'}), html.B(f"{m_max:.2f}%")]),
        html.Div(className='metric-card', children=[html.Div("MEAN (μ)", className='kpi-label', style={'fontSize': '0.75em'}), html.B(f"{m_mean:.2f}%")]),
        html.Div(className='metric-card', children=[html.Div("STD DEV (σ)", className='kpi-label', style={'fontSize': '0.75em'}), html.B(f"{m_std:.2f}%")])
    ]
    
    return fig, kpi_cards, metrics_cards

if __name__ == '__main__':
    app.run(debug=True)