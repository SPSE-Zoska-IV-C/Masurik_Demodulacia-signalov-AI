import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output, State, callback, DiskcacheManager, ALL, no_update, ctx
import diskcache

from Generation.generator_FINAL import generate_bulk, load_complex_file
from Augmentation.augmentation_FINAL import apply_augmentation_bulk
from Training.training_denoise_FINAL import get_available_models, train_model_process as train_denoise_process
from Testing.evaluation_FINAL import (
    run_evaluation,
    run_evaluation_bits,
    list_evaluable_files,
    list_weight_files,
    get_available_models as eval_get_available_models,
    load_evaluation_model,
    _load_and_pad,
    _load_and_pad_normalised,
    _AUGMENTED_SAMPLES,
    ask_demodulate,
    parse_txt_metadata,
    _resample_to_original,
)
from Training.training_general_FINAL import (
    train_model_process as train_bits_process,
)

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    background_callback_manager=background_callback_manager,
    suppress_callback_exceptions=True
)


LAYER_TYPES = ['Conv1d', 'MaxPool1d', 'Linear', 'Flatten', 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid']

_PARAM_META = {
    'Conv1d':    ('In Ch',  'Out Ch', 'Kernel', 'Padding'),
    'MaxPool1d': ('Kernel', None,     None,     None),
    'Linear':    ('In',     'Out',    None,     None),
    'Flatten':   (None,     None,     None,     None),
    'ReLU':      (None,     None,     None,     None),
    'LeakyReLU': (None,     None,     None,     None),
    'Tanh':      (None,     None,     None,     None),
    'Sigmoid':   (None,     None,     None,     None),
}

_PARAM_DEFAULTS = {
    'Conv1d':    (2,    32,   7,    3),
    'MaxPool1d': (2,    None, None, None),
    'Linear':    (512,  256,  None, None),
    'Flatten':   (None, None, None, None),
    'ReLU':      (None, None, None, None),
    'LeakyReLU': (None, None, None, None),
    'Tanh':      (None, None, None, None),
    'Sigmoid':   (None, None, None, None),
}



def make_input_group(label, id_min, id_max, val_min, val_max, step=None):
    return dbc.Row([
        dbc.Col(html.Label(label, style={"fontSize": "0.85rem"}), width=4, className="align-self-center"),
        dbc.Col(dbc.Input(id=id_min, type="number", value=val_min, step=step, size="sm"), width=4),
        dbc.Col(dbc.Input(id=id_max, type="number", value=val_max, step=step, size="sm"), width=4),
    ], className="mb-2")


def file_manager_layout(tab_prefix):
    return dbc.Card([
        dbc.CardHeader(dbc.Row([
            dbc.Col(html.B("Náhľad"), width=8),
            dbc.Col(dbc.Button("Refresh", id=f"{tab_prefix}-refresh-btn", size="sm",
                               color="secondary", className="w-100"), width=4)
        ])),
        dbc.CardBody([
            dcc.Dropdown(id=f"{tab_prefix}-file-dropdown", placeholder="Vyber súbor", className="mb-3"),
            dcc.Graph(id=f"{tab_prefix}-waveform-graph", style={"height": "450px"})
        ])
    ], className="shadow-sm")



def render_generation_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Konfigurácia")),
                dbc.CardBody([
                    html.Label("Výstupný priečinok", className="small"),
                    dbc.Input(id="gen-folder", type="text", value="./ask_dataset", className="mb-2", size="sm"),
                    dbc.Row([
                        dbc.Col([html.Label("Count", className="small"),
                                 dbc.Input(id="gen-num-files", type="number", value=100, size="sm")], width=6),
                        dbc.Col([html.Label("Sample Rate", className="small"),
                                 dbc.Input(id="gen-samp-rate", type="number", value=1280000, size="sm")], width=6),
                    ], className="mb-3"),
                    html.Div([
                        html.B("Rozsahy", className="small text-muted"),
                        html.Hr(className="my-1"),
                        make_input_group("Bity", "gen-bits-min", "gen-bits-max", 1, 32, 1),
                        make_input_group("Vzorky na bit", "gen-spb-min", "gen-spb-max", 32, 256, 1),
                        make_input_group("Frekvencia", "gen-freq-min", "gen-freq-max", 100000, 1000000, 1000),
                        make_input_group("Šum", "gen-noise-min", "gen-noise-max", 0.1, 2.0, 0.1),
                    ]),
                    dbc.Button("Generovať Dataset", id="gen-run-btn", color="primary", className="w-100 mt-3"),
                    dbc.Progress(id="gen-progress", value=0, striped=True, animated=True,
                                 className="mt-3", style={"height": "10px"}),
                    html.Div(id="gen-status", className="small text-center mt-2 text-success")
                ])
            ], className="shadow-sm mb-3")
        ], width=4),
        dbc.Col(file_manager_layout("gen"), width=8)
    ])


def render_augmentation_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Augmentácia")),
                dbc.CardBody([
                    html.Label("Cieľový priečinok dataset-u", className="small"),
                    dbc.Input(id="aug-folder", type="text", value="./ask_dataset", className="mb-3", size="sm"),
                    html.Label("Cieľový počet vzoriek", className="small"),
                    dbc.Input(id="aug-target-len", type="number", value=9000, className="mb-3", size="sm"),
                    html.Label("Spôsob procesovania", className="small"),
                    dcc.RadioItems(
                        id="aug-method",
                        options=[
                            {'label': ' Padding / Trimming', 'value': 'pad'},
                            {'label': ' Fourier Resampling', 'value': 'resample'}
                        ],
                        value='pad',
                        labelStyle={'display': 'block', 'marginBottom': '10px'}
                    ),
                    dbc.Button("Aplikovať", id="aug-run-btn", color="warning", className="w-100 mt-2"),
                    dbc.Progress(id="aug-progress", value=0, striped=True, animated=True,
                                 className="mt-3", style={"height": "10px"}),
                    html.Div(id="aug-status", className="small text-center mt-2 text-primary")
                ])
            ], className="shadow-sm mb-3")
        ], width=4),
        dbc.Col(file_manager_layout("aug"), width=8)
    ])


def create_layer_card(index, layer_type='Conv1d', pa=None, pb=None, pc=None, pd=None):
    meta = _PARAM_META.get(layer_type, (None, None, None, None))
    defs = _PARAM_DEFAULTS.get(layer_type, (None, None, None, None))
    vals = [
        pa if pa is not None else defs[0],
        pb if pb is not None else defs[1],
        pc if pc is not None else defs[2],
        pd if pd is not None else defs[3],
    ]

    def _inp(key, val, label):
        disabled = label is None
        return dbc.Col(
            dbc.Input(
                id={'type': f'arch-p{key}', 'index': index},
                type="number", value=val, size="sm",
                disabled=disabled,
                placeholder=label or "—",
                style={"backgroundColor": "#efefef" if disabled else ""}
            ),
            width=2
        )

    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Small(f"#{index}", className="text-muted"),
                    dcc.Dropdown(
                        id={'type': 'arch-layer-type', 'index': index},
                        options=[{'label': t, 'value': t} for t in LAYER_TYPES],
                        value=layer_type, clearable=False,
                    )
                ], width=3),
                _inp('a', vals[0], meta[0]),
                _inp('b', vals[1], meta[1]),
                _inp('c', vals[2], meta[2]),
                _inp('d', vals[3], meta[3]),
                dbc.Col(
                    dbc.Button("✕", id={'type': 'arch-del-layer', 'index': index},
                               color="outline-danger", size="sm", className="w-100",
                               style={"marginTop": "18px"}),
                    width=1
                ),
            ], className="align-items-end")
        ], className="p-2")
    ], className="mb-1 shadow-sm")


def render_architecture_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(dbc.Row([
                    dbc.Col(html.B("Vytváranie modelu"), width=8),
                    dbc.Col(dbc.Button("+ Pridať vrstvu", id="arch-add-layer-btn", size="sm",
                                       color="success", className="w-100"), width=4)
                ])),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("počet vstupných vzoriek", className="small text-muted"),
                            dbc.Input(id="arch-input-samples", type="number", value=9000, size="sm"),
                        ], width=6),
                    ], className="mb-3"),
                    html.Hr(className="my-2"),
                    dbc.Row([
                        dbc.Col(html.Small("Type", className="text-muted fw-bold"), width=3),
                        dbc.Col(html.Small("P1", className="text-muted fw-bold"), width=2),
                        dbc.Col(html.Small("P2", className="text-muted fw-bold"), width=2),
                        dbc.Col(html.Small("P3", className="text-muted fw-bold"), width=2),
                        dbc.Col(html.Small("P4", className="text-muted fw-bold"), width=2),
                        dbc.Col(width=1),
                    ], className="mb-1 px-2"),
                    html.Div(id="arch-layers-container", children=[create_layer_card(0)]),
                    html.Hr(),
                    dbc.Button("Vygenerovať kód", id="arch-generate-btn", color="primary", className="w-100")
                ])
            ], className="shadow-sm mb-3")
        ], width=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Náhľad")),
                dbc.CardBody([
                    html.Label("Názov súboru", className="small"),
                    dbc.Input(id="arch-model-name", type="text", value="Model_v1", size="sm", className="mb-2"),
                    dcc.Textarea(id="arch-code-output",
                                 style={"width": "100%", "height": "450px", "fontFamily": "monospace",
                                        "backgroundColor": "#f8f9fa"}, readOnly=True),
                    dbc.Button("Uložiť", id="arch-save-btn", color="info", className="w-100 mt-3"),
                    html.Div(id="arch-save-status", className="small text-center mt-2 text-info")
                ])
            ], className="shadow-sm")
        ], width=5)
    ])


def render_training_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Konfigurácia")),
            dbc.CardBody([
                html.Label("Režim tréningu", className="small"),
                dcc.RadioItems(
                    id="train-mode",
                    options=[
                        {"label": " Odšumovanie", "value": "denoise"},
                        {"label": " Predikcia bitov", "value": "bits"},
                    ],
                    value="denoise",
                    labelStyle={"display": "block", "marginBottom": "8px"},
                    className="mb-3",
                ),

                html.Label("Tréningový dataset", className="small"),
                    dbc.Input(id="train-folder", type="text",
                              value="./ask_dataset", className="mb-3", size="sm"),

                    html.Label("Validačný dataset", className="small"),
                    dbc.Input(id="train-val-folder", type="text", value="./ask_dataset",
                              className="mb-3", size="sm"),

                    html.Label("Architektúra modelu", className="small"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id="train-model-dropdown",
                                             placeholder="Vyber model"), width=9),
                        dbc.Col(dbc.Button("↻", id="train-refresh-models-btn",
                                           color="secondary", size="sm",
                                           className="w-100"), width=3),
                    ], className="mb-3"),

                    dbc.Row([
                        dbc.Col([html.Label("Epochy", className="small"),
                                 dbc.Input(id="train-epochs", type="number",
                                           value=15, size="sm")], width=6),
                        dbc.Col([html.Label("Batch Size", className="small"),
                                 dbc.Input(id="train-batch", type="number",
                                           value=16, size="sm")], width=6),
                    ], className="mb-3"),

                    html.Label("Learning Rate", className="small"),
                    dbc.Input(id="train-lr", type="number", value=0.001,
                              step=0.0001, size="sm", className="mb-3"),

                    html.Label("Uložiť ako (.pth)", className="small"),
                    dbc.Input(id="train-save-name", type="text",
                              value="denoiser_v1.pth", size="sm", className="mb-3"),

                    dbc.Button("Spustiť Tréning", id="train-run-btn",
                               color="danger", className="w-100"),
                    dbc.Progress(id="train-progress", value=0, striped=True,
                                 animated=True, className="mt-3",
                                 style={"height": "15px"}),
                    html.Div(id="train-status-text",
                             className="small text-center mt-2 fw-bold text-muted")
                ])
            ], className="shadow-sm")
        ], width=4),

        dbc.Col([
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.B("Train Loss")),
                    dbc.CardBody(dcc.Graph(id="train-loss-graph",
                                          style={"height": "250px"},
                                          config={"displayModeBar": False}))
                ], className="shadow-sm"), width=6),

                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.B("Train Bit Accuracy")),
                    dbc.CardBody(dcc.Graph(id="train-acc-graph",
                                          style={"height": "250px"},
                                          config={"displayModeBar": False}))
                ], className="shadow-sm"), width=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.B("Validation Loss")),
                    dbc.CardBody(dcc.Graph(id="train-val-loss-graph",
                                          style={"height": "250px"},
                                          config={"displayModeBar": False}))
                ], className="shadow-sm"), width=6),

                dbc.Col(dbc.Card([
                    dbc.CardHeader(html.B("Validation Bit Accuracy")),
                    dbc.CardBody(dcc.Graph(id="train-val-acc-graph",
                                          style={"height": "250px"},
                                          config={"displayModeBar": False}))
                ], className="shadow-sm"), width=6),
            ]),

            dcc.Store(id="train-loss-store", data={
                "steps":          [],
                "losses":         [],
                "acc_steps":      [],
                "acc_values":     [],
                "val_loss_steps": [],
                "val_losses":     [],
                "val_acc_steps":  [],
                "val_acc_values": [],
            })
        ], width=8)
    ])

@callback(
    Output("train-status-text", "children"),
    Output("train-loss-store", "data"),
    Input("train-run-btn", "n_clicks"),
    State("train-mode", "value"),
    State("train-folder", "value"),
    State("train-val-folder", "value"),
    State("train-model-dropdown", "value"),
    State("train-batch", "value"),
    State("train-epochs", "value"),
    State("train-lr", "value"),
    State("train-save-name", "value"),
    background=True,
    running=[(Output("train-run-btn", "disabled"), True, False)],
    progress=[
        Output("train-progress", "value"),
        Output("train-progress", "max"),
        Output("train-status-text", "children"),
        Output("train-loss-store", "data"),
    ],
    prevent_initial_call=True
)
def run_train(set_progress, n, mode, folder, val_folder, m_path,
              batch, epochs, lr, s_name):
    if not m_path:
        return "Vyber model", no_update

    if mode == "denoise":
        res = train_denoise_process(
            folder, m_path, batch, epochs, lr, s_name,
            val_path=val_folder or None,
            set_progress=set_progress
        )
    else:
        res = train_bits_process(
            folder, m_path, batch, epochs, lr, s_name,
            val_path=val_folder or None,
            set_progress=set_progress
        )

    return res, no_update


@callback(
    Output("train-loss-graph",     "figure"),
    Output("train-acc-graph",      "figure"),
    Output("train-val-loss-graph", "figure"),
    Output("train-val-acc-graph",  "figure"),
    Input("train-loss-store", "data")
)
def update_train_graphs(data):
    def _placeholder(msg):
        fig = go.Figure()
        fig.add_annotation(text=msg, xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=13, color="#bbb"))
        fig.update_layout(template="plotly_white",
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          margin=dict(l=20, r=20, t=10, b=20))
        return fig

    def _line_fig(x, y, color, x_label, y_label, y_range=None, ref_line=None):
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=color, width=2)
        ))
        layout = dict(template="plotly_white",
                      xaxis_title=x_label, yaxis_title=y_label,
                      margin=dict(l=40, r=20, t=10, b=40))
        if y_range:
            layout["yaxis"] = dict(range=y_range)
        fig.update_layout(**layout)
        if ref_line is not None:
            fig.add_hline(y=ref_line, line_dash="dot", line_color="#ccc")
        return fig

    if data["steps"]:
        loss_fig = _line_fig(data["steps"], data["losses"],
                             "#e74c3c", "Step", "MSE Loss")
    else:
        loss_fig = _placeholder("Tréning nedokončený")

    if data["acc_steps"]:
        acc_fig = _line_fig(data["acc_steps"], data["acc_values"],
                            "#27ae60", "Step", "Bit Accuracy (%)",
                            y_range=[0, 105], ref_line=100)
    else:
        acc_fig = _placeholder("Tréning nedokončený")

    if data["val_loss_steps"]:
        val_loss_fig = _line_fig(data["val_loss_steps"], data["val_losses"],
                                 "#8e44ad", "Step", "Val MSE Loss")
    else:
        val_loss_fig = _placeholder("Tréning nedokončený")

    if data["val_acc_steps"]:
        val_acc_fig = _line_fig(data["val_acc_steps"], data["val_acc_values"],
                                "#2980b9", "Step", "Val Bit Accuracy (%)",
                                y_range=[0, 105], ref_line=100)
    else:
        val_acc_fig = _placeholder("Tréning nedokončený")

    return loss_fig, acc_fig, val_loss_fig, val_acc_fig


def render_evaluation_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.B("Konfigurácia vyhodnotenia")),
                dbc.CardBody([
                    html.Label("Režim vyhodnotenia", className="small"),
                    dcc.RadioItems(
                        id="eval-mode",
                        options=[
                            {"label": " Odšumovanie", "value": "denoise"},
                            {"label": " Predikcia bitov", "value": "bits"},
                        ],
                        value="denoise",
                        labelStyle={"display": "block", "marginBottom": "8px"},
                        className="mb-3",
                    ),

                    html.Label("Dataset", className="small"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="eval-folder", type="text",
                                         value="./ask_dataset", size="sm"), width=9),
                        dbc.Col(dbc.Button("↻", id="eval-refresh-files-btn", color="secondary",
                                           size="sm", className="w-100"), width=3),
                    ], className="mb-3"),

                    html.Label("Signálový súbor", className="small"),
                    dcc.Dropdown(id="eval-file-dropdown",
                                 placeholder="Vyber súbor", className="mb-3"),

                    html.Hr(),

                    html.Label("Architektúra modelu", className="small"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id="eval-arch-dropdown",
                                             placeholder="Vyber architektúru"), width=9),
                        dbc.Col(dbc.Button("↻", id="eval-refresh-arch-btn", color="secondary",
                                           size="sm", className="w-100"), width=3),
                    ], className="mb-3"),

                    html.Label("Váhy modelu", className="small"),
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(id="eval-weights-dropdown",
                                             placeholder="Vyber .pth"), width=9),
                        dbc.Col(dbc.Button("↻", id="eval-refresh-weights-btn", color="secondary",
                                           size="sm", className="w-100"), width=3),
                    ], className="mb-3"),

                    html.Hr(),

                    dbc.Button("Spustiť vyhodnotenie", id="eval-run-btn",
                               color="success", className="w-100 mt-1"),

                    dbc.Progress(id="eval-progress", value=0, striped=True, animated=True,
                                 className="mt-3", style={"height": "10px"}),

                    html.Div(id="eval-status", className="small text-center mt-2 text-success"),

                    dcc.Store(id="eval-result-store", data=None),

                    html.Hr(),

                    html.Label("Vyhodnotiť celý dataset", className="small fw-bold"),
                    dbc.Button("Spustiť batch vyhodnotenie", id="eval-batch-run-btn",
                               color="primary", className="w-100 mt-1"),

                    dbc.Progress(id="eval-batch-progress", value=0, striped=True, animated=True,
                                 className="mt-3", style={"height": "10px"}),

                    html.Div(id="eval-batch-status", className="small text-center mt-2"),

                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div("—", id="eval-batch-trad-result",
                                             className="text-center fs-4 fw-bold text-primary"),
                                    html.Div("Tradičná metóda", className="text-center small text-muted"),
                                ], width=6),
                                dbc.Col([
                                    html.Div("—", id="eval-batch-model-result",
                                             className="text-center fs-4 fw-bold text-success"),
                                    html.Div("Model", className="text-center small text-muted"),
                                ], width=6),
                            ])
                        ], className="py-2")
                    ], className="mt-3 shadow-sm", id="eval-batch-results-card",
                       style={"display": "none"}),
                ])
            ], className="shadow-sm")
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    dbc.Tabs(
                        id="eval-graph-tabs",
                        active_tab="tab-signals",
                        children=[
                            dbc.Tab(label="Signály (IQ)", tab_id="tab-signals"),
                            dbc.Tab(label="ASK Demodulácia", tab_id="tab-demod"),
                            dbc.Tab(label="Porovnanie bitov", tab_id="tab-bits"),
                        ]
                    )
                ),
                dbc.CardBody([
                    dcc.Graph(id="eval-main-graph", style={"height": "600px"}),
                ])
            ], className="shadow-sm")
        ], width=8)
    ])



app.layout = html.Div([
    dbc.NavbarSimple(brand="Demodulácia rádiových signálov",
                     color="dark", dark=True, className="mb-4"),
    dbc.Container(id="page-content", fluid=True, style={"paddingBottom": "100px"}),
    html.Div([
        dbc.Tabs(
            id="taskbar-tabs", active_tab="gen",
            children=[
                dbc.Tab(label="Generovanie",  tab_id="gen"),
                dbc.Tab(label="Augmentácia",  tab_id="aug"),
                dbc.Tab(label="Architektúra", tab_id="arch"),
                dbc.Tab(label="Tréning",      tab_id="train"),
                dbc.Tab(label="Vyhodnotenie", tab_id="eval"),
            ],
            style={"backgroundColor": "#f8f9fa", "justifyContent": "center"}
        )
    ], style={"position": "fixed", "bottom": 0, "width": "100%",
              "zIndex": 1000, "borderTop": "1px solid #dee2e6"})
])


@callback(Output("page-content", "children"), Input("taskbar-tabs", "active_tab"))
def switch_tab(at):
    if at == "gen":   return render_generation_tab()
    if at == "aug":   return render_augmentation_tab()
    if at == "arch":  return render_architecture_tab()
    if at == "train": return render_training_tab()
    if at == "eval":  return render_evaluation_tab()
    return html.Div([html.P("Sekcia vo vývoji", className="text-center mt-5")])




def register_viewer_callbacks(prefix):
    @callback(
        Output(f'{prefix}-file-dropdown', 'options'),
        Input(f'{prefix}-refresh-btn', 'n_clicks'),
        Input('aug-status', 'children') if prefix == "aug" else Input(f'{prefix}-status', 'children'),
        State(f'{prefix}-folder', 'value'),
        prevent_initial_call=True
    )
    def update_list(n, status, folder):
        if not folder or not os.path.exists(folder):
            return []
        files = sorted([f for f in os.listdir(folder) if f.endswith('.complex')])
        return [{'label': f, 'value': os.path.join(folder, f)} for f in files]

    @callback(
        Output(f'{prefix}-waveform-graph', 'figure'),
        Input(f'{prefix}-file-dropdown', 'value'),
        prevent_initial_call=True
    )
    def update_graph(filepath):
        if not filepath or not os.path.exists(filepath):
            return go.Figure()
        data = load_complex_file(filepath)[:9000]
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=data.real, name="I", line=dict(color='#2ecc71')))
        fig.add_trace(go.Scatter(y=data.imag, name="Q", line=dict(color='#e74c3c', dash='dot')))
        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
        return fig


register_viewer_callbacks("gen")
register_viewer_callbacks("aug")




@callback(
    Output('gen-status', 'children'),
    Input('gen-run-btn', 'n_clicks'),
    State('gen-folder', 'value'), State('gen-num-files', 'value'),
    State('gen-bits-min', 'value'), State('gen-bits-max', 'value'),
    State('gen-freq-min', 'value'), State('gen-freq-max', 'value'),
    State('gen-spb-min', 'value'), State('gen-spb-max', 'value'),
    State('gen-noise-min', 'value'), State('gen-noise-max', 'value'),
    State('gen-samp-rate', 'value'),
    background=True,
    running=[(Output("gen-run-btn", "disabled"), True, False)],
    progress=[Output('gen-progress', 'value'), Output('gen-progress', 'max')],
    prevent_initial_call=True
)
def run_gen(set_progress, n, folder, num, bmin, bmax, fmin, fmax, smin, smax, nmin, nmax, sr):
    generate_bulk(folder, num, bmin, bmax, fmin, fmax, smin, smax, nmin, nmax, sr,
                  lambda c, t: set_progress((c, t)))
    return f"Hotovo: {num} súborov"


@callback(
    Output('aug-status', 'children'),
    Input('aug-run-btn', 'n_clicks'),
    State('aug-folder', 'value'), State('aug-target-len', 'value'), State('aug-method', 'value'),
    background=True,
    running=[(Output("aug-run-btn", "disabled"), True, False)],
    progress=[Output('aug-progress', 'value'), Output('aug-progress', 'max')],
    prevent_initial_call=True
)
def run_aug(set_progress, n, folder, target, method):
    apply_augmentation_bulk(folder, target, method, lambda c, t: set_progress((c, t)))
    return "Augmentácia dokončená"



@callback(
    Output('arch-layers-container', 'children'),
    Input('arch-add-layer-btn', 'n_clicks'),
    Input({'type': 'arch-del-layer', 'index': ALL}, 'n_clicks'),
    State({'type': 'arch-layer-type', 'index': ALL}, 'value'),
    State({'type': 'arch-pa', 'index': ALL}, 'value'),
    State({'type': 'arch-pb', 'index': ALL}, 'value'),
    State({'type': 'arch-pc', 'index': ALL}, 'value'),
    State({'type': 'arch-pd', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def manage_layers(add_clicks, del_clicks, types, pa, pb, pc, pd):
    triggered = ctx.triggered_id
    layers = list(zip(types or [], pa or [], pb or [], pc or [], pd or []))

    if triggered == 'arch-add-layer-btn':
        layers.append(('Conv1d', None, None, None, None))

    elif isinstance(triggered, dict) and triggered.get('type') == 'arch-del-layer':
        del_pos = triggered['index']
        if 0 <= del_pos < len(layers) and len(layers) > 1:
            layers.pop(del_pos)

    return [create_layer_card(i, t, a, b, c, d) for i, (t, a, b, c, d) in enumerate(layers)]




@callback(
    Output({'type': 'arch-pa', 'index': ALL}, 'disabled'),
    Output({'type': 'arch-pb', 'index': ALL}, 'disabled'),
    Output({'type': 'arch-pc', 'index': ALL}, 'disabled'),
    Output({'type': 'arch-pd', 'index': ALL}, 'disabled'),
    Output({'type': 'arch-pa', 'index': ALL}, 'placeholder'),
    Output({'type': 'arch-pb', 'index': ALL}, 'placeholder'),
    Output({'type': 'arch-pc', 'index': ALL}, 'placeholder'),
    Output({'type': 'arch-pd', 'index': ALL}, 'placeholder'),
    Input({'type': 'arch-layer-type', 'index': ALL}, 'value'),
)
def sync_layer_params(types):
    da, db, dc, dd = [], [], [], []
    lpa, lpb, lpc, lpd = [], [], [], []
    for t in (types or []):
        meta = _PARAM_META.get(t, (None, None, None, None))
        da.append(meta[0] is None)
        db.append(meta[1] is None)
        dc.append(meta[2] is None)
        dd.append(meta[3] is None)
        lpa.append(meta[0] or "—")
        lpb.append(meta[1] or "—")
        lpc.append(meta[2] or "—")
        lpd.append(meta[3] or "—")
    return da, db, dc, dd, lpa, lpb, lpc, lpd



@callback(
    Output('arch-code-output', 'value'),
    Input('arch-generate-btn', 'n_clicks'),
    State('arch-input-samples', 'value'),
    State({'type': 'arch-layer-type', 'index': ALL}, 'value'),
    State({'type': 'arch-pa', 'index': ALL}, 'value'),
    State({'type': 'arch-pb', 'index': ALL}, 'value'),
    State({'type': 'arch-pc', 'index': ALL}, 'value'),
    State({'type': 'arch-pd', 'index': ALL}, 'value'),
    prevent_initial_call=True
)
def gen_code(n, input_samples, types, pa, pb, pc, pd):
    input_samples = input_samples or 9000

    def _layer_line(t, a, b, c, d, in_feat_override=None):
        if t == 'Conv1d':
            return f"nn.Conv1d({a}, {b}, kernel_size={c}, padding={d})"
        elif t == 'MaxPool1d':
            return f"nn.MaxPool1d({a})"
        elif t == 'Linear':
            in_f = in_feat_override if in_feat_override is not None else a
            return f"nn.Linear({in_f}, {b})"
        elif t == 'Flatten':
            return "nn.Flatten()"
        elif t in ('ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid'):
            return f"nn.{t}()"
        return ""

    flatten_pos = next((i for i, t in enumerate(types) if t == 'Flatten'), None)

    if flatten_pos is None:
        lines = []
        for i, t in enumerate(types):
            line = _layer_line(t, pa[i], pb[i], pc[i], pd[i])
            if line:
                lines.append(f"            {line}")
        code = (
            "import torch\nimport torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.net = nn.Sequential(\n"
            + ",\n".join(lines) + "\n"
            "        )\n\n"
            "    def forward(self, x):\n"
            "        return self.net(x)\n"
        )
    else:
        in_ch = 2
        for i, t in enumerate(types):
            if t == 'Conv1d' and pa[i] is not None:
                in_ch = pa[i]
                break

        feat_lines = []
        for i in range(flatten_pos + 1):
            line = _layer_line(types[i], pa[i], pb[i], pc[i], pd[i])
            if line:
                feat_lines.append(f"            {line}")

        head_lines = []
        first_linear_done = False
        for i in range(flatten_pos + 1, len(types)):
            t = types[i]
            override = None
            if t == 'Linear' and not first_linear_done:
                override = "_flat"
                first_linear_done = True
            line = _layer_line(t, pa[i], pb[i], pc[i], pd[i], in_feat_override=override)
            if line:
                head_lines.append(f"            {line}")

        code = (
            "import torch\nimport torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            f"    def __init__(self, input_samples={input_samples}):\n"
            "        super().__init__()\n"
            "        self._features = nn.Sequential(\n"
            + ",\n".join(feat_lines) + "\n"
            "        )\n"
            "        with torch.no_grad():\n"
            f"            _flat = self._features(torch.zeros(1, {in_ch}, input_samples)).shape[1]\n"
            "        self._head = nn.Sequential(\n"
            + ",\n".join(head_lines) + "\n"
            "        )\n\n"
            "    def forward(self, x):\n"
            "        return self._head(self._features(x))\n"
        )

    return code


@callback(
    Output("arch-save-status", "children"),
    Input("arch-save-btn", "n_clicks"),
    State("arch-code-output", "value"), State("arch-model-name", "value"),
    prevent_initial_call=True
)
def save_arch(n, code, name):
    path = os.path.join(os.getcwd(), "Model_architectures")
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f"{name}.py"), "w") as f:
        f.write(code)
    return f"Uložené v Model_architectures"



@callback(
    Output("train-model-dropdown", "options"),
    Input("train-refresh-models-btn", "n_clicks"),
    Input("taskbar-tabs", "active_tab")
)
def update_train_models(n, tab):
    arch_dir = os.path.join(os.getcwd(), "Model_architectures")
    return [{'label': f, 'value': os.path.join(arch_dir, f)} for f in get_available_models(arch_dir)]




@callback(
    Output("eval-file-dropdown", "options"),
    Input("eval-refresh-files-btn", "n_clicks"),
    State("eval-folder", "value"),
    prevent_initial_call=True
)
def eval_refresh_files(n, folder):
    ids = list_evaluable_files(folder)
    return [{'label': fid, 'value': fid} for fid in ids]


@callback(
    Output("eval-arch-dropdown", "options"),
    Input("eval-refresh-arch-btn", "n_clicks"),
    Input("taskbar-tabs", "active_tab"),
)
def eval_refresh_arch(n, tab):
    arch_dir = os.path.join(os.getcwd(), "Model_architectures")
    if not os.path.exists(arch_dir):
        return []
    files = sorted([f for f in os.listdir(arch_dir) if f.endswith(".py")])
    return [{'label': f, 'value': os.path.join(arch_dir, f)} for f in files]


@callback(
    Output("eval-weights-dropdown", "options"),
    Input("eval-refresh-weights-btn", "n_clicks"),
    Input("taskbar-tabs", "active_tab"),
)
def eval_refresh_weights(n, tab):
    files = list_weight_files(os.getcwd())
    return [{'label': f, 'value': os.path.join(os.getcwd(), f)} for f in files]



@callback(
    Output("eval-status", "children"),
    Output("eval-result-store", "data"),
    Input("eval-run-btn", "n_clicks"),
    State("eval-mode", "value"),
    State("eval-folder", "value"),
    State("eval-arch-dropdown", "value"),
    State("eval-weights-dropdown", "value"),
    State("eval-file-dropdown", "value"),
    background=True,
    running=[
        (Output("eval-run-btn", "disabled"), True, False),
    ],
    progress=[
        Output("eval-progress", "value"),
        Output("eval-progress", "max"),
    ],
    prevent_initial_call=True
)
def run_evaluation_dispatch(set_progress, single_clicks,
                            mode, folder, arch_path, weights_path, file_id):

    if not all([folder, arch_path, weights_path]):
        return "Vyplň všetky polia", no_update

    if not file_id:
        return "Vyber súbor", no_update

    def prog(msg, cur, total):
        set_progress((cur, total))

    try:
        if mode == "bits":
            res = run_evaluation_bits(
                data_dir=folder,
                arch_path=arch_path,
                weights_path=weights_path,
                file_id=file_id,
                progress_callback=prog,
            )
            store_data = {
                "mode":              "bits",
                "bits_true":         res["bits_true"],
                "bits_predicted":    res["bits_predicted"],
                "bits_traditional":  res["bits_traditional"],
                "ber":               res["ber"],
                "ber_traditional":   res["ber_traditional"],
                "file_id":           res["file_id"],
                "device":            res["device"],
            }
        else:
            res = run_evaluation(
                data_dir=folder,
                arch_path=arch_path,
                weights_path=weights_path,
                file_id=file_id,
                progress_callback=prog,
            )
            store_data = {
                "mode":                   "denoise",
                "has_noiseless":          res["has_noiseless"],
                "noisy_real":             res["noisy"].real.tolist(),
                "noisy_imag":             res["noisy"].imag.tolist(),
                "denoised_real":          res["denoised"].real.tolist(),
                "denoised_imag":          res["denoised"].imag.tolist(),
                "noiseless_real":         res["noiseless"].real.tolist() if res["has_noiseless"] else [],
                "noiseless_imag":         res["noiseless"].imag.tolist() if res["has_noiseless"] else [],
                "noisy_aug_real":         res["noisy_aug"].real.tolist(),
                "noisy_aug_imag":         res["noisy_aug"].imag.tolist(),
                "denoised_aug_real":      res["denoised_aug"].real.tolist(),
                "denoised_aug_imag":      res["denoised_aug"].imag.tolist(),
                "bits_true":              res["bits_true"],
                "bits_noisy":             res["bits_noisy"],
                "bits_denoised":          res["bits_denoised"],
                "sample_points_noisy":    res["sample_points_noisy"],
                "sample_points_denoised": res["sample_points_denoised"],
                "filtered_noisy":         res["filtered_noisy"].tolist(),
                "filtered_denoised":      res["filtered_denoised"].tolist(),
                "threshold_noisy":        res["threshold_noisy"],
                "threshold_denoised":     res["threshold_denoised"],
                "mse_noisy":              res["mse_noisy"],
                "mse_denoised":           res["mse_denoised"],
                "snr_noisy":              res["snr_noisy"],
                "snr_denoised":           res["snr_denoised"],
                "ber_noisy":              res["ber_noisy"],
                "ber_denoised":           res["ber_denoised"],
                "samples_per_bit":        res["samples_per_bit"],
                "file_id":                res["file_id"],
                "device":                 res["device"],
            }
    except Exception as e:
        return f"Chyba: {e}", no_update

    return f"Hotovo: súbor {file_id}", store_data


@callback(
    Output("eval-batch-status",       "children"),
    Output("eval-batch-trad-result",  "children"),
    Output("eval-batch-model-result", "children"),
    Output("eval-batch-results-card", "style"),
    Input("eval-batch-run-btn", "n_clicks"),
    State("eval-mode",             "value"),
    State("eval-folder",           "value"),
    State("eval-arch-dropdown",    "value"),
    State("eval-weights-dropdown", "value"),
    background=True,
    running=[(Output("eval-batch-run-btn", "disabled"), True, False)],
    progress=[
        Output("eval-batch-progress", "value"),
        Output("eval-batch-progress", "max"),
    ],
    prevent_initial_call=True,
)
def run_batch_evaluation(set_progress, n_clicks,
                         mode, folder, arch_path, weights_path):
    import torch

    if not all([folder, arch_path, weights_path]):
        return "Vyplň všetky polia", "—", "—", {"display": "none"}

    ids = list_evaluable_files(folder)
    if not ids:
        return "Žiadne súbory na vyhodnotenie", "—", "—", {"display": "none"}

    try:
        model, device = load_evaluation_model(arch_path, weights_path)
    except Exception as e:
        return f"Chyba pri načítaní modelu: {e}", "—", "—", {"display": "none"}

    total         = len(ids)
    ber_model_sum = 0.0
    ber_trad_sum  = 0.0
    errors        = 0

    for idx, file_id in enumerate(ids):
        set_progress((idx, total))
        try:
            if mode == "bits":
                res = run_evaluation_bits(
                    data_dir=folder,
                    arch_path=arch_path,
                    weights_path=weights_path,
                    file_id=file_id,
                    model=model,
                    device=device,
                )
                ber_model_sum += res["ber"]
                ber_trad_sum  += res["ber_traditional"]

            else:  # denoise — fast path: skip SNR/MSE, only demodulate
                complex_path = os.path.join(folder, file_id + ".complex")
                txt_path     = os.path.join(folder, file_id + ".txt")
                meta         = parse_txt_metadata(txt_path)
                bits_true    = meta["bits"]
                spb          = meta["samples_per_bit"]

                noisy_aug = _load_and_pad(complex_path, _AUGMENTED_SAMPLES)

                x   = np.stack([noisy_aug.real, noisy_aug.imag], axis=0)
                x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.inference_mode():
                    y_t = model(x_t)
                y            = y_t.squeeze(0).cpu().numpy()
                denoised_aug = (y[0] + 1j * y[1]).astype(np.complex64)

                orig_len      = len(bits_true) * spb
                noisy_orig    = _resample_to_original(noisy_aug,    orig_len)
                denoised_orig = _resample_to_original(denoised_aug, orig_len)

                bits_noisy,    _, _, _ = ask_demodulate(noisy_orig,    spb)
                bits_denoised, _, _, _ = ask_demodulate(denoised_orig, spb)

                n = min(len(bits_noisy), len(bits_true))
                ber_trad_sum  += sum(p != r for p, r in zip(bits_noisy[:n],    bits_true[:n])) / max(n, 1)

                n = min(len(bits_denoised), len(bits_true))
                ber_model_sum += sum(p != r for p, r in zip(bits_denoised[:n], bits_true[:n])) / max(n, 1)

        except Exception:
            errors += 1

    set_progress((total, total))

    evaluated = total - errors
    if evaluated == 0:
        return "Všetky súbory zlyhali", "—", "—", {"display": "none"}

    acc_trad  = (1 - ber_trad_sum  / evaluated) * 100
    acc_model = (1 - ber_model_sum / evaluated) * 100

    status = f"Hotovo: {evaluated}/{total} súborov" + (f" ({errors} chýb)" if errors else "")
    return (
        status,
        f"{acc_trad:.1f} %",
        f"{acc_model:.1f} %",
        {"display": "block"},
    )


def _empty_fig(msg="Spusti vyhodnotenie"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=16, color="#aaa"))
    fig.update_layout(template="plotly_white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _build_bit_comparison_fig(bits_true, bits_lists, labels, title_prefix=""):
    n_pred = len(bits_lists)
    n_rows = 1 + n_pred
    titles = [f"{title_prefix}Ground truth bity"] + [
        f"{title_prefix}{lbl}" for lbl in labels
    ]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        subplot_titles=titles,
        vertical_spacing=0.10
    )

    n_show = len(bits_true)
    for bl in bits_lists:
        n_show = min(n_show, len(bl))
    n_show = min(n_show, 200)
    xs = list(range(n_show))

    fig.add_trace(go.Bar(
        x=xs, y=bits_true[:n_show],
        marker_color="#2ecc71", name="Ground truth",
        width=0.8, showlegend=False,
    ), row=1, col=1)

    for row_idx, (bits_pred, label) in enumerate(zip(bits_lists, labels), start=2):
        colors = []
        bar_values = []
        for i in range(n_show):
            if bits_pred[i] == bits_true[i]:
                colors.append("#2ecc71")
                bar_values.append(bits_pred[i])
            else:
                colors.append("#e74c3c")
                bar_values.append(1)

        fig.add_trace(go.Bar(
            x=xs, y=bar_values,
            marker_color=colors,
            name=label, width=0.8, showlegend=False,
        ), row=row_idx, col=1)

    fig.update_yaxes(range=[-0.1, 1.4], tickvals=[0, 1])
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=20),
        bargap=0.1,
        legend=dict(orientation="h", y=-0.05),
    )
    return fig


@callback(
    Output("eval-main-graph", "figure"),
    Input("eval-result-store", "data"),
    Input("eval-graph-tabs", "active_tab"),
    prevent_initial_call=True
)
def render_eval_results(data, active_tab):
    if not data:
        return _empty_fig()

    eval_mode = data.get("mode", "denoise")

    if eval_mode == "bits":
        if active_tab != "tab-bits":
            return _empty_fig("Prepni na záložku 'Porovnanie bitov' pre zobrazenie výsledkov")

        bt = data["bits_true"]
        bp = data["bits_predicted"]
        b_trad = data["bits_traditional"]
        ber = data.get("ber", 0)
        ber_trad = data.get("ber_traditional", 0)

        fig = _build_bit_comparison_fig(
            bt,
            [b_trad, bp],
            [
                f"Tradičná ASK demodulácia",
                f"Predikované bity",
            ],
        )
        return fig

    if active_tab == "tab-signals":
        has_nl = data.get("has_noiseless", False)
        t_orig = list(range(len(data["noisy_real"])))

        n_rows = 3 if has_nl else 2
        titles = (
            ["Zašumený signál (IQ)", "Odšumený signál (IQ)", "Pôvodný signál (IQ)"]
            if has_nl else
            ["Zašumený signál (IQ)", "Odšumený signál (IQ)"]
        )

        fig = make_subplots(
            rows=n_rows, cols=1, shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.08
        )
        fig.add_trace(go.Scatter(x=t_orig, y=data["noisy_real"], name="Noisy I",
                                 line=dict(color="#3498db", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_orig, y=data["noisy_imag"], name="Noisy Q",
                                 line=dict(color="#3498db", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_orig, y=data["denoised_real"], name="Denoised I",
                                 line=dict(color="#e67e22", width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_orig, y=data["denoised_imag"], name="Denoised Q",
                                 line=dict(color="#e67e22", width=1, dash="dot")), row=2, col=1)
        if has_nl:
            t_nl = list(range(len(data["noiseless_real"])))
            fig.add_trace(go.Scatter(x=t_nl, y=data["noiseless_real"], name="Noiseless I",
                                     line=dict(color="#2ecc71", width=1)), row=3, col=1)
            fig.add_trace(go.Scatter(x=t_nl, y=data["noiseless_imag"], name="Noiseless Q",
                                     line=dict(color="#2ecc71", width=1, dash="dot")), row=3, col=1)

        fig.update_layout(template="plotly_white", hovermode="x unified",
                          margin=dict(l=40, r=20, t=50, b=20),
                          legend=dict(orientation="h", y=-0.05))

    elif active_tab == "tab-demod":
        fn = data["filtered_noisy"]
        fd = data["filtered_denoised"]
        sp_n = data["sample_points_noisy"]
        sp_d = data["sample_points_denoised"]
        thr_n = data["threshold_noisy"]
        thr_d = data["threshold_denoised"]

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=False,
            subplot_titles=("Zašumený — ASK obálka", "Odšumený — ASK obálka"),
            vertical_spacing=0.12
        )
        fig.add_trace(go.Scatter(y=fn, name="Obálka",
                                 line=dict(color="#3498db", width=1)), row=1, col=1)
        fig.add_hline(y=thr_n, line_dash="dash", line_color="#e74c3c",
                      annotation_text=f"Prah {thr_n:.3f}", row=1, col=1)
        fig.add_trace(go.Scatter(
            x=sp_n, y=[fn[i] for i in sp_n],
            mode="markers", name="Vzorky bitov",
            marker=dict(color="#e74c3c", size=6, symbol="circle")), row=1, col=1)

        fig.add_trace(go.Scatter(y=fd, name="Obálka (odšumený)",
                                 line=dict(color="#e67e22", width=1)), row=2, col=1)
        fig.add_hline(y=thr_d, line_dash="dash", line_color="#27ae60",
                      annotation_text=f"Prah {thr_d:.3f}", row=2, col=1)
        fig.add_trace(go.Scatter(
            x=sp_d, y=[fd[i] for i in sp_d],
            mode="markers", name="Vzorky bitov (odšumený)",
            marker=dict(color="#27ae60", size=6, symbol="circle")), row=2, col=1)

        fig.update_layout(template="plotly_white",
                          margin=dict(l=40, r=20, t=50, b=20),
                          legend=dict(orientation="h", y=-0.05))

    elif active_tab == "tab-bits":
        bt = data["bits_true"]
        bn = data["bits_noisy"]
        bd = data["bits_denoised"]

        fig = _build_bit_comparison_fig(
            bt, [bn, bd],
            ["Dekódované bity — zašumený signál",
             "Dekódované bity — odšumený signál"],
        )
    else:
        fig = _empty_fig()

    return fig


if __name__ == '__main__':
    app.run(debug=True)
