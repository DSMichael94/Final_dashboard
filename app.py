
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

CSV = os.path.join("data", "students_synth.csv")
df = pd.read_csv(CSV)

X = pd.concat([df[['horas_estudio','horas_suenio','gpa_previo']],
               pd.get_dummies(df['metodo'], drop_first=True)], axis=1)
y = df['aprueba']

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=77, stratify=y)
scaler = StandardScaler()
Xtr_sc = scaler.fit_transform(Xtr)
Xte_sc = scaler.transform(Xte)

clf = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0)
clf.fit(Xtr_sc, ytr)
proba = clf.predict_proba(Xte_sc)[:,1]

app = Dash(__name__)
server = app.server
app.title = "Actividad Final — Dashboard"

def metrics(thr):
    yhat = (proba >= thr).astype(int)
    return dict(
        accuracy = round(accuracy_score(yte, yhat),3),
        precision = round(precision_score(yte, yhat, zero_division=0),3),
        recall = round(recall_score(yte, yhat),3),
        f1 = round(f1_score(yte, yhat),3),
        roc_auc = round(roc_auc_score(yte, proba),3),
        cm = confusion_matrix(yte, yhat).ravel().tolist()
    )

app.layout = html.Div(className="container", children=[
    html.H2("Actividad Final — Dashboard (Demo)"),
    html.P("Explora el dataset sintético y un modelo logístico sencillo."),

    html.Div(className="panel", children=[
        html.Label("Umbral de decisión"),
        dcc.Slider(id="thr", min=0.05, max=0.95, step=0.05, value=0.5,
                   marks={i/20: str(round(i/20,2)) for i in range(1,20)}),
        html.Div(id="cards", className="cards")
    ]),

    html.Div(className="row", children=[
        html.Div(className="col-6", children=[dcc.Graph(id="roc")]),
        html.Div(className="col-6", children=[dcc.Graph(id="coeffs")]),
    ]),

    html.Div(className="panel", children=[
        html.H3("Explorador"),
        html.Div(className="row", children=[
            html.Div(className="col-4", children=[
                html.Label("Eje X"),
                dcc.Dropdown(id="x", options=[{"label":c,"value":c} for c in ["horas_estudio","horas_suenio","gpa_previo","indice_socioeco"]],
                             value="horas_estudio", clearable=False),
                html.Label("Eje Y"),
                dcc.Dropdown(id="y", options=[{"label":"puntuacion","value":"puntuacion"}],
                             value="puntuacion", clearable=False),
                html.Label("Método"),
                dcc.Dropdown(id="mfil", options=[{"label":"Todos","value":"all"},{"label":"A","value":"A"},{"label":"B","value":"B"}],
                             value="all", clearable=False)
            ]),
            html.Div(className="col-8", children=[dcc.Graph(id="scatter")])
        ]),
        dash_table.DataTable(
            id="tbl",
            data=df.head(100).to_dict("records"),
            columns=[{"name":c,"id":c} for c in df.columns],
            page_size=10, sort_action="native", filter_action="native",
            style_table={"overflowX":"auto"}
        )
    ]),

    html.Footer("© Proyecto académico — Demo")
])

@app.callback(
    Output("cards","children"),
    Output("roc","figure"),
    Output("coeffs","figure"),
    Output("scatter","figure"),
    Input("thr","value"),
    Input("x","value"),
    Input("y","value"),
    Input("mfil","value"),
)
def update(thr, x, y, mfil):
    m = metrics(thr)
    tn, fp, fn, tp = m["cm"]

    # tarjetas
    def card(lbl,val): 
        return html.Div(className="card", children=[html.Div(lbl, className="metric-label"), html.Div(str(val), className="metric-value")])
    cards = [card(k.capitalize(), v) for k,v in m.items() if k not in ("cm",)]

    # ROC
    fpr, tpr, _ = roc_curve(yte, proba)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, name="ROC", mode="lines"))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Azar", mode="lines", line=dict(dash="dash")))
    roc_fig.update_layout(title=f"ROC — AUC={m['roc_auc']:.3f} (thr={thr:.2f}) | TP:{tp} FP:{fp} FN:{fn} TN:{tn}",
                          xaxis_title="FPR", yaxis_title="TPR")

    # Coefs
    coef = pd.Series(clf.coef_[0], index=X.columns)
    coeffs = coef.sort_values(ascending=False).reset_index()
    coeffs.columns=["feature","coef"]
    coef_fig = px.bar(coeffs, x="feature", y="coef", title="Coeficientes (LogReg)")

    # Scatter
    dff = df.copy()
    if mfil != "all":
        dff = dff[dff["metodo"]==mfil]
    scat = px.scatter(dff, x=x, y=y, color="aprueba", trendline="ols", title=f"{x} vs {y}")
    return cards, roc_fig, coef_fig, scat

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
