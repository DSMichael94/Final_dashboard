# Actividad Final — Dashboard & Notebook

Incluye:
- `notebooks/actividad_final_dashboard.ipynb` (EDA, Welch, lineal, logística, regularización, tuning, curva de aprendizaje, JupyterDash, export `app.py`)
- `app.py` (Dash listo para ejecutar/desplegar)
- `data/students_synth.csv`
- `assets/style.css`
- `requirements*.txt`
- `docs/informe_adj.docx`
- `links.txt`

## Ejecutar localmente
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements-win.txt
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt

python app.py
# http://127.0.0.1:8050
```

## Despliegue externo
- **Render.com**: Build `pip install -r requirements.txt`, Start `gunicorn app:server`
- **Hugging Face Spaces (Python)**: sube `app.py`, `requirements.txt`, `assets/`, `data/`; var `PORT=7860`.

## Nota
Puedes reemplazar `data/students_synth.csv` por tu dataset real manteniendo nombres de columnas o ajustando `app.py`.
