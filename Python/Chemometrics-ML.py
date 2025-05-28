import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    # Importiere alle benötigten Pakete
    import marimo as mo
    import random
    import numpy as np
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from ipywidgets import interact, FloatSlider
    from scipy.optimize import minimize, Bounds
    from scipy import sparse
    import os 
    return Bounds, go, minimize, mo, np, os, sparse


@app.cell(hide_code=True)
def _(np, sparse):
    # Definiere nützliche Funktionen für das Raman-Spektrum und die Datenanalyse

    # Erzeuge x-Achse für das Spektrum (Wellenzahlen)
    x = np.linspace(50, 4000, 1000)

    def lorentzian(x, x0, gamma, A):
        """
        Lorentz-Funktion mit Zentrum x0, FWHM gamma und Amplitude A.
        """
        return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2)

    def gauss(x, A,mu,sigma):
        """
        Gaußfunktion (Zentrum mu, Standardabweichung sigma, Amplitude A).
        """
        return A * np.exp(-((x - mu)**2) / (2 * sigma**2))

    def line(x, m, b):
        """
        Gerade: y = m*x + b
        """
        return m * x + b

    def raman_spectrum(x, bands):
        """
        Summiert mehrere Lorentz-Banden.
        bands: Liste von Tupeln (Zentrum, Breite, Amplitude)
        """
        y = np.zeros_like(x)
        for center, width, amp in bands:
            y += lorentzian(x, center, width, amp)
        return y

    def norm(A):
        # Normiert ein Array auf Maximum = 1
        return A / np.max(A)

    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Schätzt die Baseline eines Spektrums per asymmetrischem Least-Squares (Eilers' Methode).

        Parameter:
        y    : Originalspektrum (1D-Array)
        lam  : Glättungsparameter (je größer, desto glatter)
        p    : Asymmetrieparameter (klein: stärkeres Unterdrücken von Peaks nach oben)
        niter: Anzahl der Iterationen

        Rückgabe:
        baseline : geschätzte Baseline
        y_corr   : baseline-korrigiertes Spektrum (y - baseline)
        """
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = D.dot(D.T)  # Zweite Ableitung (Glättung)
        w = np.ones(L)
        for i in range(niter):
            W = sparse.diags(w, 0)
            Z = W + lam * D
            baseline = sparse.linalg.spsolve(Z, w * y)
            w = p * (y > baseline) + (1 - p) * (y < baseline)
        y_corr = y - baseline
        return baseline, y_corr

    def cost_function(p, wn, spectrum, refspectra):
        """
        Kostenfunktion für das Fitting eines Raman-Spektrums als Linearkombination von vier Referenzspektren,
        ergänzt um linearen und gaußförmigen Hintergrund.

        Die Referenzanteile werden so normiert, dass ihre Summe = 1 ist.
        """
        ref1, ref2, ref3, ref4 = refspectra
        # Extrahiere die Parameter für die Referenzanteile
        A_raw, B_raw, C_raw, D_raw = p[0:4]
        total = A_raw + B_raw + C_raw + D_raw
        # Verhindere Division durch Null
        if total == 0:
            A, B, C, D = 0.25, 0.25, 0.25, 0.25  # Gleichverteilt, falls alle 0
        else:
            A, B, C, D = A_raw / total, B_raw / total, C_raw / total, D_raw / total

        m, b = p[4:6]
        A0, mu, sigma = p[6:9]
        y = A * ref1 + B * ref2 + C * ref3 + D * ref4
        y += line(wn,m,b) + gauss(wn,A0,mu,sigma)
        cost = np.sum((y - spectrum)**2)
        return cost

    return baseline_als, cost_function, gauss, line, norm


@app.cell(hide_code=True)
def _(mo):
    # Titel und Kurzbeschreibung
    mo.md(
        r"""
        # Raman-Spektrum-Viewer

        Dieses Notebook erlaubt es, einen Ordnerpfad einzugeben, daraus alle `.tsv`-Dateien auszuwählen und das Spektrum zu plotten.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo, os):
    # Dateiauswahl per Dropdown
    folder = '../Data/Literatur/'
    folder_spectra = '../Data/Eisen1/2025-05-19/'
    tsv_files = [f for f in os.listdir(folder_spectra) if f.lower().endswith('.tsv')]
    if not tsv_files:
        raise FileNotFoundError(f"Keine .tsv-Dateien in {folder_spectra!r} gefunden.")

    dropdown_form = mo.ui.dropdown(
        options=tsv_files,
        value=tsv_files[5],
        label="Spektrum auswählen",
        searchable=True,
        full_width=True
    ).form()
    dropdown_form
    return dropdown_form, folder, folder_spectra


@app.cell(hide_code=True)
def _(
    Bounds,
    baseline_als,
    cost_function,
    dropdown_form,
    folder,
    folder_spectra,
    gauss,
    go,
    line,
    minimize,
    mo,
    norm,
    np,
    os,
):
    # Stoppe die Ausführung, wenn keine Datei gewählt wurde
    mo.stop(dropdown_form.value is None, "Bitte wähle eine Datei und klicke Submit")

    # Fehlerbehandlungsblock (try/except) für robustes Einlesen aller Datenquellen:
    try:
        # Raman-Spektrum und Referenzdaten einlesen
        path = os.path.join(folder_spectra, dropdown_form.value)
        data = np.loadtxt(path, delimiter='\t', skiprows=8)

        # Kalibrierung (Silizium-Peak auf 520.45 cm-1)
        calibration = np.loadtxt('../Data/Eisen1/2025-05-19/wn-calibration.tsv', delimiter='\t', skiprows=8)
        Si_peak = 520.45
        wn_corr = calibration[np.argmax(calibration[:, 1]), 0] - Si_peak

        # Dunkelstrom (Background) abziehen
        dark = np.loadtxt('../Data/Eisen1/2025-05-19/Dark.tsv', delimiter='\t', skiprows=8)

        # Wellenzahlen und Intensitäten berechnen
        wn_spectrum = data[:, 0] - wn_corr
        int_spectrum = data[:, 1] - dark[:, 1]

        # Referenzspektren laden (Hämatit, Magnetit, Goethit, Lepidokrit)
        hematite = np.loadtxt(os.path.join(folder, 'Hämatit - Fe2O3', 'fe2o3-processed-2.txt'), delimiter=',', skiprows=10)
        magnetite = np.loadtxt(os.path.join(folder, 'Magnetit-Fe3O4', 'fe3o4-processed.txt'), delimiter=',', skiprows=10)
        goethite = np.loadtxt(os.path.join(folder, 'Goethit - a-FeOOH', 'a-feooh-raw.txt'), delimiter=',', skiprows=10)
        lepido = np.loadtxt(os.path.join(folder, 'Lepidokrit - g-FeOOH', 'g-feooh-processed.txt'), delimiter=',', skiprows=10)

    except Exception as e:
        # Verbesserte Fehlerausgabe mit Info, welche Datei das Problem verursacht hat
        import traceback
        mo.md(f"**Dateifehler:** {e}\n\n```python\n{traceback.format_exc()}\n```")
        raise mo.StopException("Fehler beim Einlesen der Spektren- oder Referenzdaten.")

    # Gemeinsamen Wellenzahlbereich bestimmen (keine Extrapolation!)
    x_min = max(hematite[:, 0].min(), magnetite[:, 0].min(), goethite[:, 0].min(), lepido[:, 0].min(), wn_spectrum.min())
    x_max = min(hematite[:, 0].max(), magnetite[:, 0].max(), goethite[:, 0].max(), lepido[:, 0].max(), wn_spectrum.max())

    common_x = np.linspace(x_min, x_max, 1000)

    # Interpolation auf gemeinsamen Bereich und Baseline-Korrektur & Normierung
    y_hem_raw = norm(np.interp(common_x, hematite[:, 0], hematite[:, 1]))
    y_hem = norm(baseline_als(y_hem_raw, lam=1e6, p=0.001, niter=10)[1])

    y_mag_raw = norm(np.interp(common_x, magnetite[:, 0], magnetite[:, 1]))
    y_mag = norm(baseline_als(y_mag_raw, lam=1e6, p=0.001, niter=10)[1])

    y_goet_raw = norm(np.interp(common_x, goethite[:, 0], goethite[:, 1]))
    y_goet = norm(baseline_als(y_goet_raw, lam=1e6, p=0.001, niter=10)[1])

    y_lepid_raw = norm(np.interp(common_x, lepido[:, 0], lepido[:, 1]))
    y_lepid = norm(baseline_als(y_lepid_raw, lam=1e6, p=0.001, niter=10)[1])

    # Gemessenes Spektrum interpolieren und normieren
    y_spectrum = norm(np.interp(common_x, wn_spectrum, int_spectrum))

    refspectra = [y_hem, y_mag, y_goet, y_lepid]

    # Parametergrenzen für die Optimierung
    # Parameter    [     A,      B,      C,      D,       m,       b,      A0,                  mu, sigma]
    lower_bounds = [     0,      0,      0,      0, -np.inf,       0,       0, common_x.min()-1000,    85]
    upper_bounds = [np.inf, np.inf, np.inf, np.inf,  np.inf,       1,  1, common_x.max()+1000, 10000]
    bounds = Bounds(lower_bounds, upper_bounds)

    # Startwerte für die Optimierung
    #    [  A,   B,   C,   D, m, b, A0, mu, sigma]
    p0 = [0.1, 0.1, 0.1, 0.1, 0, 0,  0,  0,   85]

    # Fit per L-BFGS-B (mit Grenzen)
    res = minimize(
        cost_function,
        p0,
        args=(common_x, y_spectrum, refspectra),
        method='L-BFGS-B',
        bounds=bounds
    )

    # Optimal gefundene Parameter extrahieren
    p_opt = res.x
    A_raw, B_raw, C_raw, D_raw = p_opt[0:4]
    total = A_raw + B_raw + C_raw + D_raw
    if total == 0:
        A, B, C, D = 0.25, 0.25, 0.25, 0.25
    else:
        A, B, C, D = A_raw / total, B_raw / total, C_raw / total, D_raw / total
    m, b = p_opt[4:6]
    A0, mu, sigma = p_opt[6:9]

    # Fitparameter Textblock
    fittext = (
        "<span style='font-family:monospace'>"
        "<b>Fitparameter</b><br>"
        f"{'Hämatit'.ljust(11)}= {A*100:>6.1f} %<br>"
        f"{'Magnetit'.ljust(11)}= {B*100:>6.1f} %<br>"
        f"{'Goethit'.ljust(11)}= {C*100:>6.1f} %<br>"
        f"{'Lepidokrit'.ljust(11)}= {D*100:>6.1f} %<br>"
        f"{'m'.ljust(11)}= {m:>6.2g}<br>"
        f"{'b'.ljust(11)}= {b:>6.2g}<br>"
        f"{'A0'.ljust(11)}= {A0:>6.2g}<br>"
        f"{'μ'.ljust(11)}= {mu:>6.2g}<br>"
        f"{'σ'.ljust(11)}= {sigma:>6.2g}<br>"
        f"{'Kosten'.ljust(11)}= {res.fun:>6.2e}"
        "</span>"
    )

    ref1, ref2, ref3, ref4 = refspectra

    # Gefittetes Spektrum berechnen (mit normierten Referenzanteilen)
    y_fit = (A * ref1 + B * ref2 + C * ref3 + D * ref4
             + line(common_x,m,b)
             + gauss(common_x,A0,mu,sigma)
            )

    # Residuen berechnen (Messung - Fit)
    y_residuum = y_spectrum - y_fit

    # Farbschema für alle Kurven 
    colors = {
        "measurement": "#5CACEE",   # Stahlblay für Messung
        "fit": "#e68a00",           # Orange für Fit
        "hämatit": "#b2182b",       # Rot für Hämatit
        "magnetit": "#2166ac",      # Blau für Magnetit
        "goethit": "#8f7919",       # Gelb/Orange für Goethit
        "lepidokrit": "#4dac26",    # Grün für Lepidokrit
        "residuum": "#000000"       # Schwarz für Residuum
    }

    # Referenzspektren-Kurven
    width = 1.5
    ref_traces = [
        go.Scatter(x=common_x, y=ref1, mode='lines', name='Hämatit', line=dict(color=colors["hämatit"], width=width, dash='dot')),
        go.Scatter(x=common_x, y=ref2, mode='lines', name='Magnetit', line=dict(color=colors["magnetit"], width=width, dash='dot')),
        go.Scatter(x=common_x, y=ref3, mode='lines', name='Goethit', line=dict(color=colors["goethit"], width=width, dash='dot')),
        go.Scatter(x=common_x, y=ref4, mode='lines', name='Lepidokrit', line=dict(color=colors["lepidokrit"], width=width, dash='dot'))
    ]

    # Messspektrum
    measurement = go.Scatter(
        x=common_x,
        y=y_spectrum,
        mode='lines',
        name='Raman-Spektrum',
        line=dict(color=colors["measurement"], width=2)
    )

    # Gefittetes Spektrum
    fit_curve = go.Scatter(
        x=common_x,
        y=y_fit,
        mode='lines',
        name='Gefittetes Spektrum',
        line=dict(color=colors["fit"], width=2, dash='dash')
    )

    # Residuen
    residuum_curve = go.Scatter(
        x=common_x,
        y=y_residuum,
        mode='lines',
        name='Residuum (Messung - Fit)',
        line=dict(color=colors["residuum"], width=2, dash='longdash')
    )

    # Layout mit Legende
    layout = go.Layout(
        title=f'Raman-Spektrum: {dropdown_form.value}',
        xaxis=dict(title='Wellenzahl (cm⁻¹)'),
        yaxis=dict(title='Raman Intensität'),
        showlegend=True,
        template='simple_white',
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        legend=dict(
            x=1000, y=0.99,
            bgcolor='rgba(255,255,255,0.7)'
        ),
        #  Fitparameter-Block als Annotation 
        annotations=[
            dict(
                xref='paper', yref='paper',
                x=0.99, y=0.99,  # Oben rechts im Plotfenster
                showarrow=False,
                align='left',
                text=fittext,
                font=dict(size=12, family='monospace', color='black'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='gray'
            )
        ]
    )

    # Plot zusammenbauen: Messung, Fit, Residuum, Referenzspektren
    fig2 = go.Figure(data=[measurement, fit_curve, residuum_curve] + ref_traces, layout=layout)

    mo.ui.plotly(fig2)

    return ref1, ref2, ref3, ref4, y_spectrum


@app.cell
def _(np, ref1, ref2, ref3, ref4, y_spectrum):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    data_matrix = np.vstack([
        y_spectrum,
        ref1,
        ref2,
        ref3,
        ref4
    ])

    # 2. PCA fitten (n_components = 4)
    n_components = 4
    pca = PCA(n_components=n_components)
    pca.fit(data_matrix)

    # 3. Die Hauptkomponenten
    pcs = pca.components_           # (n_components, n_points)
    explained = pca.explained_variance_ratio_
    scores = pca.transform(data_matrix)  # (n_spectra, n_components)

    # 4. Plot der PCs
    plt.figure()
    for i, pc in enumerate(pcs):
        plt.plot(pc, label=f"PC {i+1}")
    plt.legend()
    plt.title("PCA-Hauptkomponenten (Ladungsvektoren)")
    plt.show()

    # 5. Wie stark ist y_spectrum in jeder PC?
    print("Scores für y_spectrum (Projektion auf PCs):")
    for i, s in enumerate(scores[0]):
        print(f"PC{i+1}: {s: .3g}")

    print("Erklärte Varianz pro Komponente:", explained)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
