import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import random
    import numpy as np
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    from ipywidgets import interact, FloatSlider
    from sklearn.linear_model import HuberRegressor
    from scipy.optimize import nnls,curve_fit
    from scipy.signal import argrelextrema
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    import os 
    return (
        HuberRegressor,
        go,
        make_subplots,
        mo,
        nnls,
        np,
        os,
        random,
        sparse,
        spsolve,
    )


@app.cell
def _(np, sparse, spsolve, x0):
    # Raman shift in cm-1
    x = np.linspace(50, 4000, 1000)

    def lorentzian(x, x0, gamma, A):
        """
        Lorentzian line shape centered at x0 with full width at half maximum gamma and amplitude A.
        """
        return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2)

    def gauss(x,A,sigma):
        """
        Gaussian line shape centered at x0 with a standard derivation of sigma and amplitude A.
        """
        return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

    def line(x,m,b):
        """
        Line
        """
        return m*x + b

    def raman_spectrum(x, bands):
        """
        Sum of Lorentzian bands. `bands` is a list of tuples (center, width, amplitude).
        """
        y = np.zeros_like(x)
        for center, width, amp in bands:
            y += lorentzian(x, center, width, amp)
        return y

    def norm(A):
        return A /np.max(A)  

    def baseline_als(y, lam=1e5, p=0.01, niter=10):
        """
        Schätzt die Baseline eines Spektrums y per asymmetrischem Least-Squares (Eilers' Methode).

        Parameter:
        -----------
        y    : array_like
               Ursprüngliches Spektrum (1D-Array).
        lam  : float
               Glättungsparameter (>0), hoher Wert => glattere Baseline.
        p    : float
               Asymmetrieparameter zwischen 0 und 1 (kleiner p => stärkere Unterdrückung von Ausreißern nach oben).
        niter: int
               Anzahl der Iterationen für die Gewichtsaktualisierung.

        Rückgabe:
        --------
        baseline : numpy.ndarray
                   Geschätzte Baseline.
        y_corr   : numpy.ndarray
                   Baseline-korrigiertes Spektrum (y - baseline).
        """

        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = D.dot(D.T)  # zweite Ableitung
        w = np.ones(L)
        for i in range(niter):
            W = sparse.diags(w, 0)
            Z = W + lam * D
            baseline = spsolve(Z, w * y)
            # Update weights: Gewichte kleiner für Punkte über der Baseline
            w = p * (y > baseline) + (1 - p) * (y < baseline)
        y_corr = y - baseline
        return baseline, y_corr

    return baseline_als, norm


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Raman-Spektrum-Viewer

    Dieses Notebook erlaubt es, einen Ordnerpfad einzugeben, daraus alle `.tsv`-Dateien auszuwählen und das Spektrum zu plotten.
    """
    )
    return


@app.cell(hide_code=True)
def _(folder, mo, os):
    folder_spectra = '../Data/Eisen1/2025-05-19/'
    # Alle .tsv-Dateien im Ordner sammeln
    tsv_files = [f for f in os.listdir(folder_spectra) if f.lower().endswith('.tsv')]
    if not tsv_files:
        raise FileNotFoundError(f"Keine .tsv-Dateien in {folder!r} gefunden.")

    dropdown_form = mo.ui.dropdown(
        options=tsv_files,
        value=tsv_files[0],
        label="Spektrum auswählen",
        searchable=True,
        full_width=True
    ).form()

    # 2. Nur die Form ausgeben
    dropdown_form
    return dropdown_form, folder_spectra


@app.cell
def _(dropdown_form, folder_spectra, go, mo, np, os):
    # 1. Solange noch kein Wert da ist, die Cell abbrechen
    mo.stop(dropdown_form.value is None, "Bitte wähle eine Datei und klicke Submit")

    # 2. Datei einlesen (skip 8 header‐Zeilen)
    path = os.path.join(folder_spectra, dropdown_form.value)
    data = np.loadtxt(path, delimiter='\t', skiprows=8)

    calibration = np.loadtxt('../Data/Eisen1/2025-05-19/wn-calibration.tsv', delimiter='\t', skiprows=8)
    Si_peak = 520.45
    wn_corr =  calibration[np.argmax(calibration[:,1]),0]-Si_peak

    wn_spectrum = data[:, 0] -wn_corr
    int_spectrum = data[:, 1]
    # 3. Plot erstellen und nur das Figure-Objekt zurückgeben
    measurement = go.Scatter(
        x=data[:, 0] -wn_corr,
        y=data[:, 1],
        mode='lines',
        name='Raman-Spektrum'
    )

    layout = go.Layout(
        title=f'Raman-Spektrum: {dropdown_form.value}',
        xaxis=dict(title='Wellenzahl (cm⁻¹)'),
        yaxis=dict(title='Raman Intensität'),
        showlegend=False
    )
    fig2 = go.Figure(data=[measurement], layout=layout)
    fig2.update_layout(
        template='simple_white',
        xaxis_showgrid=True,
        yaxis_showgrid=True
    )

    mo.ui.plotly(fig2)

    return (wn_spectrum,)


@app.cell
def _(baseline_als, norm, np, os, wn_spectrum):
    folder = '../Data/Literatur/'
    hematite  = np.loadtxt(os.path.join(folder, 'Hämatit - Fe2O3'     ,'fe2o3-processed.txt'  ),delimiter=',', skiprows=10)
    magnetite = np.loadtxt(os.path.join(folder, 'Magnetit-Fe3O4'      ,'fe3o4-processed.txt'  ),delimiter=',', skiprows=10)
    goethite  = np.loadtxt(os.path.join(folder, 'Goethit - a-FeOOH'   ,'a-feooh-raw.txt'),delimiter=',', skiprows=10)
    lepido    = np.loadtxt(os.path.join(folder, 'Lepidokrit - g-FeOOH','g-feooh-processed.txt'),delimiter=',', skiprows=10)

    # 1. Gemeinsamen x-Bereich bestimmen (Schnittmenge aller Intervalle, um Extrapolation zu vermeiden)
    x_min = max(hematite[:,0].min(),
                magnetite[:,0].min(),
                goethite[:,0].min(),
                lepido[:,0].min(),
                wn_spectrum.min()
               )
    x_max = min(hematite[:,0].max(),
                magnetite[:,0].max(),
                goethite[:,0].max(),
                lepido[:,0].max(),
                wn_spectrum.max()
               )

    # 2. Gemeinsamen x-Vektor erzeugen (z.B. 1000 Punkte)
    common_x = np.linspace(x_min, x_max, 1000)

    # 3. Mittels np.interp auf den gemeinsamen Vektor routen
    y_hem_raw   = norm(np.interp(common_x, hematite[:,0],  hematite[:,1]))
    y_hem = norm(baseline_als(y_hem_raw,lam=1e6,p=0.001,niter=10)[1])
    y_mag_raw   = norm(np.interp(common_x, magnetite[:,0], magnetite[:,1]))
    y_mag = norm(baseline_als(y_mag_raw,lam=1e6,p=0.001,niter=10)[1])
    y_goet_raw  = norm(np.interp(common_x, goethite[:,0],  goethite[:,1]))
    y_goet = norm(baseline_als(y_goet_raw,lam=1e6,p=0.001,niter=10)[1])
    y_lepid_raw = norm(np.interp(common_x, lepido[:,0],    lepido[:,1]))
    y_lepid    = norm(baseline_als(y_lepid_raw,lam=1e6,p=0.001,niter=10)[1])
    y_spectrum = norm(np.interp(common_x, wn_spectrum,    lepido[:,1]))

    # Lokale Maxima = 
    lm_hem = [17,35,82,194]

    # 4. Spektren mit summe an Lorentzfunktionen und baseline fitten
    # Initial Guesses
    m,n = 0.0,0.0


    # Bounds for parameters: spectra & baselines non-negative where applicable
    #lower = [0, 0, 0, 0,    0, common_x.min(), 1e-3, -np.inf, -np.inf]
    #upper = [np.inf, np.inf, np.inf, np.inf, np.inf, common_x.max(), common_x.max(), np.inf, np.inf]

    # Perform fit
    #popt, pcov = curve_fit(full_model, common_x, y_meas, p0=p0, bounds=(lower, upper))
    # Extract fitted parameters
    #a_fit, b_fit, c_fit, d_fit, A_g_fit, mu_fit, sigma_fit, m_fit, b0_fit = popt

    return common_x, folder, y_goet, y_hem, y_lepid, y_mag


@app.cell
def _(random, y_goet, y_hem, y_lepid, y_mag):
    spectra = [y_hem,y_mag,y_goet,y_lepid]

    # Random Konvolution der Spektren 
    a = round(random.uniform(0.01, 10.0), 2)
    b = round(random.uniform(0.01, 10.0), 2)
    c = round(random.uniform(0.01, 10.0), 2)
    d = round(random.uniform(0.01, 10.0), 2)

    per_a = a/ (a+b+c+d) * 100 
    per_b = b/ (a+b+c+d) * 100 
    per_c = c/ (a+b+c+d) * 100 
    per_d = d/ (a+b+c+d) * 100 

    y_test = a * spectra[0] + b * spectra[1] +c * spectra[2] + d * spectra[3]
    return per_a, per_b, per_c, per_d, spectra, y_test


@app.cell(hide_code=True)
def _(mo):
    # Create sliders using Marimo's slider API (0–10)
    s1 = mo.ui.slider(
        label="Hämatit", start=0, stop=10, step=0.01, value=1,
    )
    s2 = mo.ui.slider(
        label="Magnetit", start=0, stop=10, step=0.01, value=1,
    )
    s3 = mo.ui.slider(
        label="Goethit", start=0, stop=10, step=0.01, value=1,
    )
    s4 = mo.ui.slider(
        label="Lepidokrit", start=0, stop=10, step=0.01, value=1,
    )
    # Arrange sliders horizontally
    mo.hstack([s1, s2, s3,s4])

    return s1, s2, s3, s4


@app.cell(hide_code=True)
def _(HuberRegressor, nnls, np, s1, s2, s3, s4, spectra, y_test):
    # Spektren mit Faktor
    y1 = s1.value * spectra[0]
    y2 = s2.value * spectra[1]
    y3 = s3.value * spectra[2]
    y4 = s4.value * spectra[3]
    # Kombination 
    y_combined = y1+y2+y3+y4
    y_residual = y_test - y_combined

    # Designmatrix wie oben
    X = np.vstack([spectra[0], spectra[1], spectra[2], spectra[3]]).T
    y = y_test

    # Huber-Regressor (default epsilon=1.35)
    model = HuberRegressor(fit_intercept=False)
    model.fit(X, y)
    a_huber, b_huber, c_huber,d_huber = model.coef_
    per_a_huber = a_huber/(a_huber+b_huber+ c_huber + d_huber) * 100
    per_b_huber = b_huber/(a_huber+b_huber+ c_huber + d_huber) * 100
    per_c_huber = c_huber/(a_huber+b_huber+ c_huber + d_huber) * 100
    per_d_huber = d_huber/(a_huber+b_huber+ c_huber + d_huber) * 100

    y_fit_huber = model.predict(X)
    # NNLS lösen
    coeffs, rnorm = nnls(X, y)
    a_nnls, b_nnls, c_nnls,d_nnls = coeffs
    per_a_nnls = a_nnls/(a_nnls+b_nnls+ c_nnls +d_nnls) * 100
    per_b_nnls = b_nnls/(a_nnls+b_nnls+ c_nnls +d_nnls) * 100
    per_c_nnls = c_nnls/(a_nnls+b_nnls+ c_nnls +d_nnls) * 100
    per_d_nnls = d_nnls/(a_nnls+b_nnls+ c_nnls +d_nnls) * 100


    # Gefittetes Spektrum rekonstruieren
    y_fit_nnls = X.dot(coeffs)

    y_residual_fit  = y_fit_nnls - y_test
    return (
        per_a_nnls,
        per_b_nnls,
        per_c_nnls,
        y1,
        y2,
        y3,
        y4,
        y_combined,
        y_fit_huber,
        y_residual,
        y_residual_fit,
    )


@app.cell(hide_code=True)
def _(
    common_x,
    go,
    make_subplots,
    mo,
    per_a,
    per_a_nnls,
    per_b,
    per_b_nnls,
    per_c,
    per_c_nnls,
    per_d,
    s1,
    s2,
    s3,
    s4,
    y1,
    y2,
    y3,
    y4,
    y_combined,
    y_fit_huber,
    y_residual,
    y_residual_fit,
    y_test,
):
    s_a = s1.value/(s1.value+s2.value+s3.value+s4.value)*100
    s_b = s2.value/(s1.value+s2.value+s3.value+s4.value)*100
    s_c = s3.value/(s1.value+s2.value+s3.value+s4.value)*100
    s_d = s4.value/(s1.value+s2.value+s3.value+s4.value)*100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.05,
    )

    # Main plot traces in first row
    fig.add_trace(go.Scatter(
        x=common_x, y=y1, mode='lines', name='Hämatit',
        line=dict(color='green', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=common_x, y=y2, mode='lines', name='Magnetit',
        line=dict(color='coral', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=common_x, y=y3, mode='lines', name='Goethit',
        line=dict(color='steelblue', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=common_x, y=y4, mode='lines', name='Lepidokrit',
        line=dict(color='grey', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)

    # Combined spectrum 
    fig.add_trace(go.Scatter(x=common_x, y=y_combined, mode='lines', name=f'Kombiniertes Slider-Spektrum (A={s_a:.2f}, B={s_b:.2f}, C={s_c:.2f}, D={s_d:.2f})',
                             line=dict(color='black', width=2)), row=1, col=1)
    # Test spectrum 
    fig.add_trace(go.Scatter(x=common_x, y=y_test, mode='lines', name=f'Test-Spektrum (A={per_a:.2f}, B={per_b:.2f}, C={per_c:.2f}, D={per_d:.2f})',
                             line=dict(color='red', width=2)), row=1, col=1)
    # Fit 

    fig.add_trace(go.Scatter(x=common_x, y=y_fit_huber, mode='lines', name=f'Fitted (A={per_a_nnls:.2f}, B={per_b_nnls:.2f}, C={per_c_nnls:.2f}, D={per_d:.2f})', line=dict(color='blue', width=2)), row=1, col=1)

    # Residuals in second row
    fig.add_trace(go.Scatter(x=common_x, y=y_residual, mode='lines', name='Residuals',line=dict(color='steelblue')), row=2, col=1)
    # Residuals in second row
    fig.add_trace(go.Scatter(x=common_x, y=y_residual_fit, mode='lines', name='Residuals-Fit',line=dict(color='red')), row=2, col=1)


    # Update layout
    fig.update_layout(
        margin=dict(l=0, r=00, t=50, b=0),
        plot_bgcolor='white',       # remove default background
        paper_bgcolor='white',      # remove default container color
        title='Raman-Spektren und Residuals',
        xaxis1=dict(title='Wellenzahl (cm⁻¹)'),
        yaxis1=dict(title='Intensität'),
        yaxis2=dict(title='Residuals'),
        height=800
    )

    mo.ui.plotly(fig)

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
