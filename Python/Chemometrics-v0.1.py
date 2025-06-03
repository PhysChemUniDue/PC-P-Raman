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
    return go, make_subplots, mo, np, os, sparse, spsolve


@app.cell(hide_code=True)
def _(np, sparse, spsolve, x0):
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


@app.cell
def _(np):
    # Daten Laden
    data = np.loadtxt('Hier den Pfad zur .tsv Datei einfuegen', delimiter='\t', skiprows=8)


    return (data,)


@app.cell
def _(baseline_als, data, norm, np, os):
    folder = '../Data/Literatur/'
    # Referenzspektren laden (Hämatit, Magnetit, Goethit, Lepidokrit)
    hematite = np.loadtxt(os.path.join(folder, 'Hämatit - Fe2O3', 'fe2o3-processed-2.txt'), delimiter=',', skiprows=10)
    magnetite = np.loadtxt(os.path.join(folder, 'Magnetit-Fe3O4', 'fe3o4-processed.txt'), delimiter=',', skiprows=10)
    goethite = np.loadtxt(os.path.join(folder, 'Goethit - a-FeOOH', 'a-feooh-raw.txt'), delimiter=',', skiprows=10)
    lepido = np.loadtxt(os.path.join(folder, 'Lepidokrit - g-FeOOH', 'g-feooh-processed.txt'), delimiter=',', skiprows=10)


    # 1. Gemeinsamen x-Bereich bestimmen (Schnittmenge aller Intervalle, um Extrapolation zu vermeiden)
    x_min = max(hematite[:,0].min(),
                magnetite[:,0].min(),
                goethite[:,0].min(),
                lepido[:,0].min(),
                data[:,0].min()
               )
    x_max = min(hematite[:,0].max(),
                magnetite[:,0].max(),
                goethite[:,0].max(),
                lepido[:,0].max(),
                data[:,0].max()
               )

    # 2. Gemeinsamen x-Vektor erzeugen (z.B. 1000 Punkte)
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

    y_spectrum = norm(np.interp(common_x, data[:, 0], data[:, 1]))

    spectra = [y_hem,y_mag,y_goet,y_lepid]



    return common_x, spectra, y_spectrum


@app.cell(hide_code=True)
def _(mo):
    # Create sliders using Marimo's slider API (0–10)
    s1 = mo.ui.slider(
        label="Hämatit", start=0, stop=1, step=0.01, value=0.01,
    )
    s2 = mo.ui.slider(
        label="Magnetit", start=0, stop=1, step=0.01, value=0.01,
    )
    s3 = mo.ui.slider(
        label="Goethit", start=0, stop=1, step=0.01, value=0.01,
    )
    s4 = mo.ui.slider(
        label="Lepidokrit", start=0, stop=1, step=0.01, value=0.01,
    )
    s5 = mo.ui.slider(
        label="Baseline - m", start=-1e-4, stop=1e-4, step=1e-6, value=0
    )
    s6 =  mo.ui.slider(
        label="Baseline - b", start=-1, stop=1, step=0.01, value=0
    )
    # Arrange sliders horizontally
    mo.hstack([s1, s2, s3,s4,s5,s6])

    return s1, s2, s3, s4, s5, s6


@app.cell(hide_code=True)
def _(
    common_x,
    go,
    make_subplots,
    mo,
    per_a,
    per_b,
    per_c,
    per_d,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    spectra,
    y_spectrum,
):
    # Spektren mit Faktor
    y1 = s1.value * spectra[0]
    y2 = s2.value * spectra[1]
    y3 = s3.value * spectra[2]
    y4 = s4.value * spectra[3]

    # Kombination 
    y_combined = y1+y2+y3+y4 + s5.value * common_x + s6.value
    y_residual = y_spectrum - y_combined

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

    fig.add_trace(go.Scatter(
        x=common_x, y=s5.value*common_x + s6.value, mode='lines', name='Baseline',
        line=dict(color='black', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)

    # Combined spectrum 
    fig.add_trace(go.Scatter(x=common_x, y=y_combined, mode='lines', name=f'Fit (A={s_a:.2f}, B={s_b:.2f}, C={s_c:.2f}, D={s_d:.2f})',
                             line=dict(color='black', width=2)), row=1, col=1)
    # Test spectrum 
    fig.add_trace(go.Scatter(x=common_x, y=y_spectrum, mode='lines', name=f'Spektrum (A={per_a:.2f}, B={per_b:.2f}, C={per_c:.2f}, D={per_d:.2f})',
                             line=dict(color='red', width=2)), row=1, col=1)


    # Residuals in second row
    fig.add_trace(go.Scatter(x=common_x, y=y_residual, mode='lines', name='Residuals',line=dict(color='steelblue')), row=2, col=1)



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
