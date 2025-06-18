import marimo

__generated_with = "0.12.2"
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
        FloatSlider,
        HuberRegressor,
        argrelextrema,
        curve_fit,
        go,
        interact,
        make_subplots,
        mo,
        nnls,
        np,
        os,
        random,
        sparse,
        spsolve,
    )


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
        Sum of Lorentzian bands. bandsbands is a list of tuples (center, width, amplitude).
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
    return baseline_als, gauss, line, lorentzian, norm, raman_spectrum


@app.cell
def _(np):
    # Daten der Mischung laden
    data = np.loadtxt('../Data/SS25Gr1/Mischung1/001.tsv', delimiter='\t', skiprows=8)
    dark_data = np.loadtxt('../Data/SS25Gr1/Silicium/001.tsv', delimiter='\t', skiprows=8)


    return dark_data, data


@app.cell
def _(dark_data, data, norm, np):
    # Referenzspektren laden (Hämatit, Magnetit, Goethit, Lepidokrit)
    kaffee = np.loadtxt('../Data/SS25Gr1/Kaffee/001.tsv', delimiter='\t', skiprows=8)
    dark_kaffee = np.loadtxt('../Data/SS25Gr1/Kaffee/002.tsv', delimiter='\t', skiprows=8)
    ibu = np.loadtxt('../Data/SS25Gr1/Ibuprofen/001.tsv', delimiter='\t', skiprows=8)
    dark_ibu = np.loadtxt('../Data/SS25Gr1/Silicium/001.tsv', delimiter='\t', skiprows=8)

    # Dark Korrektur und Baseline
    corr_kaffee = norm(kaffee[:, 1] - dark_kaffee[:,1])
    corr_ibu = norm(ibu[:, 1] - dark_ibu[:,1])
    corr_data = norm(data[:, 1] - dark_data[:,1])

    refspectra = [corr_kaffee,corr_ibu]
    return (
        corr_data,
        corr_ibu,
        corr_kaffee,
        dark_ibu,
        dark_kaffee,
        ibu,
        kaffee,
        refspectra,
    )


@app.cell(hide_code=True)
def _(mo):
    # Create sliders using Marimo's slider API (0–10)
    s1 = mo.ui.slider(
        label="Kaffee", start=0, stop=1, step=0.01, value=0.01,
    )
    s2 = mo.ui.slider(
        label="Ibuprofen", start=0, stop=1, step=0.01, value=0.01,
    )
    s3 = mo.ui.slider(
        label="Baseline - m", start=-1e-4, stop=1e-4, step=1e-6, value=0
    )
    s4 =  mo.ui.slider(
        label="Baseline - b", start=-1, stop=1, step=0.01, value=0
    )
    # Arrange sliders horizontally
    mo.hstack([s1, s2, s3,s4])
    return s1, s2, s3, s4


@app.cell(hide_code=True)
def _(corr_data, data, go, make_subplots, mo, refspectra, s1, s2, s3, s4):
    # Spektren mit Faktor
    y1 = s1.value * refspectra[0]
    y2 = s2.value * refspectra[1]


    # Kombination 
    y_combined = y1+y2+ s3.value * data[:,0] + s4.value
    y_residual = corr_data - y_combined

    s_a = s1.value/(s1.value+s2.value+s3.value+s4.value)*100
    s_b = s2.value/(s1.value+s2.value+s3.value+s4.value)*100

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.05,
    )

    # Main plot traces in first row
    fig.add_trace(go.Scatter(
        x=data[:,0], y=y1, mode='lines', name='Kaffee',
        line=dict(color='green', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data[:,0], y=y2, mode='lines', name='Ibuprofen',
        line=dict(color='coral', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=data[:,0], y=s3.value*data[:,0] + s4.value, mode='lines', name='Baseline',
        line=dict(color='black', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)

    # Combined spectrum 
    fig.add_trace(go.Scatter(x=data[:,0], y=y_combined, mode='lines', name=f'Fit (A={s_a:.2f}, B={s_b:.2f})',
                             line=dict(color='black', width=2)), row=1, col=1)
    # Experimentelles spectrum 
    fig.add_trace(go.Scatter(x=data[:,0], y=corr_data, mode='lines', name=f'Spektrum',
                             line=dict(color='red', width=2)), row=1, col=1)


    # Residuals in second row
    fig.add_trace(go.Scatter(x=data[:,0], y=y_residual, mode='lines', name='Residuals',line=dict(color='steelblue')), row=2, col=1)



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
    return fig, s_a, s_b, y1, y2, y_combined, y_residual


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
