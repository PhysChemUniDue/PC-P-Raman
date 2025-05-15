import marimo

__generated_with = "0.13.9"
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
    from scipy.optimize import nnls
    return HuberRegressor, go, make_subplots, mo, nnls, np, random


@app.cell(hide_code=True)
def _(np):
    # Raman shift in cm-1
    x = np.linspace(50, 4000, 1000)

    def lorentzian(x, x0, gamma, A):
        """
        Lorentzian line shape centered at x0 with full width at half maximum gamma and amplitude A.
        """
        return A * (0.5 * gamma)**2 / ((x - x0)**2 + (0.5 * gamma)**2)


    def raman_spectrum(x, bands):
        """
        Sum of Lorentzian bands. `bands` is a list of tuples (center, width, amplitude).
        """
        y = np.zeros_like(x)
        for center, width, amp in bands:
            y += lorentzian(x, center, width, amp)
        return y

    return raman_spectrum, x


@app.cell(hide_code=True)
def _(raman_spectrum, random, x):
    # Define three example spectra with different sets of bands (center, )
    bands_list = [
        [(218, 10, 1.0), (288, 10, 0.4), (401, 10, 1.3), (602, 10, 0.5), (1200, 10, 0.8), (2300, 10, 0.5)],
        [(193, 10, 0.2), (306, 10, 0.7), (538, 10, 1.3), (668, 10, 0.5), (1157, 10, 1.4), (2323, 10, 0.8)],
        [(180, 10, 0.7), (267, 10, 0.6), (451, 10, 1.0), (558, 10, 0.6), (1254, 10, 1.0), (2215, 10, 0.4)],
    ]
    spectra = [raman_spectrum(x, bands) for bands in bands_list]

    # Random Konvolution der Spektren 
    a = round(random.uniform(0.01, 10.0), 2)
    b = round(random.uniform(0.01, 10.0), 2)
    c = round(random.uniform(0.01, 10.0), 2)

    per_a = a/ (a+b+c)*100
    per_b = b/ (a+b+c)*100
    per_c = c/ (a+b+c)*100
    y_test = a * spectra[0] + b * spectra[1] +c * spectra[2]
    return per_a, per_b, per_c, spectra, y_test


@app.cell(hide_code=True)
def _(mo):

    # Create sliders using Marimo's slider API (0–10)
    s1 = mo.ui.slider(
        label="Spektrum A", start=0, stop=10, step=0.01, value=1,
    )
    s2 = mo.ui.slider(
        label="Spektrum B", start=0, stop=10, step=0.01, value=1,
    )
    s3 = mo.ui.slider(
        label="Spektrum C", start=0, stop=10, step=0.01, value=1,
    )
    # Arrange sliders horizontally
    mo.hstack([s1, s2, s3])

    return s1, s2, s3


@app.cell(hide_code=True)
def _(HuberRegressor, nnls, np, s1, s2, s3, spectra, y_test):
    # Spektren mit Faktor
    y1 = s1.value * spectra[0]
    y2 = s2.value * spectra[1]
    y3 = s3.value * spectra[2]
    # Kombination 
    y_combined = y1+y2+y3
    y_residual = y_test - y_combined

    # Designmatrix wie oben
    X = np.vstack([spectra[0], spectra[1], spectra[2]]).T
    y = y_test

    # Huber-Regressor (default epsilon=1.35)
    model = HuberRegressor(fit_intercept=False)
    model.fit(X, y)
    a_huber, b_huber, c_huber = model.coef_
    per_a_huber = a_huber/(a_huber+b_huber+ c_huber ) * 100
    per_b_huber = b_huber/(a_huber+b_huber+ c_huber ) * 100
    per_c_huber = c_huber/(a_huber+b_huber+ c_huber ) * 100


    y_fit_huber = model.predict(X)
    # NNLS lösen
    coeffs, rnorm = nnls(X, y)
    a_nnls, b_nnls, c_nnls = coeffs
    per_a_nnls = a_nnls/(a_nnls+b_nnls+ c_nnls ) * 100
    per_b_nnls = b_nnls/(a_nnls+b_nnls+ c_nnls ) * 100
    per_c_nnls = c_nnls/(a_nnls+b_nnls+ c_nnls ) * 100


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
        y_combined,
        y_fit_huber,
        y_residual,
        y_residual_fit,
    )


@app.cell(hide_code=True)
def _(
    go,
    make_subplots,
    mo,
    per_a,
    per_a_nnls,
    per_b,
    per_b_nnls,
    per_c,
    per_c_nnls,
    x,
    y1,
    y2,
    y3,
    y_combined,
    y_fit_huber,
    y_residual,
    y_residual_fit,
    y_test,
):
    # Set up subplots: 2 rows, shared x-axis, with different heights
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.8, 0.2],
        vertical_spacing=0.05,
    )

    # Main plot traces in first row
    fig.add_trace(go.Scatter(
        x=x, y=y1, mode='lines', name='Spektrum A',
        line=dict(color='green', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=y2, mode='lines', name='Spektrum B',
        line=dict(color='orange', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=x, y=y3, mode='lines', name='Spektrum C',
        line=dict(color='purple', width=2, dash='dash'), opacity=0.8
    ), row=1, col=1)
    # Combined spectrum 
    fig.add_trace(go.Scatter(x=x, y=y_combined, mode='lines', name='Kombiniertes Slider-Spektrum',
                             line=dict(color='black', width=2)), row=1, col=1)
    # Test spectrum 
    fig.add_trace(go.Scatter(x=x, y=y_test, mode='lines', name=f'Test-Spektrum (A={per_a:.2f}, B={per_b:.2f}, C={per_c:.2f})',
                             line=dict(color='red', width=2)), row=1, col=1)
    # Fit 

    fig.add_trace(go.Scatter(x=x, y=y_fit_huber, mode='lines', name=f'Fitted (A={per_a_nnls:.2f}), B={per_b_nnls:.2f}), C={per_c_nnls:.2f})', line=dict(color='blue', width=2)), row=1, col=1)

    # Residuals in second row
    fig.add_trace(go.Scatter(x=x, y=y_residual, mode='lines', name='Residuals'), row=2, col=1)
    # Residuals in second row
    fig.add_trace(go.Scatter(x=x, y=y_residual_fit, mode='lines', name='Residuals-Fit'), row=2, col=1)


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


@app.cell(hide_code=True)
def _():
    return


if __name__ == "__main__":
    app.run()
