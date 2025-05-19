"""
---
packages:
  - numpy==1.24.3
  - pandas==2.0.2
  - matplotlib==3.7.2
  - scipy==1.10.1
---
"""
# Dieses Skript erstellt eine interaktive Marimo-App zur Analyse von Spektraldaten.
# Es lädt Rohdaten, berechnet Transmission und Extinktion, erstellt Diagramme
# und führt lineare Fits durch, um Übergänge im Spektrum zu untersuchen.

import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Importiere notwendige Pakete""")
    return


@app.cell(hide_code=True)
def _():
    import numpy as np                    # Mathepaket
    import pandas as pd                   # DataFrames
    import matplotlib.pyplot as plt       # Erstellung von Diagrammen
    import marimo as mo                   # Marimo für interaktive App-Elemente
    from scipy.optimize import curve_fit  # Paket zum Fitten
    return curve_fit, mo, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Funktionen""")
    return


@app.cell(hide_code=True)
def round_sig():
    # Wir wollen nachher unsere Werte auf eine  bestimmte Anzahl an signifikanten Stellen runden.
    def round_sig(x, sig=6):
        try:
            return float(f"{x:.{sig}g}")
        except (ValueError, TypeError):
            return x  # Falls x nicht numerisch ist (z.B. NaN oder None), wird der ursprüngliche Wert zurückgegeben.
    return (round_sig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Lade die Daten""")
    return


@app.cell(hide_code=True)
def _(np):
    # Lade die Rohdaten des Iod Spektrums aus "raw.asc" und überspringe die ersten 88 Zeilen (Kopfzeilen).
    data_raw = np.loadtxt("Data/raw.asc", skiprows=88)
    # Lade die BackgroundDaten aus "baseline.asc" auf die gleiche Weise.
    data_bl = np.loadtxt("Data/baseline.asc", skiprows=88)
    # Lade die Übergangsindizes aus den Dateien. Ich habe die damals für meine Auswertung mit Julia erstellt. Aus diesem 
    # Grund passen wir sie von 1-indexiert auf 0-indexiert an. (Julia vs. Python)
    # Diese Dateien habe ich selber erstellt. Für die Identifikation der einzelnen Übergänge lohnt sich ein Blick 
    # ins Skript oder das Paper von McNaught
    dp0 = np.trunc(np.loadtxt("Data/0_to_n.dlm")[:, 0]).astype(int) - 1
    dp1 = np.trunc(np.loadtxt("Data/1_to_n.dlm")[:, 0]).astype(int) - 1
    dp2 = np.trunc(np.loadtxt("Data/2_to_n.dlm")[:, 0]).astype(int) - 1
    return data_bl, data_raw, dp0, dp1, dp2


@app.cell(hide_code=True)
def _(data_bl, data_raw, np, pd):
    # Erstelle ein DataFrame
    df = pd.DataFrame({
        "Wavelength": data_raw[:, 0],
        "Raw": data_raw[:, 1],
        "Baseline": data_bl[:, 1]
    })

    # Berechne die Transmission, bzw. Extinktion, wenn man den richtigen ordinaten Mode wählt.
    transmission = df["Raw"] / df["Baseline"]
    # Berechne die Wellenzahl in cm-1
    wavenumber = 1 / df["Wavelength"] * 1e7
    # Berechne die Extinktion, falls du so wie den ordinaten Mode auf Transmission hattest ;)
    extinktion  = np.log(1/transmission)

    # Füge die berechneten Werte als neue Spalten zum DataFrame hinzu.
    df["Transmission"] = transmission
    df["Wavenumber"] = wavenumber
    df["Extinktion"] = extinktion

    # Zur Überprüfung kannst du dir den DataFrame anzeigen lassen, lösche dazu einfach das #
    # df
    return df, extinktion, transmission, wavenumber


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Plots
        ### Bereinigtes Spektrum von Iod
        """
    )
    return


@app.cell(hide_code=True)
def _(df, mo, plt):
    # Erstelle eine neue Abbildung für das Spektrum.
    p_spec = plt.figure()
    # Zeichne das bereinigte Spektrum
    plt.plot(df["Wavelength"], df["Extinktion"])
    plt.xlabel("Wellenlänge' in nm")
    plt.ylabel("Extinktion in %")
    mo.mpl.interactive(plt.gcf()) # Interactive Plot
    return (p_spec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Zeichne die Übergänge in das Spektrum ein
        Erstelle dafür zuächst eine Datei mit den jeweiligen Indices der Übergänge in der Spektren
        """
    )
    return


@app.cell(hide_code=True)
def _(df, dp0, dp1, dp2, mo, plt):
    # Erstelle eine neue Abbildung für das Spektrum mit den verschiedenen Übergängen
    p_spec_dp = plt.figure()

    # Zeichne das Spektrum
    plt.plot(df["Wavelength"], df["Extinktion"], linewidth=0.5, label="Data")

    # Zeichne die Übergangserien aus den verschieden vibratorischen Zuständen des elektr. GZ
    # Übergang von vibr. GZ
    plt.plot(df["Wavelength"].iloc[dp0], df["Extinktion"].iloc[dp0], ".", label=r"v$^{\prime\prime}_0\rightarrow$v$^{\prime}_n$")
    # Übergang von 1. angeregten vibr. Zustand
    plt.plot(df["Wavelength"].iloc[dp1], df["Extinktion"].iloc[dp1], ".", label=r"v$^{\prime\prime}_1\rightarrow$v$^{\prime}_n$")
    # Übergang von 2. angeregten vibr. Zustand
    plt.plot(df["Wavelength"].iloc[dp2], df["Extinktion"].iloc[dp2], ".", label=r"v$^{\prime\prime}_2\rightarrow$v$^{\prime}_n$")
    plt.xlabel("Wellenlänge in nm")
    plt.ylabel("Extinktion in %")
    plt.xlim(510, 615)  # Begrenze den sichtbaren Bereich der x-Achse
    plt.legend()        # Zeige die Legende an
    mo.mpl.interactive(plt.gcf()) # Interactive Plot
    return (p_spec_dp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Deslandre Tabelle""")
    return


@app.cell(hide_code=True)
def _(df, dp0, dp1, dp2, np, pd):
    # Erstelle die 'deslandre' Tabelle mit der Spalte 'vn' für die vibr. Zustände des 2. ange. elek. Zustandes
    deslandre = pd.DataFrame()
    deslandre["vn"] = np.arange(9, 45)  # ich konnte insgesamt die vibr. Zustände 9 bis 45 beobachten
    n = len(deslandre["vn"])  # Anzahl der Einträge, damit wir neue Spalten in der Tabelle erstellen können

    # Erstelle leere Spalten für die Wellenlängen (wl0, wl1, wl2) und fülle sie mit NaN.
    # Für die Übergänge von vibr. GZ in vn
    deslandre["wl0"] = np.full(n, np.nan)
    # Für die Übergänge von 1. ange. vibr. Zustand in vn
    deslandre["wl1"] = np.full(n, np.nan)
    # Für die Übergänge von 2. ange. vibr. Zustand in vn
    deslandre["wl2"] = np.full(n, np.nan)

    # Definiere Startpositionen für die Zuordnung der jeweligen Wellenlängen 
    dp0_start = 5   # Für die Übergänge von vibr. GZ in vn sehen wir erst eine Bande ab der 6. Reihe in der Tabelle
    dp1_start = 5   # Für die Übergänge von 1. ange. vibr. Zustand in vn genauso
    dp2_start = 0   # Für die Übergänge von 2. ange. vibr. Zustand in vn 

    # Wir extrahiern nun die Wellenlängen an den aus dp0/dp1/dp2 festgelegten Indices in unserem Spektrum
    wl0 = df["Wavelength"].iloc[dp0]  # Wellenlängen für dp0
    wl1 = df["Wavelength"].iloc[dp1]  # Wellenlängen für dp1
    wl2 = df["Wavelength"].iloc[dp2]  # Wellenlängen für dp2

    # Weise die extrahierten Wellenlängen den entsprechenden Bereichen in der 'deslandre'-Tabelle zu.
    deslandre.loc[deslandre.index[dp0_start : dp0_start + len(wl0)], "wl0"] = wl0.values
    deslandre.loc[deslandre.index[dp1_start : dp1_start + len(wl1)], "wl1"] = wl1.values
    deslandre.loc[deslandre.index[dp2_start : dp2_start + len(wl2)], "wl2"] = wl2.values

    deslandre
    return deslandre, dp0_start, dp1_start, dp2_start, n, wl0, wl1, wl2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Modifizierte Deslandre Tabelle""")
    return


@app.cell(hide_code=True)
def _(deslandre, np, round_sig):
    # Wir berechnen die Wellenzahl
    deslandre["wn0"] = 1 / deslandre["wl0"] * 1e7
    deslandre["wn1"] = 1 / deslandre["wl1"] * 1e7
    deslandre["wn2"] = 1 / deslandre["wl2"] * 1e7

    # Wir berechnen die Differenzen zwischen den vibr. Niveas des elek. GZ
    deslandre["d01"] = deslandre["wn0"] - deslandre["wn1"]
    deslandre["d12"] = deslandre["wn1"] - deslandre["wn2"]

    # Wir berechnen die Differenzen zwischen den benachbarten vibr. Niveas des 2. angeregten elek. Zustand
    d0 = np.append(np.diff(deslandre["wn0"].values), np.nan)
    d1 = np.append(np.diff(deslandre["wn1"].values), np.nan)
    d2 = np.append(np.diff(deslandre["wn2"].values), np.nan)

    # Füge die berechneten Differenzen als neue Spalten zur 'deslandre'-Tabelle hinzu
    deslandre["d0"] = d0
    deslandre["d1"] = d1
    deslandre["d2"] = d2

    # Runde alle relevanten Spaltenwerte auf 6 signifikante Stellen.
    for col in ["wn0", "wn1", "wn2", "d01", "d12", "d0", "d1", "d2"]:
        deslandre[col] = deslandre[col].apply(lambda x: round_sig(x, 6))
    deslandre
    return col, d0, d1, d2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Birge-Sponer-Plot - Angeregter elektronischer Zustand
        Morse-Potential in Schrödinger-Gleichung gibt Eigenwertsgleichung für die vibr. Energielevels im elek. Zustand:

        \begin{equation}
        E'(\nu) = \tilde{\nu}_e'\left(\nu + \frac{1}{2} \right) - \tilde{\nu}_e'\chi'_e \left(\nu + \frac{1}{2} \right)^2
        \end{equation}

        Birge-Sponer-Gleichung (Integralkorrektur mit $\nu +\frac{1}{2}$)

        \begin{equation}
        \Delta E'(\nu) = \tilde{\nu}_e' -  \tilde{\nu}_e' \chi'_e - 2 \tilde{\nu}_e' \chi'_e\left(\nu +\frac{1}{2}\right) 
        \tag{4.2}
        \end{equation}

        Die Birge-Sponer Gleichung ist eine lineare Funktion: 

        \begin{equation}
        \Delta E(x) = mx + b
        \tag{4.2}
        \end{equation}
        """
    )
    return


@app.cell(hide_code=True)
def _(curve_fit, deslandre, mo, np, plt):
    # Alle Differenzen beschreiben den Abstand der benachabarten vibr. Zustände im 2. angeregten elek. Zustand.
    # Demnach können alle identifizierten Übergänge auch in einem Plot zusammengefasst werden. Nicht in drei, das ist sinnlos
    # Wir packen demnach alle Quantenzahlen der drei Übergangsserien zusammen 
    v_1 = np.concatenate((
        deslandre["vn"].iloc[5:35].values,  # Übergänge von vibr. GZ
        deslandre["vn"].iloc[5:19].values,  # Übergänge von 1. ange. vibr. Zustand
        deslandre["vn"].iloc[0:12].values   # Übergänge von 2. ange. vibr. Zustand
    )) + 0.5                               # Integralkorrektur nach Birge-Sponer!

    # Nun fassen wir die bestimmten Energiedifferenzen der drei Übergangsserien zusammen. Analog zu oben
    dnu_1  = np.concatenate((
        deslandre["d0"].iloc[5:35].values, # Übergänge von vibr. GZ
        deslandre["d1"].iloc[5:19].values, # Übergänge von 1. ange. vibr. Zustand
        deslandre["d2"].iloc[0:12].values  # Übergänge von 2. ange. vibr. Zustand
    ))

    # Wir erwarten eine lineare Verteilung der Energiedifferenzenzen zwischen benachbarten vibr. Zuständen
    # Demnach brauchen wir eine lineare Fitfunktion:
    # x: Schwingungsquantenzahl
    # m: Steigung
    # b: y-Achsenabschnitt
    def model(x, m,b):
        return m * x  + b

    # Führe den Fit für den 2. angeregten elek. Zustand durch
    popt, pcov = curve_fit(model, v_1, dnu_1, p0=[-1.0, 150.0])

    # Erstelle einen Textstring, der die Fit-Parameter m und b anzeigt. 
    # Damit wir diesen nachher im Spektrum anzeigen können
    fit_text = f"m' = {popt[0]:.3g}\nb' = {popt[1]:.4g}"

    # Erstelle eine neue Abbildung für den Birge-Sponer-Plot des angeregten elek. Zustands.
    p_bs_1 = plt.figure()
    # Zeichne die Datenpunkte ein.
    plt.plot(v_1, dnu_1, "o", label="Data")
    # Zeichne den linearen Fit.
    plt.plot(v_1, model(v_1, *popt), 'k', label='Fit')
    # Füge den Fit-Text an einer definierten Position (38,95) in dem Plot ein.
    plt.text(38, 95, fit_text, fontsize=12, verticalalignment='top')
    # Achsenbeschriftung, Titel und Legende nicht vergessen
    plt.xlabel(r"v$+\frac{1}{2}$")
    plt.ylabel(r"$\Delta \tilde{\nu}_{n+1,n}$")
    plt.title(r"Birge-Sponer-Plot für v $^{\prime}$")
    plt.legend()
    mo.mpl.interactive(plt.gcf()) # Interactive Plot
    return dnu_1, fit_text, model, p_bs_1, pcov, popt, v_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Berechnung der Parameter:

        \begin{equation}
        \tilde{\nu}_e' = b' - \frac{m'}{2}
        \end{equation}

        \begin{equation}
        \chi_e' = -\frac{m'}{2\tilde{\nu}_e'}
        \end{equation}

        \begin{equation}
        k' = \left(2\pi c \tilde{\nu}_e' \right)^2
        \end{equation}

        \begin{equation}
        \nu_{max}' = -\frac{b'}{m'}
        \end{equation}

        \begin{equation}
        D_0' = \int_0^{\nu_{max}'} m'x+b'
        \end{equation}

        \begin{equation}
        D_E' = D_0' + E'(0)
        \end{equation}
        """
    )
    return


@app.cell(hide_code=True)
def _(np, pd, popt):
    # Für die einfachere Nutzung extrahieren wir die Steigung und den y-Achsenabschnitt
    m_1 = popt[0]
    b_1 = popt[1]
    # Wir benötigen für die Berechnungen die Lichtgeschwindigkeit, atomare Masserneinheit und Iods Masse
    c = 2.9979E8
    u = 1.66053E-27 
    m_I2 = 126.9 * u
    mu_I2 = m_I2/2

    # Anschließend können wir die Parameter berechen
    nu_e1 = b_1 - m_1/2
    chi_e1 = -m_1/(2*nu_e1)
    k_1 = (2*np.pi*c*nu_e1 * 100)**2  * mu_I2
    v_max1 = - b_1/m_1
    D_01 = 0.5*m_1*v_max1**2 + b_1 *v_max1
    E_01  = nu_e1/2 + nu_e1*chi_e1/4
    D_E1 = D_01 + E_01

    # Definition der Variablen, Einheiten und Startwerte
    data_1 = {
        "nu_e'": ["[cm-1]", nu_e1],
        "chi_e'": ["[–]", chi_e1],
        "k'": ["[N/m]", k_1],
        "nu_{max}'": ["[–]", v_max1],
        "D_0'": ["[cm-1]", D_01],
        "D_E'": ["[cm-1]", D_E1],
    }

    # Transponieren und Umwandeln in DataFrame
    df_1 = pd.DataFrame(data_1, index=["Einheit", "Wert"])

    # Ausgabe anzeigen (Marimo zeigt DataFrames automatisch an)
    df_1
    return (
        D_01,
        D_E1,
        E_01,
        b_1,
        c,
        chi_e1,
        data_1,
        df_1,
        k_1,
        m_1,
        m_I2,
        mu_I2,
        nu_e1,
        u,
        v_max1,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Birge-Sponer-Plot - Elektronischer Grundzustand
        Morse-Potential in Schrödinger-Gleichung gibt Eigenwertsgleichung für die vibr. Energielevels im elek. Zustand:

        \begin{equation}
        E''(\nu) = \tilde{\nu}_e{''}\left(\nu + \frac{1}{2} \right)- \tilde{\nu}_e{''}\chi{''}_e \left(\nu + \frac{1}{2} \right)^2
        \end{equation}

        Birge-Sponer-Gleichung (Integralkorrektur mit $\nu +\frac{1}{2}$)

        \begin{equation}
        \Delta E{''}(\nu) = \tilde{\nu}_e{''} -  \tilde{\nu}_e{''} \chi{''}_e - 2 \tilde{\nu}_e{''} \chi{''}_e\left(\nu +\frac{1}{2}\right) 
        \end{equation}

        Die Birge-Sponer Gleichung ist eine lineare Funktion: 

        \begin{equation}
        \Delta E(x) = a - bx
        \end{equation}
        """
    )
    return


@app.cell(hide_code=True)
def _(curve_fit, deslandre, mo, model, np, plt):
    # Wir regen aus dem Grund, dem 1. und 2. vibr. Zustand des. elektronisches Zustands an
    # In d01 und d12 haben wir bereits den Energieunterschied zwischen den benachbarten Zuständen erstellt
    # Nun fassen wir diese Datenpunke in einem Array zusammen und stellen in dar
    dnu_0 = np.concatenate((
        deslandre["d01"].iloc[5:20].values,
        deslandre["d12"].iloc[5:13].values
    ))

    # Die quantenzahlen sind entweder 0 oder 1 mit der respektiven Länge der d01, d12 arrays.
    # Wir berücksichtigen die Integralkorrektur und erhalten so
    v_0 = np.zeros(len(dnu_0))
    # Länge von d01
    v_0[:15] = 0.5
    # Läange von d12
    v_0[15:] = 1.5

    # Fitfür den elektronischen Grundzustand.
    popt_0, pcov_0 = curve_fit(model, v_0, dnu_0, p0=[-1.0, 150.0])
    # Erstelle einen Textstring mit den Fit-Parametern, den wir dann nachher im Plot zeigen.
    fit_text_0 = f"m'' = {popt_0[0]:.3g}\nb'' = {popt_0[1]:.4g}"

    # Erstelle eine neue Abbildung für den Birge-Sponer-Plot des elektronischen Grundzustands.
    p_bs_0 = plt.figure()
    # Zeichne die Datenpunkte ein.
    plt.plot(v_0, dnu_0, "o", label="Data")
    # Zeichne den linearen Fit ein.
    plt.plot(v_0, model(v_0, *popt_0), 'k', label='Fit')
    # Zeige die Fitparameter an der Stelle (1.35,215.7).
    plt.text(1.3, 215.7, fit_text_0, fontsize=12, verticalalignment='top')
    # Achsenbeschriftung, Titel und Legende nicht vergessen.
    plt.xlabel(r"v$+\frac{1}{2}$")
    plt.ylabel(r"$\Delta \tilde{\nu}^{\prime\prime}_{n+1,n}$")
    plt.title(r"Birge-Sponer-Plot für v $^{\prime\prime}$")
    mo.mpl.interactive(plt.gcf()) # Interactive Plot
    return dnu_0, fit_text_0, p_bs_0, pcov_0, popt_0, v_0


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Berechnung der Parameter:

        \begin{equation}
        \tilde{\nu}_e{''} = b{''} - \frac{m{''}}{2}
        \end{equation}

        \begin{equation}
        \chi_e{''} = -\frac{m{''}}{2\tilde{\nu}_e{''}}
        \end{equation}

        \begin{equation}
        k'' = \left(2\pi c \tilde{\nu}_e{''} \right)^2
        \end{equation}

        \begin{equation}
        \nu_{max}{''} = -\frac{b{''}}{m{''}}
        \end{equation}

        \begin{equation}
        D_0{''} = \int_0^{\nu_{max}{''}} m{''}x+b{''}
        \end{equation}

        \begin{equation}
        D_E{''} = D_0{''} + E{''}(0)
        \end{equation}
        """
    )
    return


@app.cell(hide_code=True)
def _(c, mu_I2, np, pd, popt_0):
    # Für die einfachere Nutzung extrahieren wir die Steigung und den y-Achsenabschnitt
    m_2 = popt_0[0]
    b_2 = popt_0[1]

    # Anschließend können wir die Parameter berechen
    nu_e2 = b_2 - m_2/2
    chi_e2 = -m_2/(2*nu_e2)
    k_2 = (2*np.pi*c*nu_e2 * 100)**2  * mu_I2
    v_max2 = - b_2/m_2
    D_02 = 0.5*m_2*v_max2**2 + b_2 *v_max2
    E_02  = nu_e2/2 + nu_e2*chi_e2/4
    D_E2 = D_02 + E_02

    # Definition der Variablen, Einheiten und Startwerte
    data_2 = {
        "nu_e''": ["[cm-1]", nu_e2],
        "chi_e''": ["[–]", chi_e2],
        "k''": ["[N/m]", k_2],
        "nu_{max}''": ["[–]", v_max2],
        "D_0''": ["[cm-1]", D_02],
        "D_E''": ["[cm-1]", D_E2],
    }

    # Transponieren und Umwandeln in DataFrame
    df_2 = pd.DataFrame(data_2, index=["Einheit", "Wert"])

    # Ausgabe anzeigen (Marimo zeigt DataFrames automatisch an)
    df_2
    return D_02, D_E2, E_02, b_2, chi_e2, data_2, df_2, k_2, m_2, nu_e2, v_max2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Berechnung der Potentialverläufe

        Morse-Potential (V(r)): 

        \begin{equation}
        V(r) = D_e \left(\text{exp}\left(-\alpha (r-r_e\right))- 1\right)^2
        \end{equation}

        mit der Steifigkeit des Potentials ($\alpha$): 

        \begin{equation}
        \alpha =  \sqrt{\frac{k}{2D_e}}
        \end{equation}

        Weiterhin muss noch der Abstand der Potentialminima ($\sigma_e$) sowie die Differenz der Gleichgewichtsabstände ($r_e'-r_e''$) berechnet werden. Für die Herleitung in McNaught Paper schauen.

        \begin{equation}
        \sigma_e = \Delta E - E'(\nu') + E''(\nu'')
        \end{equation}

        \begin{equation}
        r_e'-r_e'' = u_0^{max} + \frac{1}{\alpha'} \cdot ln\left(1 \pm \sqrt{\frac{\sigma_{int} - \sigma_e + \frac{\tilde{\nu}_e''}{2} - \frac{\tilde{\nu}_e''\chi_e'''}{4}}{D_e'}} \right)
        \end{equation}

        Mit $u_0^{max}$: 

        \begin{equation}
        u_0^{max} = \frac{1}{\alpha''} \cdot ln\left(\frac{k''}{k''-1}\right)
        \end{equation}
        """
    )
    return


@app.cell(hide_code=True)
def _(
    D_E1,
    D_E2,
    c,
    chi_e1,
    chi_e2,
    deslandre,
    k_1,
    k_2,
    mo,
    np,
    nu_e1,
    nu_e2,
    plt,
):
    # Funktion für das Morse-Potential[kJ/mol]:
    h = 6.6261e-34 
    Na = 6.02214076e23
    alpha1= np.sqrt(k_1/(2*D_E1*100*h*c)) # Einheitenlos
    alpha2= np.sqrt(k_2/(2*D_E2*100*h*c)) # Einheitenlos

    def morse(x,De,alpha):
        De_kJpermol = De*100*h*c*Na/1000
        V_r = De_kJpermol *(np.exp(-alpha*x)-1)**2
        return V_r

    # Wir berechnen ersteinmal die beiden Morse-Potential
    ## Morse-Potential fuer den angeregten elektronischen Zustand
    r_re1 = np.linspace(-0.357e-10, 3e-10, 1000)
    V_r1 = morse(r_re1,D_E1,alpha1)
    ## Morse-Potential fuer den elektronischen Grundzustand
    r_re2 = np.linspace(-0.318e-10, 3e-10, 1000)
    V_r2 = morse(r_re2,D_E2,alpha2)

    # Abstand der Potentialminima
    # Dafür suchen wir einen Übergang aus dem Spektrum aus.
    # Wir können demnach das DeltaE direkt aus dem Spektrum ablesen
    # Nun brauchen wir noch die beteiligten vibr. Niveaus unseres ausgewählten vibronischen Übergangs
    # Ich habe den Übergang von v'' = 0 in v' = 30 bei  534.8 nm ausgewaehlt
    # Für die bessere Lesbarkeit formulieren wir Formeln für die Energie der Zustände
    # Vibr. Energieniveau des angeregten elek. Zustandes
    def E1(v):
        E = nu_e1*(v+0.5) - nu_e1*chi_e1*(v+0.5)**2
        return E

    # Vibr. Energieniveau des elek. Grundzustandes
    def E2(v):
        E = nu_e2*(v+0.5) - nu_e2*chi_e2*(v+0.5)**2
        return E

    # berechnen Den Potentialabstand und rechnen ihn in kJ/mol um
    sigma_e = (deslandre["wn0"][21] - E1(deslandre["vn"][21]) + E2(0)) 
    sigma_e_kJpermol = sigma_e * 100*h*c*Na/1000

    # Für den Abstand der Gleichgewichtsabstände berechnen wir zunächst das Maximum des anharmonisches Oszillators u0max
    u0max = -  1/alpha2 * np.log((k_2-1)/k_2)

    # sigma_int ist die energie des intesivsten Übergang v'' = 0 in v' = 30 bei einer Energie 534.8 nm unter dem Index 21
    sigma_int = deslandre["wn0"][21]
    dre = u0max + 1/alpha1 * np.log(1+np.sqrt((sigma_int - sigma_e + nu_e2/2 - (nu_e2*chi_e2)/4)/D_E1))

    plt.plot((r_re1+dre) * 1e10, V_r1 + sigma_e_kJpermol,label=r"V'$_R$")  
    plt.plot(r_re2 * 1e10, V_r2,label=r"V''$_R$")  
    plt.xlabel(r"$r - r_e$ [$\AA$]")
    plt.ylabel(r"V$_r$ [kJ/mol]")  
    plt.title("Morse-Potential")
    plt.legend()
    mo.mpl.interactive(plt.gcf()) # Interactive Plot
    return (
        E1,
        E2,
        Na,
        V_r1,
        V_r2,
        alpha1,
        alpha2,
        dre,
        h,
        morse,
        r_re1,
        r_re2,
        sigma_e,
        sigma_e_kJpermol,
        sigma_int,
        u0max,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
