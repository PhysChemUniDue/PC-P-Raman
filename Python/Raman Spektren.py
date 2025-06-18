import marimo

__generated_with = "0.13.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, os, plt


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
def _(mo, os):
    folder = '../Data/SS25Gr1/Mischung1/'
    # Alle .tsv-Dateien im Ordner sammeln
    tsv_files = [f for f in os.listdir(folder) if f.lower().endswith('.tsv')]
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
    return dropdown_form, folder


@app.cell
def _(dropdown_form, folder, mo, np, os, plt):
    # 1. Solange noch kein Wert da ist, die Cell abbrechen
    mo.stop(dropdown_form.value is None, "Bitte wähle eine Datei und klicke Submit")

    # 2. Datei einlesen (skip 8 header‐Zeilen)
    path = os.path.join(folder, dropdown_form.value)
    data = np.loadtxt(path, delimiter='\t', skiprows=8)
    x, y = data[:,0], data[:,1]

    # 3. Plot erstellen und nur das Figure-Objekt zurückgeben
    fig = plt.plot(x, y) 
    plt.xlabel('Wellenzahl (cm⁻¹)')
    plt.ylabel('Raman Intensität')
    plt.title(f'Raman-Spektrum: {dropdown_form.value}')
    plt.grid(True)
    mo.mpl.interactive(plt.gcf())



    fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
