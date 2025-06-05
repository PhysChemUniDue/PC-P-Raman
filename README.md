# Chemometrics with Marimo Notebooks

Diese Anleitung hilft euch beim Einstieg und der ersten Auswertung eurer Raman-Messdaten.

## Projektstruktur

Öffnet das Projekt am besten in **VS Code**.  
Im Verzeichnis `./Python` findet ihr die **Marimo-Notebooks** sowie die wichtigsten Skripte.

## Installation

1. Öffnet einen Terminal und wechselt ins Python-Verzeichnis:
   ```bash
   cd ./Python
    ```
2. Installiert Marimo mit folgendem Befehl:
   ```bash
   pip install marimo[recommend]
   ```
3. Startet Marimo mit:
   ```bash
   marimo edit
    ```
## Grundfunktionen

Die Kernfunktionen findet ihr in der Datei **`Chemometrics-v0.1.py`**. .
Öffnet diese Datei, prüft ggf. die Kommentare und installiert bei Bedarf zusätzliche Python-Pakete.
In Zelle 3 steht:
> „Hier den Pfad zur .tsv-Datei einfügen.“

Tragt dort einfach den Pfad zu eurer Messdatei ein, dann sollte das Skript laufen.

## Hinweise zur Nutzung

- Falls etwas nicht funktioniert: Gebt mir Bescheid!
- Wenn ihr mehrere Messungen auswerten wollt, kopiert das Notebook einfach und öffnet es mehrfach. Oder kopiert die relevanten Zellen und ändert die Variablen entsprechend.

## Auswertungsschritte (Chemometrie)

1. Vergleicht das **Silicium-Referenzspektrum** mit der Referenz, berechnet die Abweichung markanter Peaks und prüft, ob eine Neukalibrierung nötig ist.
2. **Kombiniert** die vier Referenzspektren mit den Slidern.
3. Passt gleichzeitig **Steigung** und **Achsenabschnitt** der linearen Baseline an.
   - Tipp: Die Slidergrenzen ggf. anpassen.
   - Die absoluten Werte sind weniger wichtig; entscheidend sind die relativen Prozentwerte in der Plot-Legende.
4. **Bewertet** eure Fits anhand der Residuen:
   - Wo gibt es größere Abweichungen?
   - Sind Referenzspektren ggf. um ein paar Wellenzahlen verschoben? → **Kalibrierung prüfen.**
   - Können thermische Effekte stören?

Wiederholt das Ganze für **alle Positionen** eurer Probe.

## Fragen?

Wenn ihr Fragen zur Auswertung oder zum Code habt, meldet euch gern jederzeit bei mir.  
Viel Erfolg und Spaß beim Auswerten!  
*(Ich hoffe, ich habe nichts vergessen.)*