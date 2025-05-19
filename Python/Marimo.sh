#!/bin/bash
cd ./Python || { echo "Fehler: Verzeichnis ./Python nicht gefunden."; exit 1; }

# Betriebssystem ermitteln via uname
os_name="$(uname -s)"

case "${os_name}" in
  Darwin*)
    # macOS
    if [ -f C1-Tutorial-env/bin/activate ]; then
      echo "macOS erkannt – aktiviere virtualenv…"
      source C1-Tutorial-env/bin/activate
    else
      echo "Fehler: activate-Skript unter C1-Tutorial-env/bin/activate nicht gefunden."
      exit 1
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT*)
    # Windows (Git Bash, MSYS, Cygwin, WSL)
    if [ -f C1-Tutorial-env/Scripts/activate ]; then
      echo "Windows-Umgebung erkannt – aktiviere virtualenv…"
      source C1-Tutorial-env/Scripts/activate
    else
      echo "Fehler: activate-Skript unter C1-Tutorial-env/Scripts/activate nicht gefunden."
      exit 1
    fi
    ;;
  *)
    echo "Unbekanntes Betriebssystem: ${os_name}"
    exit 1
    ;;
esac

# Optional: hier kannst du weitere Befehle anhängen, z.B.
# marimo edit
marimo edit