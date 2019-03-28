# rz-hub

# Die Idee - Ein Schachbrett-StepSequencer

Beispiel für einen StepSequencer: https://youtu.be/GVilj3bChHY?t=41
Die Software ist zum Aufbau einer musikalischen Installation: Auf einem Schachbrett werden farbkodierte Figuren (Rot, Grün, Blau) gelegt und damit werden "Beats" gebaut - da ein Schachbrett 8*8 Felder hat, eignet es sich hierfür ideal, da die Rythmen im 4/4 Takt eingespielt werden und somit jedes Feld für eine 8tel oder eine 16tel stehen kann. 

## Aufbau

Ein Computer, der mit einer DAW (Digital Audio Workstation, wie Ableton Live, Cubase, Logic etc.) verbunden ist, bekommt von dieser eine Clock (ist also zeitlich synchronisiert). Durch die Clock wird ein sog. Step-Sequencer angesteuert, der einen Takt (in sechzehntel aufgeteilt) immer wiederholt (das entspricht einer Sequenz) mit insgesamt 12 Voices (es können also maximal 12 Samples angesteuert werden). 
Ein zweiter Computer ist mit einer Webcam verbunden, die auf ein Schachbrett gerichtet ist. Mittels cv2 werden zuerst die Felder des Bretts erkannt und immer, wenn die Webcam geupdated wird (aktuelle über ein Midi-Signal), wird ermittelt, auf welchen Feldern sich Objekte befinden und welche Farbe sie haben (möglich: Rot, Grün, Blau). Die erfassten Felder werden dann als Sequenz gespeichert, per Websocket an den Step-Sequencer geschickt und ersetzen die vorherige Sequenz. Auf diese Weise kann die Sequenz ohne Verzögerung weitergespielt werden - das ist deshalb besonders wichtig, da auch kleine Verzögerungen hier sehr schnell unangenehm auffallen.


## Installation

Alle Abhängigkeiten können mit der setup.py installiert werden.
```
python setup.py
```

## Aufbau

Ableton <-> Client  <-> Server <- Chesscam
