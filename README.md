## Descrizione
Questo progetto è un framework realizzato ed impiegato per simulare varie tipologie d'errore nelle misure di Spettroscopia Elettrochimica d'Impedenza, degradando la qualità delle acquisizioni in svariati modi, a partire da un dataset open [1]. Le metodologie di errore sono state ideate in ottica di implementazione di misura/stima dello State of Health a partire dalle misure di spettroscopia di impedenza a bordo di vetture elettriche (EVs).
## Struttura del progetto

- `degrad_cable_movement/` → simulazione movimento cavi, perturbazione della fixture/geometria del setup di misura
- `degrad_drift/` → deriva strumentale alle basse frequenze
- `degrad_linKK/` → degradazione basata su fit LinKK
- `degrad_orazem/` → modello di errore stocastico di Orazem et al.
- `degrad_snr/` → degradazione del rapporto segnale/rumore sul sensing di tensione e corrente 
- `degrad_sys_err/` → errori sistematici dovuti a calibrazione / introdotti dal sistema di misura
- `VAE/` → Variational Autoencoder allenato su feature di impedenza + Temperatura + State of Charge. I pesi sono dell'encoder sono impiegati (congelati) per effettuare fine-tuning / transfer-learning su un head di regressione che opera sulle feature latenti.

## Installazione
Il progetto è stato creato in modo tale da avere un virtual environment locale 'venv' all'interno della cartella radice del progetto.

Per l'installazione, effettuare il download dell'archivio ed estrarlo in una cartella.
Nella cartella, creare il virtual environment: 

Windows:

Linux/MacOs:

Attivare il virtual environment:



## Utilizzo

## Bibliografia 
[1] M. Rashid et al., "Dataset for rapid state of health estimation of lithium batteries using EIS and machine learning: Training and validation," Data in Brief, vol. 48, p. 109157, 2023. doi:10.1016/j.dib.2023.109157
