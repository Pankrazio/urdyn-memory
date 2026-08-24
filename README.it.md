# 🧠 Urdyn

> **I modelli sono sostituibili. La memoria dovrebbe restare portabile.**

Urdyn è un motore di memoria local-first e privato per impostazione predefinita, pensato per progetti, persone e agenti AI. Conserva decisioni, evidenze, tentativi, lezioni e contesto di progetto durevoli in `.urdyn/`, indipendentemente da qualsiasi modello o provider.

## ⚡ Avvio rapido

Installa urdyn-memory da PyPI:

```bash
pip install urdyn-memory
```
Oppure installalo da un checkout locale dei sorgenti:

```bash
cd /percorso/a/urdyn-memory
python -m pip install .
```

Poi inizializza Urdyn all'interno di un progetto:

```bash
cd mio-progetto
urdyn init dev
urdyn status
```

Il motore base richiede Python 3.12+, dichiara zero dipendenze runtime obbligatorie e non necessita di alcun modello o API key.

## Perché Urdyn?

La cronologia del provider appartiene a un provider e a una sessione. Quando la sessione termina, il modello cambia o subentra un altro strumento, la conoscenza operativa del progetto non dovrebbe scomparire con essi.

**Il modello è sostituibile. La memoria è persistente.**

Urdyn non è un archivio di conversazioni. Conserva record espliciti e strutturati, con provenienza e regole per lo stato corrente, poi recupera il materiale rilevante per il task in corso. Una nuova sessione AI può ricostruire il contesto di progetto utile senza dipendere dalla trascrizione di un provider precedente.

## 🧠 Cosa ricorda Urdyn

Urdyn mantiene separati tipi di informazione diversi, invece di appiattire tutto in testo di chat:

| Concetto | Cosa rappresenta |
| --- | --- |
| **Memory** | Note, decisioni, cause radice, lavoro in sospeso, domande, invarianti, fatti d'ambiente e lezioni |
| **Evidence** | Dichiarazioni o conferme dell'utente, output di comandi/test/strumenti, errori osservati, riferimenti a file e osservazioni di documenti |
| **Attempt** | Cosa è stato provato, in che modo e se il risultato è stato positivo, negativo o parziale |
| **Skill** | Una procedura ordinata promossa deliberatamente da una Lesson; non viene mai creata automaticamente |
| **Source** | L'identità di un file di progetto con una storia append-only delle osservazioni |
| **Stato corrente** | La proiezione corrente delle Memory dopo l'esclusione dei record superati; la storia completa resta disponibile |

Questo preserva decisioni, fallimenti, provenienza e lezioni come record distinti, senza fingere che abbiano tutti la stessa autorità.

## 🔒 Local-first e privato per impostazione predefinita

- I dati canonici vivono dentro `.urdyn/` nel tuo workspace.
- Il motore base non richiede account, servizi cloud o API key.
- Il funzionamento base non carica automaticamente dati all'esterno e non scarica modelli.
- `urdyn init` aggiunge automaticamente `.urdyn/` al `.gitignore` del progetto.
- Il seed di un documento conserva localmente in `.urdyn/` il contenuto osservato; non lo invia altrove.

L'extra semantico opzionale è l'eccezione esplicita all'assenza di download: il suo setup scarica da Hugging Face un modello di embedding con versione fissata.

## 🤖 Funziona con gli strumenti AI

Qualunque strumento AI o coding agent con accesso alla shell può usare Urdyn attraverso la CLI pubblica. Gli strumenti senza accesso alla shell possono ricevere il contesto che esporti e fornisci loro manualmente.

```text
AI / strumento
   │
   ▼
CLI pubblica / API Python di Urdyn
   │
   ▼
validazione · provenienza · regole della memoria
   │
   ▼
.urdyn/
```

Questo è un confine di integrazione generico, non un'integrazione automatica con i provider. Urdyn 0.3.0 non include adapter specifici per provider, supporto MCP, curatela autonoma o invocazione automatica. Gli strumenti AI devono usare la CLI/API pubblica e non modificare mai direttamente `.urdyn/`.

## File di progetti esistenti

Il seed esplicito dei file funziona in ogni profilo. In un workspace `dev`, `urdyn seed` senza percorsi elenca i candidati di scoperta e non registra nulla. Indica esplicitamente file di testo UTF-8 regolari per osservarli:

```bash
urdyn seed                           # elenca i candidati; non registra nulla
urdyn seed README.md pyproject.toml  # registra file specifici
```

La scoperta in `dev` è una **camminata ricorsiva delimitata**: file manifest di root (`README*`, `LICENSE*`, `pyproject.toml`, ...) più file simil-documentazione (`.md`, `.txt`) ovunque nell'albero del progetto, a qualunque profondità. Resta delimitata e sicura per la privacy per costruzione, non perché lo si chiede:

- non entra mai in `.git/`, `.venv`/`venv/`, `node_modules/`, `dist/`/`build/`, cache, o nella directory `.urdyn/` di Urdyn stesso;
- rispetta `.gitignore` e `.git/info/exclude` quando presenti — tutto ciò che un progetto ha già detto a Git di dimenticare, Urdyn non lo mostra, indicizza o osserva mai;
- non segue mai un symlink che esce dal workspace, non legge mai un file binario o troppo grande, e applica un filtro sui nomi sensibili specifico della sola scoperta automatica (`*secret*`, `*password*`, `*token*`, ...) in aggiunta al controllo sui nomi di credenziali già presente per il seed esplicito;
- Git è del tutto opzionale: i file di esclusione vengono letti direttamente come testo semplice, mai tramite un sottoprocesso `git`, quindi la scoperta funziona identicamente con o senza un repository.

Ogni file sottoposto a seed diventa una Source con un record Evidence di tipo osservazione del documento. Urdyn conserva testo osservato, digest, dimensione e timestamp, ma non tratta le affermazioni del documento come conoscenza verificata. Elencare un candidato non registra mai nulla da solo — resta un `urdyn seed <percorso>...` esplicito.

## 👀 Watcher di progetto

Il profilo `dev` può mantenere aggiornate in background le osservazioni dei documenti di progetto:

```bash
urdyn watch status
urdyn watch start
urdyn watch stop
```

`urdyn init dev` abilita e avvia il watcher. Osserva ogni Source già tracciata più tutto ciò che la stessa scoperta delimitata e filtrata per la privacy propone al momento — incluso un file appena creato che nessuno ha ancora sottoposto a seed, non solo i file che hanno già una storia tracciata. I file già tracciati sono controllati con una cadenza rapida e adattiva (fino a ogni 2 secondi durante l'attività); accorgersi di un file nuovo mai visto prima usa una cadenza più lenta (circa ogni 10 secondi), perché nulla riguardo un file senza una base di confronto può andare perso trovandolo un po' più tardi. Le modifiche creano record Source/Observation/Evidence, mai Memory automatica o altra conoscenza canonica, e restano locali. `urdyn watch stop` lo ferma e lo disabilita in modo persistente.

Il watcher è validato e supportato su Linux in questa release. Limiti noti della 0.3.0:

- Cancellazioni e rinomini non sono tracciati. La storia esistente viene conservata e un file rinominato inizia una nuova storia Source.
- Non è un servizio di avvio del sistema. Dopo un riavvio, il successivo comando `urdyn` normale riavvia un watcher abilitato e ricontrolla i file già tracciati.
- Un file creato per la prima volta mentre il watcher non è in esecuzione viene scoperto soltanto dopo una sua modifica successiva, non retroattivamente al riavvio.

## Evidence ≠ Knowledge

**Evidence registra ciò che è stato osservato. Memory registra ciò che chi usa Urdyn gli chiede di trattare come conoscenza, con uno stato epistemico esplicito.**

Registrare Evidence non crea mai automaticamente Memory, Lesson o Skill. Un README sottoposto a seed è evidenza fedele di ciò che il file affermava in quel momento; non è una prova che il README sia corretto. Una nuova Memory può essere `user_asserted`, `inferred` o `verified`, e `verified` richiede Evidence di supporto, indicata esplicitamente e di un tipo idoneo. Urdyn applica questo gate strutturale, ma non afferma di comprendere se l'evidenza dimostri davvero la conclusione.

## 🧩 Profili

```bash
urdyn init [general|dev|lab]
```

| Profilo | Comportamento implementato |
| --- | --- |
| **`general`** | Motore base; il seed esplicito funziona, ma discovery senza percorsi e watcher non sono disponibili |
| **`dev`** | Aggiunge la scoperta dei file di progetto senza percorsi e il watcher in background validato su Linux |
| **`lab`** | Identificatore di profilo canonico riservato; attualmente si comporta come `general` |

Tutti i profili condividono archivio canonico, recupero, preflight, context ed export. Oggi il profilo cambia soltanto la scoperta seed senza percorsi e la disponibilità del watcher.

## 📦 API Python

La distribuzione è `urdyn-memory`, il package importabile è `urdyn` e la classe pubblica del workspace è `Urdyn`:

```python
from urdyn import Urdyn

ud = Urdyn.discover()
ud.remember("SQLite è l'archivio canonico del progetto.", kind="decision")

for memory in ud.recall("SQLite è l'archivio canonico del progetto"):
    print(memory.content)
```

L'API Python e la CLI `urdyn` condividono le regole core di validazione e persistenza. Questo README mostra soltanto i punti di ingresso essenziali; per integrare la libreria usa i tipi pubblici esportati da `urdyn`.

## Compilazione ed export del contesto

Prima di iniziare un lavoro, chiedi a Urdyn l'esperienza pregressa rilevante:

```bash
urdyn preflight "racchiudere una migrazione multi-step in un'unica transazione"
urdyn context "racchiudere una migrazione multi-step in un'unica transazione"
urdyn export "racchiudere una migrazione multi-step in un'unica transazione"
```

`context` compila un contesto di lavoro relativo al task e limitato da un budget di caratteri. `export` rende lo stesso tipo di contesto come testo generico portabile, adatto a redirezione o pipe:

```bash
urdyn export "<descrizione del task>" > context.txt
```

Questo export è un contesto limitato al task, non un backup completo né un export dell'intero archivio di memoria.

I documenti di progetto sottoposti a seed possono contribuire al contesto compilato come **Project Evidence** rilevante per il task, tramite recupero lessicale e, quando abilitato, semantico. Per i documenti grandi, Urdyn recupera le porzioni rilevanti per il task invece di richiedere che l'intero documento rientri nel budget del contesto. Il testo recuperato resta Evidence documentale con la provenienza della Source: **Source != Evidence != Durable Memory**, e recuperare Evidence non la promuove mai a Memory.

## Recupero semantico

Il motore base funziona offline con recupero lessicale/full-text. Il recupero semantico è opzionale:

```bash
pip install "urdyn-memory[semantic]"
urdyn semantic setup
```

Il setup scarica un modello di embedding con versione fissata e costruisce un indice locale derivato accanto all'archivio canonico. L'indice è ricostruibile; quando il recupero semantico non è disponibile, i dati canonici restano intatti e Urdyn torna al recupero lessicale.

## 🛠 Ambito attuale e limiti

Urdyn 0.3.0 è una release alpha. Attualmente non include:

- sincronizzazione cloud;
- GUI o applicazione desktop;
- adapter nativi per provider o integrazione MCP;
- curatela autonoma della memoria guidata dall'AI;
- acquisizione automatica delle conversazioni;
- import/export completo dell'archivio di memoria.

Il confine CLI/API è deliberato: Urdyn fornisce il motore di memoria e le sue regole, mentre una persona o uno strumento esterno decide cosa registrare e quando consultarlo.

## Sviluppo

```bash
uv sync --extra semantic
uv run pytest
HF_HUB_OFFLINE=1 uv run pytest -m real_model
uv build
```
La suite completa esercita anche il backend semantico opzionale, quindi l’ambiente di sviluppo installa l’extra semantic. Il pacchetto base continua a non avere dipendenze runtime obbligatorie.

Lo sviluppo richiede Python 3.12+ e [uv](https://docs.astral.sh/uv/).

## Licenza

Apache License 2.0. Vedi [LICENSE](LICENSE).

## 🌍 Lingue

Questo documento è in italiano. Vedi [README.md](README.md) per la versione inglese.
