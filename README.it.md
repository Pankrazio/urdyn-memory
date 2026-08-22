# Urdyn Memory Engine

Un motore di memoria locale, persistente, strutturato e indipendente dal modello, pensato per persone, sistemi AI e agenti.

I modelli sono sostituibili. La memoria dovrebbe restare portabile.

## Perché Urdyn

Le sessioni AI, i modelli e gli strumenti cambiano di continuo — parte una nuova sessione, un modello viene sostituito, un agente diverso riprende il lavoro. Ciò che non dovrebbe andare perso ogni volta è la conoscenza reale accumulata sul progetto: le decisioni prese e il perché, le cause radice individuate, le lezioni verificate contro evidenze reali, e la traccia di cosa è stato provato e non ha funzionato. Urdyn è un archivio piccolo e indipendente per questa conoscenza, così da sopravvivere a qualsiasi singolo strumento o modello.

## Principi fondamentali

- **Local-first** — la memoria vive in una directory di workspace sul disco, sotto il tuo controllo.
- **Privata di default** — nulla lascia la macchina durante il funzionamento normale; non serve un account e non serve il cloud.
- **Indipendente dal modello** — l'archivio canonico, il recupero delle informazioni e la compilazione del contesto funzionano tutti senza alcun modello AI. Il recupero semantico è un'aggiunta opzionale, non un requisito.
- **L'evidenza non è verità canonica** — l'Evidence registra ciò che è stato osservato (l'output di un comando, la conferma di un utente, il contenuto di un file); la Memory registra ciò che si ritiene vero. Una memory diventa `verified` solo se cita evidenza di supporto di un tipo sufficientemente solido da giustificarlo (un gate deliberato, non una formalità).
- **La storia canonica è preservata** — le memorie possono essere superate o invalidate, ma la storia di cosa è stato registrato, e quando, non viene mai riscritta silenziosamente.
- **Nessun LLM richiesto** — il motore di base (registrazione, ricerca, preflight, compilazione del contesto, export) funziona con puro recupero lessicale/full-text, senza alcun download di modelli.

## Installazione

Da PyPI, quando il pacchetto sarà pubblicato:

```bash
pip install urdyn-memory
```

Da un checkout locale dei sorgenti già disponibile:

```bash
cd /percorso/a/urdyn-memory
python -m pip install .
```

Il recupero semantico è un extra opzionale (vedi [Recupero semantico](#recupero-semantico)):

```bash
pip install "urdyn-memory[semantic]"
```

Per un checkout locale dei sorgenti, usa invece `python -m pip install ".[semantic]"`.

Richiede Python 3.12+.

## Avvio rapido

```bash
mkdir mio-progetto && cd mio-progetto
urdyn init dev
```

Registra qualcosa di rilevante, sostenuto da evidenza reale:

```bash
urdyn evidence add "La migrazione 042 è fallita a metà su staging, lasciando lo schema parzialmente aggiornato." --kind error_observation

urdyn evidence add "Rieseguendo la migrazione in un'unica transazione su staging: confermata l'assenza di stato di schema parziale dopo un fallimento forzato." --kind user_confirmation

urdyn learn "Racchiudere sempre le migrazioni di schema multi-step in un'unica transazione, così un fallimento lascia lo schema invariato." \
  --supporting-evidence <id-evidence-dello-step-user_confirmation> --verified
```

Prima di iniziare un lavoro correlato in seguito, verifica cosa Urdyn già sa:

```bash
urdyn preflight "racchiudere una migrazione di schema multi-step in un'unica transazione"
```

Compila un contesto di lavoro con budget per lo stesso task:

```bash
urdyn context "racchiudere una migrazione di schema multi-step in un'unica transazione"
```

Esporta lo stesso contesto in forma portabile:

```bash
urdyn export "racchiudere una migrazione di schema multi-step in un'unica transazione"
```

Esegui `urdyn --help` per l'elenco completo dei comandi (`remember`, `recall`, `timeline`, `attempt`, `skills`, `guard` e altri).

## Usare Urdyn con il tuo assistente AI

Non è necessario memorizzare tutti i comandi di `urdyn`. Qualunque coding agent o strumento AI che disponga di accesso shell al workspace del progetto può usare direttamente la CLI `urdyn` — questo è un percorso di integrazione CLI generico, non un adapter specifico per un provider. Funziona allo stesso modo con qualsiasi agente in grado di eseguire comandi shell, perché è semplicemente la stessa CLI pubblica che digiteresti tu.

L'agente dovrebbe attenersi esclusivamente alla superficie pubblica CLI/API e non modificare mai direttamente `.urdyn/` — `urdyn --help` e l'help di ogni subcommand sono sufficienti per scegliere il primitive giusto (`remember`, `learn`, `evidence add`, `preflight`, `context`, `export` e così via).

Una semplice istruzione al tuo assistente AI è sufficiente per stabilire questo comportamento, ad esempio:

> Use Urdyn as the persistent memory for this project. Use the `urdyn` CLI and never edit `.urdyn/` directly. Before significant work, consult the relevant Urdyn context. During work, record meaningful evidence, attempts, and durable project knowledge when appropriate. After verified outcomes, preserve reusable lessons. Use `urdyn --help` when needed.

Se il modello con cui lavori non ha accesso alla shell, puoi invece compilare tu stesso il contesto e passarlo direttamente:

```bash
urdyn export "<descrizione del task>"
```

e fornire al modello il contesto compilato e portabile risultante come parte del tuo prompt.

Questo è un confine di integrazione CLI generico, non un'integrazione nativa: Urdyn non include adapter specifici per Claude/Codex/ChatGPT, supporto MCP, curatela autonoma della memoria o invocazione automatica, e usare un assistente AI per guidarlo non implica che la memoria finisca organizzata meglio di quanto farebbe una persona digitando gli stessi comandi. Il confine è sempre:

```
AI / strumento -> CLI/API pubblica di Urdyn -> validazione/policy di Urdyn -> .urdyn/
```

Il modello interagisce solo attraverso la CLI/API pubblica; non manipola mai direttamente l'archivio o i file interni di `.urdyn/`.

## Progetti esistenti

`urdyn seed` (disponibile nel profilo `dev`) permette a Urdyn di prendere conoscenza dei file già presenti nel progetto:

```bash
urdyn seed                    # senza percorsi: elenca i candidati, non registra nulla
urdyn seed README.md pyproject.toml  # registra file specifici
```

I file sottoposti a seed diventano osservazioni **Source / Evidence** — un record di cosa conteneva un file e quando è stato osservato. Non diventano silenziosamente verità canonica: il seed di un file aggiunge provenienza che Urdyn potrà citare in seguito, non crea di per sé una memory verificata.

## Watcher di progetto (profilo dev)

`urdyn init dev` abilita anche un processo locale in background che mantiene automaticamente aggiornati i documenti di progetto tracciati, così non serve ricordarsi di rilanciare `urdyn seed` dopo ogni modifica:

```bash
urdyn watch status   # stato, pid, ultima osservazione, source tracciate mancanti su disco
urdyn watch start    # abilita + avvia (è anche ciò che fa "init dev")
urdyn watch stop     # ferma il processo e lo disabilita in modo persistente
```

Osserva soltanto percorsi già tracciati come Source, più la stessa allowlist di scoperta usata da `urdyn seed` — mai una scansione dell'intero progetto. Non crea mai una Memory né altra conoscenza canonica: produce le stesse osservazioni Source/Evidence di `urdyn seed`, e nulla lascia questa macchina. Tre limiti noti della V1: le cancellazioni e i rinomini di file non sono tracciati (la storia di un file cancellato resta, e un file rinominato ne inizia una nuova); il watcher non si riavvia da solo dopo un riavvio del sistema — il comando `urdyn` successivo in quel workspace lo riavvia e ricontrolla ogni file già tracciato per le modifiche perse; e un file creato mentre il watcher non era in esecuzione viene recuperato solo alla sua modifica successiva, non retroattivamente al riavvio. Validato su Linux; su altre piattaforme `urdyn watch status` lo segnala come non disponibile invece di dichiarare un supporto mai testato lì.

## Compilazione del contesto

```bash
urdyn context "<descrizione del task>"
```

Data una descrizione di task, Urdyn recupera le memorie, le lezioni e le evidenze rilevanti e le compila in un unico contesto di lavoro entro un budget di caratteri (`--budget`, default 4000), dando priorità al materiale più rilevante e canonico.

## Export generico portabile

```bash
urdyn export "<descrizione del task>"
urdyn export "<descrizione del task>" > context.txt
urdyn export "<descrizione del task>" | altro-strumento
```

`export` compila lo stesso tipo di contesto di lavoro task-aware di `context`, formattato come blocco di testo portabile e generico (`--for generic`, l'unico target di export attuale), pensato per essere reindirizzato o passato in pipe verso un altro strumento o prompt. È un contesto compilato e limitato al task — non un export completo dell'archivio di memoria.

## Recupero semantico

Il recupero semantico (basato su embedding) è **opzionale**. Il motore di base funziona interamente offline con ricerca lessicale/full-text e non richiede alcun download di modelli.

Per abilitarlo su un workspace:

```bash
pip install "urdyn-memory[semantic]"
urdyn semantic setup
```

Questo scarica e fissa una specifica versione di un modello sentence-transformers al primo utilizzo e costruisce un indice semantico locale accanto all'archivio di memoria. L'indice è derivato, ricostruibile e può essere eliminato in sicurezza — Urdyn torna al recupero solo lessicale se non è presente.

## Privacy

- La memoria è archiviata localmente, in una directory di workspace (`.urdyn/`) sulla tua macchina.
- Non è richiesto alcun account o registrazione.
- Non è richiesto alcun servizio cloud per il funzionamento di base (registrazione, ricerca, preflight, compilazione del contesto, export).
- Non è richiesto alcun LLM o modello AI per il funzionamento di base.
- `urdyn init` aggiunge automaticamente `.urdyn/` al `.gitignore` del workspace, così l'archivio di memoria non viene per default incluso nel repository del progetto.

Abilitare l'extra semantico opzionale scarica un modello da Hugging Face al primo setup; il funzionamento di base no.

## Profili

```bash
urdyn init [general|dev|lab]
```

- **`dev`** — il profilo con il comportamento più concreto oggi: abilita la scoperta automatica dei file di progetto con `urdyn seed` e avvia il [watcher di progetto in background](#watcher-di-progetto-profilo-dev). È anche il profilo più esercitato dalla suite di test.
- **`general`** — il profilo predefinito per un uso non di sviluppo di Urdyn; si comporta come il motore di base, senza scoperta automatica dei file di progetto.
- **`lab`** — un identificatore di profilo canonico riservato a un uso sperimentale/esplorativo; oggi si comporta come `general`.

Tutti e tre i profili condividono lo stesso archivio canonico e lo stesso comportamento di recupero, preflight, context ed export. Al momento il profilo influisce su due cose: la possibilità che `urdyn seed` (senza percorsi) scopra automaticamente i file di progetto, e l'avvio del watcher di progetto in background.

## Ambito attuale / limiti

Urdyn v1 non include:

- Integrazione MCP
- Sincronizzazione cloud
- Una GUI o un'app desktop
- Adapter integrati per provider AI
- Curatela autonoma della memoria guidata da AI
- Import/export completo dell'intero archivio di memoria (solo l'`export` task-scoped descritto sopra)

## Sviluppo

```bash
uv sync
uv run pytest                          # suite di test completa
uv run pytest -m real_model            # test che richiedono il modello semantico già in cache (altrimenti skippati)
uv build                               # build di wheel + sdist
```

Richiede Python 3.12+ e [uv](https://docs.astral.sh/uv/).

## Licenza

Apache License 2.0. Vedi [LICENSE](LICENSE).

## Documentazione in inglese

Vedi [README.md](README.md) per la versione inglese di questo documento.
