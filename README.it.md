# Cortex Memory Engine

Un motore di memoria locale, persistente, strutturato e indipendente dal modello, pensato per persone, sistemi AI e agenti.

I modelli sono sostituibili. La memoria dovrebbe restare portabile.

## Perché Cortex

Le sessioni AI, i modelli e gli strumenti cambiano di continuo — parte una nuova sessione, un modello viene sostituito, un agente diverso riprende il lavoro. Ciò che non dovrebbe andare perso ogni volta è la conoscenza reale accumulata sul progetto: le decisioni prese e il perché, le cause radice individuate, le lezioni verificate contro evidenze reali, e la traccia di cosa è stato provato e non ha funzionato. Cortex è un archivio piccolo e indipendente per questa conoscenza, così da sopravvivere a qualsiasi singolo strumento o modello.

## Principi fondamentali

- **Local-first** — la memoria vive in una directory di workspace sul disco, sotto il tuo controllo.
- **Privata di default** — nulla lascia la macchina durante il funzionamento normale; non serve un account e non serve il cloud.
- **Indipendente dal modello** — l'archivio canonico, il recupero delle informazioni e la compilazione del contesto funzionano tutti senza alcun modello AI. Il recupero semantico è un'aggiunta opzionale, non un requisito.
- **L'evidenza non è verità canonica** — l'Evidence registra ciò che è stato osservato (l'output di un comando, la conferma di un utente, il contenuto di un file); la Memory registra ciò che si ritiene vero. Una memory diventa `verified` solo se cita evidenza di supporto di un tipo sufficientemente solido da giustificarlo (un gate deliberato, non una formalità).
- **La storia canonica è preservata** — le memorie possono essere superate o invalidate, ma la storia di cosa è stato registrato, e quando, non viene mai riscritta silenziosamente.
- **Nessun LLM richiesto** — il motore di base (registrazione, ricerca, preflight, compilazione del contesto, export) funziona con puro recupero lessicale/full-text, senza alcun download di modelli.

## Installazione

```bash
pip install cortex-memory
```

Il recupero semantico è un extra opzionale (vedi [Recupero semantico](#recupero-semantico)):

```bash
pip install "cortex-memory[semantic]"
```

Richiede Python 3.12+.

## Avvio rapido

```bash
mkdir mio-progetto && cd mio-progetto
cortex init dev
```

Registra qualcosa di rilevante, sostenuto da evidenza reale:

```bash
cortex evidence add "La migrazione 042 è fallita a metà su staging, lasciando lo schema parzialmente aggiornato." --kind error_observation

cortex evidence add "Rieseguendo la migrazione in un'unica transazione su staging: confermata l'assenza di stato di schema parziale dopo un fallimento forzato." --kind user_confirmation

cortex learn "Racchiudere sempre le migrazioni di schema multi-step in un'unica transazione, così un fallimento lascia lo schema invariato." \
  --supporting-evidence <id-evidence-dello-step-user_confirmation> --verified
```

Prima di iniziare un lavoro correlato in seguito, verifica cosa Cortex già sa:

```bash
cortex preflight "racchiudere una migrazione di schema multi-step in un'unica transazione"
```

Compila un contesto di lavoro con budget per lo stesso task:

```bash
cortex context "racchiudere una migrazione di schema multi-step in un'unica transazione"
```

Esporta lo stesso contesto in forma portabile:

```bash
cortex export "racchiudere una migrazione di schema multi-step in un'unica transazione"
```

Esegui `cortex --help` per l'elenco completo dei comandi (`remember`, `recall`, `timeline`, `attempt`, `skills`, `guard` e altri).

## Usare Cortex con il tuo assistente AI

Non è necessario memorizzare tutti i comandi di `cortex`. Qualunque coding agent o strumento AI che disponga di accesso shell al workspace del progetto può usare direttamente la CLI `cortex` — questo è un percorso di integrazione CLI generico, non un adapter specifico per un provider. Funziona allo stesso modo con qualsiasi agente in grado di eseguire comandi shell, perché è semplicemente la stessa CLI pubblica che digiteresti tu.

L'agente dovrebbe attenersi esclusivamente alla superficie pubblica CLI/API e non modificare mai direttamente `.cortex/` — `cortex --help` e l'help di ogni subcommand sono sufficienti per scegliere il primitive giusto (`remember`, `learn`, `evidence add`, `preflight`, `context`, `export` e così via).

Una semplice istruzione al tuo assistente AI è sufficiente per stabilire questo comportamento, ad esempio:

> Use Cortex as the persistent memory for this project. Use the `cortex` CLI and never edit `.cortex/` directly. Before significant work, consult the relevant Cortex context. During work, record meaningful evidence, attempts, and durable project knowledge when appropriate. After verified outcomes, preserve reusable lessons. Use `cortex --help` when needed.

Se il modello con cui lavori non ha accesso alla shell, puoi invece compilare tu stesso il contesto e passarlo direttamente:

```bash
cortex export "<descrizione del task>"
```

e fornire al modello il contesto compilato e portabile risultante come parte del tuo prompt.

Questo è un confine di integrazione CLI generico, non un'integrazione nativa: Cortex non include adapter specifici per Claude/Codex/ChatGPT, supporto MCP, curatela autonoma della memoria o invocazione automatica, e usare un assistente AI per guidarlo non implica che la memoria finisca organizzata meglio di quanto farebbe una persona digitando gli stessi comandi. Il confine è sempre:

```
AI / strumento -> CLI/API pubblica di Cortex -> validazione/policy di Cortex -> .cortex/
```

Il modello interagisce solo attraverso la CLI/API pubblica; non manipola mai direttamente l'archivio o i file interni di `.cortex/`.

## Progetti esistenti

`cortex seed` (disponibile nel profilo `dev`) permette a Cortex di prendere conoscenza dei file già presenti nel progetto:

```bash
cortex seed                    # senza percorsi: elenca i candidati, non registra nulla
cortex seed README.md src/     # registra percorsi specifici
```

I file sottoposti a seed diventano osservazioni **Source / Evidence** — un record di cosa conteneva un file e quando è stato osservato. Non diventano silenziosamente verità canonica: il seed di un file aggiunge provenienza che Cortex potrà citare in seguito, non crea di per sé una memory verificata.

## Watcher di progetto (profilo dev)

`cortex init dev` abilita anche un processo locale in background che mantiene automaticamente aggiornati i documenti di progetto tracciati, così non serve ricordarsi di rilanciare `cortex seed` dopo ogni modifica:

```bash
cortex watch status   # stato, pid, ultima osservazione, source tracciate mancanti su disco
cortex watch start    # abilita + avvia (è anche ciò che fa "init dev")
cortex watch stop     # ferma il processo e lo disabilita in modo persistente
```

Osserva soltanto percorsi già tracciati come Source, più la stessa allowlist di scoperta usata da `cortex seed` — mai una scansione dell'intero progetto. Non crea mai una Memory né altra conoscenza canonica: produce le stesse osservazioni Source/Evidence di `cortex seed`, e nulla lascia questa macchina. Tre limiti noti della V1: le cancellazioni e i rinomini di file non sono tracciati (la storia di un file cancellato resta, e un file rinominato ne inizia una nuova); il watcher non si riavvia da solo dopo un riavvio del sistema — il comando `cortex` successivo in quel workspace lo riavvia e ricontrolla ogni file già tracciato per le modifiche perse; e un file creato mentre il watcher non era in esecuzione viene recuperato solo alla sua modifica successiva, non retroattivamente al riavvio. Validato su Linux; su altre piattaforme `cortex watch status` lo segnala come non disponibile invece di dichiarare un supporto mai testato lì.

## Compilazione del contesto

```bash
cortex context "<descrizione del task>"
```

Data una descrizione di task, Cortex recupera le memorie, le lezioni e le evidenze rilevanti e le compila in un unico contesto di lavoro entro un budget di caratteri (`--budget`, default 4000), dando priorità al materiale più rilevante e canonico.

## Export generico portabile

```bash
cortex export "<descrizione del task>"
cortex export "<descrizione del task>" > context.txt
cortex export "<descrizione del task>" | altro-strumento
```

`export` compila lo stesso tipo di contesto di lavoro task-aware di `context`, formattato come blocco di testo portabile e generico (`--for generic`, l'unico target di export attuale), pensato per essere reindirizzato o passato in pipe verso un altro strumento o prompt. È un contesto compilato e limitato al task — non un export completo dell'archivio di memoria.

## Recupero semantico

Il recupero semantico (basato su embedding) è **opzionale**. Il motore di base funziona interamente offline con ricerca lessicale/full-text e non richiede alcun download di modelli.

Per abilitarlo su un workspace:

```bash
pip install "cortex-memory[semantic]"
cortex semantic setup
```

Questo scarica e fissa una specifica versione di un modello sentence-transformers al primo utilizzo e costruisce un indice semantico locale accanto all'archivio di memoria. L'indice è derivato, ricostruibile e può essere eliminato in sicurezza — Cortex torna al recupero solo lessicale se non è presente.

## Privacy

- La memoria è archiviata localmente, in una directory di workspace (`.cortex/`) sulla tua macchina.
- Non è richiesto alcun account o registrazione.
- Non è richiesto alcun servizio cloud per il funzionamento di base (registrazione, ricerca, preflight, compilazione del contesto, export).
- Non è richiesto alcun LLM o modello AI per il funzionamento di base.
- `cortex init` aggiunge automaticamente `.cortex/` al `.gitignore` del workspace, così l'archivio di memoria non viene per default incluso nel repository del progetto.

Abilitare l'extra semantico opzionale scarica un modello da Hugging Face al primo setup; il funzionamento di base no.

## Profili

```bash
cortex init [general|dev|lab]
```

- **`dev`** — il profilo con il comportamento più concreto oggi: abilita la scoperta automatica dei file di progetto con `cortex seed` e avvia il [watcher di progetto in background](#watcher-di-progetto-profilo-dev). È anche il profilo più esercitato dalla suite di test.
- **`general`** — il profilo predefinito per un uso non di sviluppo di Cortex; si comporta come il motore di base, senza scoperta automatica dei file di progetto.
- **`lab`** — un identificatore di profilo canonico riservato a un uso sperimentale/esplorativo; oggi si comporta come `general`.

Tutti e tre i profili condividono lo stesso archivio canonico e lo stesso comportamento di recupero, preflight, context ed export. Al momento il profilo influisce su due cose: la possibilità che `cortex seed` (senza percorsi) scopra automaticamente i file di progetto, e l'avvio del watcher di progetto in background.

## Ambito attuale / limiti

Cortex v1 non include:

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
