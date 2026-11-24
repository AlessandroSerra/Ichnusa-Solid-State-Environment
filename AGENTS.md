# Mappa del progetto ISSE

> Scopo: fornire a un agente successivo una vista rapida, mirata ed estendibile della repo, evitando una nuova analisi completa prima di riprendere il lavoro.

## 1. Identità del progetto

- **Nome package**: `isse`
- **Descrizione**: *Ichnusa Solid State Environment*, ambiente Python per workflow di fisica dello stato solido e simulazioni atomistiche.
- **Layout**: package in `src/isse`.
- **Python**: `>=3.11`
- **Build backend**: `uv_build`
- **Dipendenze runtime**: `numpy`, `spglib`, `numba`
- **Unità interne usate dai parser/workflow**:
  - lunghezze/posizioni/celle: Å
  - velocità: Å/fs
  - masse: amu
  - forze: eV/Å

## 2. Stato rapido della repo

- `README.md` è minimale.
- `src/isse/__init__.py`, `src/isse/helpers/__init__.py`, `src/isse/io/__init__.py` espongono API pubbliche aggregate: workflow/costanti dal top-level, parser/writer da `isse.io`, helper stabili da `isse.helpers`.
- `src/isse/convert_file.py` è ora una CLI/converter basata su **ASE** con test di equivalenza integrati; `ase` e `scipy` non sono però nelle dipendenze runtime di `pyproject.toml`.
- `src/isse/convert.py` è un converter nuovo/non tracciato che reimplementa la conversione senza ASE, usando `Atoms`, `Trajectory` e i moduli I/O e writer ISSE per i formati supportati internamente.
- `src/isse/radial_distribution.py` e `src/isse/helpers/periodic.py` aggiungono calcolo RDF e utility PBC/minimum-image.
- Sono presenti file in `tests/`, ma `.gitignore` ignora `tests/`: quindi non sono versionati. `.gitignore` ignora anche `dist/`, `archive/` e `src/isse/TODO.md`.
- Nota working tree dopo questo aggiornamento: `src/isse/constants.py` modificato e `src/isse/spectral.py` non tracciato (`git status --short`).

## 3. Albero logico

```text
src/isse/
├── constants.py              # costanti fisiche, conversioni unità, masse atomiche
├── structures.py             # Atoms e Trajectory lazy anche multi-file
├── phonon_temperatures.py    # entry point alto livello per temperature modali
├── project_velocities.py     # validazione/batching velocità e workflow pubblico
├── project_displacements.py  # proiezione spostamenti dalla reference su coordinate normali
├── convert_file.py           # CLI/converter ASE con test di equivalenza
├── convert.py                # converter ISSE-native senza ASE, non tracciato
├── radial_distribution.py    # RDF totale su Trajectory lazy
├── spectral.py               # VACF e VDOS/spettri vibrazionali
├── helpers/
│   ├── cell_mapping.py       # mapping atomi supercella -> cella primitiva + basis
│   ├── mode_projection.py    # motore condiviso interno, kernel Numba obbligatorio
│   ├── periodic.py           # wrap/unwrap e minimum-image in PBC
│   └── symmetry.py           # scaled positions, primitive cell via spglib, qpoint helpers
└── io/
    ├── parse_alamode.py      # lettura file ALAMODE .evec
    ├── parse_gpumddump.py    # parser lazy GPUMD dump
    ├── parse_lammps.py       # parser LAMMPS data/dump
    ├── parse_vasp.py         # parser POSCAR
    ├── write_gpumddump.py    # writer GPUMD/extended XYZ
    ├── write_lammps.py       # writer LAMMPS data/dump
    └── write_vasp.py         # writer POSCAR
```

## 4. Modello dati centrale

### `src/isse/structures.py`

#### `Atoms`
Dataclass `slots=True` che rappresenta una configurazione atomistica.

Campi principali:
- `symbols: list[str]`
- `cell: np.ndarray`, shape `(3, 3)`, vettori di cella per riga
- `positions: np.ndarray`, shape `(n_atoms, 3)`
- opzionali: `unwrapped_positions`, `velocities`, `masses`, `forces`
- `arrays: dict[str, np.ndarray]` per array per-atomo extra, es. `id`, `type`, `groups`
- `info: dict[str, str]` per metadata, es. `timestep`, header POSCAR

Validazione in `__post_init__` su shape. Proprietà booleane: `has_unwrapped_positions`, `has_velocities`, `has_masses`, `has_forces`.

#### `Trajectory`
Sequenza lazy di `Atoms` basata su offset byte e una funzione `reader(path, offset)`.
Può rappresentare uno o più file concatenati, purché letti dallo stesso reader/reader compatibile.

Caratteristiche:
- `__getitem__(int)` legge un frame singolo.
- `__getitem__(slice)` restituisce una nuova `Trajectory` lazy sugli offset selezionati, anche per traiettorie multi-file.
- `__iter__()` itera leggendo frame on demand.
- `__add__()` concatena lazy due `Trajectory` compatibili; `sum([traj1, traj2, ...])` è supportato.
- `path` è disponibile solo per traiettorie single-file; `paths` restituisce tutti i file backing.
- Memoria proporzionale al numero di frame, non alla dimensione del file.

## 5. Flussi principali

### 5.1 I/O -> dati atomistici

```text
file esterno
  ├─ POSCAR              -> parse_poscar(...)       -> Atoms
  ├─ LAMMPS data         -> parse_lammps_data(...)  -> Atoms
  ├─ LAMMPS dump         -> parse_lammps_dump(...)  -> Trajectory lazy
  └─ GPUMD dump          -> parse_gpumd_dump(...)   -> Trajectory lazy
```

### 5.2 Proiezione velocità su fononi

```text
Trajectory + reference Atoms + ALAMODE .evec
    │
    ├─ read_alamode_evec(evec_filepath)
    │     -> primitive_cell, qpoints, eigenvalues, eigenvectors
    │
    ├─ map_atoms_to_primitive(reference_atoms)
    │     -> cell_indices, basis_indices
    │
    ├─ _precompute_coefficients(...)
    │     -> coefficienti complessi frame-independent
    │
    ├─ _iter_velocity_batches(trajectory, natoms, batch_size)
    │     -> batch velocità, shape (nframes_batch, natoms, 3)
    │
    ├─ _project_batch_numba
    │     -> qdot2, atomic_norms
    │
    └─ _compute_parseval_errors(...)
          -> errori Parseval per frame
```

Funzione pubblica: `project_velocities(...)` in `src/isse/project_velocities.py`.

### 5.3 Temperature modali

```text
calculate_temperature(...)
    └─ project_velocities(...)
         └─ qpoints, qdot2, atomic_norms, parseval_errors
    └─ converte <qdot2> medio in temperature usando
       AMU_A2_FS2_TO_EV / KB_EV_K
```

Output dict attuale:
- `qpoints`, shape `(nqpoints, 3)`
- `mode_temperatures`, shape `(nqpoints, nmodes)`
- `mean_mode_temperature`
- opzionale `selected_mode_temperatures`, shape `(nframes, nselected, nmodes)`

Nota: `reconstructed_temperature` viene calcolata e loggata, ma non inserita in `results`; la docstring è stata aggiornata per riflettere l'output effettivo.

### 5.4 Coordinate normali da spostamenti

```text
Trajectory (positions/unwrapped_positions) + reference Atoms + PhononModes/.evec
    └─ project_displacements(..., position_mode=...)
         ├─ _iter_displacement_batches: u = r - r_reference, con gestione immagini esplicita
         └─ helpers.mode_projection._project_vectors: mapping, coefficienti, kernel Numba e Parseval condivisi
              └─ qpoints, |Q|², sum(m|u|²), errori, opzionale Q complesso
```

### 5.5 Spostamento quadratico medio (MSD)

```text
Trajectory → coordinate continue → rimozione COM opzionale → selezione specie
           → autocorrelazione FFT + somme ai bordi → MSD medio su tutte le origini
```

## 6. Moduli e responsabilità

### `constants.py`

Contiene:
- conversioni: `ANGSTROM_TO_BOHR`, `BOHR_TO_ANGSTROM`, `AMU_A2_FS2_TO_EV`, `HZ_TO_CM`, `KB_EV_K`, `KCAL_MOL_TO_EV`, `PS_TO_FS`
- tabella `ATOMIC_MASSES`
- helper:
  - `mass_from_symbol(symbol)`
  - `masses_from_symbols(symbols)`
  - `symbol_from_mass(mass, tolerance=1e-3)`
  - `symbols_from_masses(masses, tolerance=1e-3)`

### `io/parse_lammps.py`

API:
- `parse_lammps(filename, format="dump"|"data", symbols=None, units=None)` dispatcher
- `parse_lammps_data(...) -> Atoms`
- `parse_lammps_dump(...) -> Trajectory`

Note:
- Supporta `units="metal"` e `units="real"`.
- Data file: supporto dichiarato per `atom_style atomic`, sezioni `Masses` e `Velocities` opzionali.
- Dump: supporta posizioni `x/y/z`, scaled `xs/ys/zs`, unwrapped `xu/yu/zu`, velocità, forze, masse, `id`, `type`.
- Box ortogonali e triclinici convertiti in cella 3x3 con vettori per riga.

### `io/parse_gpumddump.py`

API:
- `parse_gpumd_dump(filename) -> Trajectory`
- reader interno `_read_frame_gpumd_dump(filepath, offset) -> Atoms`

Note:
- Richiede header con `Lattice` e `Properties`.
- Legge proprietà tipo extended XYZ: `species`, `pos`, opzionali `vel`, `force`, `unwrapped_position`, `mass`, `group`, `Time`.

### `io/parse_vasp.py`

API:
- `parse_poscar(filename) -> Atoms`

Note:
- Legge POSCAR con scaling factor positivo.
- Supporta coordinate Direct e Cartesian/Kartesian.
- Calcola masse dai simboli tramite `mass_from_symbol`.
- Ignora/gestisce parzialmente Selective dynamics.

### `io/parse_alamode.py`

API:
- `read_alamode_evec(filename) -> (primitive_cell, qpoints, eigenvalues, eigenvectors)`

Note:
- Legge file ALAMODE `.evec`.
- Converte i vettori di reticolo da Bohr ad Å.
- `eigenvectors` shape `(nq, nmodes, nat_primitive, 3)` complessa.

### `io/write_lammps.py`

API:
- `write_lammps_data(filename, atoms, units="metal") -> None`
- `write_lammps_dump(filename, trajectory, units="metal", fractional=False) -> None` con `trajectory` = `Atoms | Trajectory | list[Atoms]`

Note:
- Scrive LAMMPS data `atom_style atomic` single-frame.
- Scrive LAMMPS dump multi-frame da iterabile di `Atoms`.
- Converte velocità/forze da unità interne verso `metal` o `real`.
- Richiede celle in forma restricted-triclinic.

### `io/write_gpumddump.py`

API:
- `write_gpumd_dump(filename, trajectory) -> None` con `trajectory` = `Atoms | Trajectory | list[Atoms]`

Note:
- Scrive formato GPUMD/extended XYZ multi-frame.
- Include proprietà opzionali se presenti: `vel`, `force`, `unwrapped_position`, `mass`, `group`, `Time`.

### `io/write_vasp.py`

API:
- `write_poscar(filename, atoms, direct=True) -> None`

Note:
- Scrive POSCAR single-frame.
- Raggruppa le righe di posizione per specie in accordo con le righe specie/conteggi.
- Supporta coordinate Direct o Cartesian.

### `helpers/symmetry.py`

API/helper:
- `get_scaled_positions(atoms)` oppure `get_scaled_positions(positions, cell)`
- `find_primitive_cell(atoms, tolerance=1e-5) -> Atoms` usando `spglib`
- `_get_supercell_transform_matrix(supercell, primitive_cell, tolerance=1e-6)` *(privata)*
- `_generate_qpoints(supercell)` *(privata)*

Note:
- `find_primitive_cell` solleva errore se non trova una cella più piccola; quindi una struttura già primitiva viene trattata come caso di errore.

### `helpers/cell_mapping.py`

API:
- `map_atoms_to_primitive(atoms, primitive_cell=None, basis=None, tolerance=1e-3)`

Responsabilità:
- Mappa ogni posizione cartesiana come:
  `position = (cell_index + basis[basis_index]) @ primitive_cell`
- Se `primitive_cell` e `basis` non sono forniti, usa `find_primitive_cell`.
- Restituisce `(cell_indices, basis_indices), residuals`.
- Verifica residui e popolazioni per sito di basis.

### `phonon_modes.py`

API pubblica:
- `PhononModes(primitive, qpoints, frequencies, eigenvectors)` con primitive `isse.structures.Atoms` (nessuna dipendenza ASE o extra), q ridotti `(nq, 3)`, frequenze cm⁻¹ `(nq, nmodes)`, autovettori complessi ortonormali `(nq, nmodes, nprim, 3)`.
- `PhononModes.from_evec(filename, primitive=...)`: ordine primitive esplicito; converte autovalori signed omega² Ry in cm⁻¹.
- `PhononModes.from_anphon(ph, qpoints, primitive=..., source_order_confirmed=True)`: API ALAMODE 2.0; nessun import obbligatorio di anphon.

Contratti:
- primitive nell'ordine, masse e immagini della sorgente, gauge Bloch con sole traslazioni di cella. L'API anphon non espone metadata primitive: la conferma dell'utente è obbligatoria, non si deduce l'ordine dalle shape.
- Validazione shape, finitezza, masse esplicite positive e ortonormalità; nessuna rinormalizzazione silenziosa né inferenza delle masse dai simboli. Copia indipendente della primitive e degli array.
- Documentazione/verifica sorgenti ufficiali: `docs/phonon_projection.md`; regressioni sintetiche: `tests/test_phonon_projection.py`.

### `helpers/mode_projection.py`

Responsabilità (API interna, non riesportata da `helpers/__init__.py`):
- `_project_vectors(..., *, vector_batches, modes=None, return_complex=False)`: mapping, masse, coefficienti, orchestrazione batch e Parseval. L'iteratore di vettori cartesiani è obbligatorio; nessuna lettura implicita di velocità/spostamenti dalla trajectory.
- `_iter_validated_frames(trajectory, reference_atoms)`: validazione metadata comune a entrambi i reader, per ogni frame: sequenza specie, cella finita uguale alla reference (atol=1e-8 Å, rtol=0), masse opzionali finite/positive e coerenti con masse reference/primo frame (atol=rtol=1e-6). Nessun requisito di metadata masse ridondanti. Permutazioni di atomi dello stesso elemento non rilevabili: identità/ordine devono essere preservati dal chiamante.
- `_precompute_coefficients(...)`, `_compute_parseval_errors(squared_moduli, atomic_norms)` e `_project_batch_numba(...)` condivisi dai workflow. Risultati/norme/errori non finiti causano RuntimeError anche con Parseval disabilitato; NaN intenzionale solo per norma atomica zero e norma modale zero.
- Nomi generici (`vectors`, `modal_values`, `squared_moduli`); unità determinate dai vettori forniti. Numba obbligatorio, nessun kernel NumPy o fallback, nessun parametro `backend`.
- Test architetturali in `tests/test_mode_projection.py`, regressioni di validazione/temperature parziali/overflow in `tests/test_projection_validation.py` e regressioni analitiche nei test dei workflow.

### `project_velocities.py`

API principale:
- `project_velocities(trajectory, reference_atoms, evec_filepath=None, batch_size=100, parseval_tolerance=1e-6, *, modes=None, return_complex=False)`
- `evec_filepath` può anche essere un `PhononModes`; default conserva i quattro output storici, `return_complex=True` aggiunge qdot complesso come quinto output (sqrt(amu) Å/fs).
- Kernel Numba obbligatorio a batch tramite `helpers/mode_projection.py`; coefficienti e convenzione di fase invariati. Input modes usa il mapping esplicito source-ordered con validazione specie/masse e supercella; il percorso file-only mantiene spglib per retrocompatibilità.
- Basi parziali/Gamma-only richiedono `parseval_tolerance=None`; nessuna rinormalizzazione. Per basi complete si verificano anche ortogonalità/commensurabilità dei qpoints.
- `calculate_temperature` accetta anch'esso `modes=`; nessun parametro `backend`.

Helper locale:
- `_iter_velocity_batches(trajectory, natoms, batch_size)`; l'orchestrazione è importata da `helpers.mode_projection`.

Contratti importanti:
- Masse da `reference_atoms` o dal primo frame della trajectory; finite e positive. Con `PhononModes` devono coincidere con quelle della primitive.
- Ogni frame della trajectory deve contenere `velocities` shape `(natoms, 3)`.
- `parseval_tolerance: float | None`; `None` disabilita la validazione per proiezioni parziali.

### `project_displacements.py`

API pubblica:
- `project_displacements(trajectory, reference_atoms, evec_filepath=None, batch_size=100, parseval_tolerance=1e-6, *, modes=None, return_complex=False, position_mode="minimum_image")`.
- Stessi input fononici, mapping, coefficienti e kernel Numba di `project_velocities`, importati da `helpers.mode_projection`; nessuna dipendenza fra i due workflow, nessuna nuova dipendenza esterna.
- Output `(qpoints, q2, atomic_norms, parseval_errors)`; `return_complex=True` aggiunge Q complesso `(nframes, nq, nmodes)` in sqrt(amu) Å. q2 e norme in amu Å², non energie.

Contratti/assunzioni:
- Reference di equilibrio fissa e frame nello stesso ordine/specie; celle uguali entro 1e-8 Å. Celle variabili rifiutate, nessuna sottrazione implicita di COM, rotazione o strain.
- `minimum_image`: differenze posizioni rispetto alla reference con MIC nella supercella, non nella primitiva; escursioni < metà box in coordinate frazionarie. Limite MIC su celle fortemente skewed come helper esistente.
- `unwrapped`: richiede `frame.unwrapped_positions`; sottrae reference.unwrapped_positions se presente, altrimenti reference.positions. Origini delle immagini devono essere coerenti, nessuna correzione automatica.
- `cartesian`: differenza diretta di positions, senza MIC.
- Basi parziali richiedono `parseval_tolerance=None`, senza rinormalizzazione. Qdot = dQ/dt solo a reference/basis fissa e immagini continue.
- Test sintetici in `tests/test_displacement_projection.py`; doc in `docs/phonon_projection.md`.

### `phonon_temperatures.py`

API:
- `calculate_temperature(trajectory, reference_atoms, evec_filepath, selected_iqs=None, batch_size=100, parseval_tolerance=1e-6)`

Responsabilità:
- Usa `project_velocities`.
- Calcola temperature modali e temperatura ricostruita.
- Esclude i primi 3 modi a Gamma dalla media termica solo per liste di modi complete per q (assunzione legacy acoustic-first). Con liste parziali `PhononModes`, mantiene tutti i modi forniti e logga un warning se Gamma è presente: nessuna esclusione silenziosa di modi ottici.

Note attuali:
- Calcola `reconstructed_temperature`, ma non la restituisce.
- Restituisce `mean_mode_temperature`; la docstring è allineata all'output effettivo.

### `radial_distribution.py`

API:
- `calculate_rdf(trajectory, r_max, n_bins, batch_size=100, partial=False, species_pairs=None) -> dict[str, NDArray[np.float64] | dict[tuple[str, str], NDArray[np.float64]]]` con `trajectory` = `Trajectory | Atoms`

Responsabilità:
- Calcola la radial distribution function totale `g(r)` per frame su una `Trajectory` lazy o su un singolo `Atoms`.
- Opzionalmente calcola RDF parziali per tutte le coppie di specie o per `species_pairs`.
- Processa posizioni/celle a batch.
- Usa esclusivamente kernel Numba per gli istogrammi (Numba obbligatorio).

Contratti/assunzioni:
- Ogni frame deve avere posizioni shape `(natoms, 3)` e cella `(3, 3)`.
- Tutti i frame devono avere lo stesso numero di atomi e la stessa sequenza di simboli.
- Output base: `r`, `g_r`; con RDF parziali anche `partial_g_r` indicizzato da tuple di specie canoniche, es. `("O", "Si")`.
- Per input `Atoms`, `g_r` e ciascuna RDF parziale hanno shape `(n_bins,)`; per input `Trajectory` hanno shape `(n_frames, n_bins)`.
- Normalizzazione frame-by-frame con il volume della cella del frame.

Dipendenze:
- usa `numba.njit` e `numba.prange` per gli istogrammi, con minimum-image calcolata nei kernel.

### `helpers/periodic.py`

API/helper:
- `minimum_image_displacements(displacements, cell)`
- `minimum_image_distances(displacements, cell)`
- `wrap_positions(positions, cell)`
- `unwrap_positions(positions, cells)`

Responsabilità:
- Utility PBC con celle a vettori per riga (`cartesian = fractional @ cell`).
- Minimum-image, wrap di posizioni cartesiane, unwrap di una traiettoria.

Contratti/assunzioni:
- Celle shape `(3, 3)`; per `unwrap_positions`, celle fisse `(3, 3)` o per-frame `(n_frames, 3, 3)`.
- Minimum-image veloce via coordinate frazionarie arrotondate; può non trovare l'immagine cartesiana più vicina per celle fortemente skewed/non ridotte.

### `convert_file.py`

API/CLI:
- `convert(infile, outfile, infile_type, outfile_type, replicate=None, fractional=False) -> ase.Atoms`
- `main()` entry point script

Responsabilità:
- Converter legacy tramite ASE.
- Supporta input/output custom `alm.lmp`, `lammpstrj`; input custom `alm.xyz`.
- Supporta supercelle (`--replicate`), coordinate frazionarie (`--frac`), stampa POSCAR-style di coordinate scalate (`--print-scaled`).
- Esegue test di equivalenza opzionali: conteggio atomi, composizione, parametri reticolari, Minkowski, spglib, coordinate frazionarie via KDTree/scipy, Kabsch.

Punti aperti:
- Dipende da `ase` e opzionalmente da `scipy`, ma questi pacchetti non sono dichiarati in `pyproject.toml`.
- È script-style dentro il package e non risulta esposto come console script.

### `convert.py`

API:
- `convert(infile, outfile, infile_type, outfile_type, input_units=None, output_units=None, symbols=None, replicate=None, fractional=False, frame=None) -> Atoms | Trajectory`
- `read_file(filename, file_type, units=None, symbols=None) -> Atoms | Trajectory`
- `write_file(filename, atoms_or_trajectory, file_type, units=None, fractional=False) -> None`

Dipendenze interne:
- legge tramite `io/parse_*.py`.
- scrive delegando a `io/write_vasp.py`, `io/write_lammps.py`, `io/write_gpumddump.py`.
- i writer di traiettoria accettano un parametro `trajectory`, supportano anche un singolo `Atoms`, e hanno docstring in stile NumPy.

Responsabilità:
- Conversione ISSE-native senza ASE, usando `Atoms`/`Trajectory` e i moduli I/O interni.
- Formati supportati: `poscar`/`vasp`, `lammps-data`, `lammps-dump`/`lammpstrj`, `gpumd-dump`/`extxyz`.
- Supporta traiettorie lazy, selezione frame, supercelle, coordinate frazionarie dove sensato e unità LAMMPS input/output (`metal`/`real`).
- Include test di equivalenza leggeri e dependency-free: numero frame, numero atomi, composizione, cella, posizioni frazionarie in stesso ordine.

Contratti/assunzioni:
- LAMMPS output usa celle in forma restricted-triclinic compatibile con i parser ISSE.
- Le unità LAMMPS non hanno default: l'utente deve specificare esplicitamente `input_units`/`output_units` o `units` per evitare assunzioni silenziose.
- `lammps-data`, POSCAR/VASP sono output single-frame; per input trajectory multi-frame serve `--frame`.
- Per LAMMPS data, i simboli chimici sono recuperabili in lettura solo passando `symbols`; i test tentano di derivarli dai simboli originali.

### `mean_squared_displacement.py`

- `calculate_msd(trajectory, *, every=1, max_lag=None, species=None, position_mode="auto", remove_com=False)`, disponibile anche da `isse`.
- MSD in Å² ai lag `0, every, 2*every, ...` fino a `max_lag` incluso, usando tutte le origini disponibili. Se `frame.info["timestep"]` è presente in tutti i frame, restituisce `(lag_time, msd)` nelle unità dei metadata (step di simulazione per dump LAMMPS); se assente da tutti restituisce solo `msd`. Metadata incompleti, non finiti, non crescenti o non equispaziati causano errore.
- Usa posizioni unwrapped esplicite o ricostruisce immagini da frame consecutivi in cella fissa, con spostamenti frazionari inferiori a mezzo box; non accetta disponibilità mista in modalità `auto`. La rimozione COM opzionale usa le masse dell'intero sistema prima della selezione specie.
- Legge ogni frame una volta e materializza solo le coordinate selezionate per la FFT. Regressioni: `tests/test_mean_squared_displacement.py`.

### `spectral.py`

API:
- `velocity_autocorrelation(trajectory, atom_groups=None, group=None, max_correlation_len=None, mass_weighted=False, remove_com=True, time_step=None, batch_size=100) -> NDArray | tuple[NDArray, NDArray]`, restituisce `vacf` oppure `(time, vacf)` se `time_step` è fornito
- `vibrational_density_of_states(data, time_step, atom_groups=None, group=None, max_correlation_len=None, mass_weighted=False, remove_com=True, batch_size=100, gaussian_filter_width=None, method="fft") -> tuple[NDArray, NDArray]`, con `data` = `Trajectory | array VACF | tuple(time, vacf)`; restituisce `(frequency, spectrum)`
- `calculate_vdos(corr_values, time_step, gaussian_filter_width=None, method="fft", frequency_step=None, frequency_max=None) -> tuple[NDArray, NDArray]`
- `vibrational_density_of_states` e `calculate_vdos` accettano `frequency_step`/`frequency_max` keyword-only in cm⁻¹, solo per Filon. Senza opzioni la griglia legacy è invariata; solo step usa massimo Nyquist, solo max usa step legacy. Massimi custom oltre Nyquist rifiutati. Valutazione diretta su griglia indipendente dai campioni, senza migliorare la risoluzione fisica; doc `docs/spectral.md`, regressioni `tests/test_spectral.py`.

Responsabilità:
- calcolo della velocity autocorrelation function via FFT da sequenze lazy di frame.
- trasformata coseno veloce via FFT per ottenere VDOS/spettri vibrazionali da una `Trajectory` o da una VACF già calcolata; backend Filon legacy disponibile con `method="filon"`.
- selezione opzionale di gruppi atomici tramite array di appartenenza per-atomo (`atom_groups`) e label `group`, rimozione velocità del centro di massa e pesatura in massa.

Contratti/assunzioni:
- velocità in unità interne Å/fs; `time_step` in fs; asse spettrale restituito sempre in `cm^-1` usando frequenze in ps^-1/THz.
- per la VACF via FFT le velocità selezionate vengono materializzate in array `(n_frames, n_atoms, 3)`.
- `method="fft"` è il default veloce O(N log N); `method="filon"` è legacy O(N²), parallelizzato con `numba.prange` (Numba obbligatorio, nessun fallback Python).

## 7. Punti di attenzione / debito tecnico

Questa sezione va aggiornata man mano che il codice evolve.

1. **TODO esistente** (`src/isse/TODO.md`):
   - fallback masse dal primo frame implementato in `project_velocities` (usato anche da `calculate_temperature`).
   - TODO dice anche che `calculate_temperature` dovrebbe restituire i qpoints, ma il codice attuale li include già in `results["qpoints"]`.
2. **API package parzialmente esposta**: gli `__init__.py` espongono funzioni/workflow principali, parser/writer e helper stabili; `Atoms` e `Trajectory` restano importabili da `isse.structures` per uso avanzato/test.
3. **Assenza test**: nessuna suite test versionata. Priorità alta per parser, unità, shape, lazy loading, Parseval.
4. **`convert_file.py` / dipendenze CLI**: il converter è stato migrato nel file principale, ma usa `ase` e `scipy` non dichiarati. Decidere se:
   - aggiungerli a dipendenze/extra opzionali,
   - spostare la CLI in un extra/tool separato,
   - esporre un console script ufficiale.
5. **`convert.py` non tracciato**: implementato come converter ISSE-native, ma resta da tracciare/versionare e decidere se esporlo come console script ufficiale.
6. **`parse_vasp.py` Selective dynamics**: la gestione è fragile; dopo una riga `Selective dynamics`, legge il tipo coordinate successivo, ma il ramo `elif coords_type.startswith("s")` resta nel codice e può lasciare `positions` non definito in casi anomali.
7. **`find_primitive_cell`**: se l'input è già primitivo solleva errore. Verificare se questo è desiderato per i workflow di mapping.
8. **Temperature API**: `calculate_temperature` calcola e logga `reconstructed_temperature`, ma non la restituisce; valutare se aggiungerla al dizionario risultati.
9. **Logging**:
   - typo in messaggi tipo `Succesfully`.
   - in `phonon_temperatures.py` il messaggio `logger.info` sembra mancare una parentesi `)` nel testo formattato.
10. **Type hints**:
   - `parseval_tolerance` ora annotato `float | None` in proiezione e temperature.
   - alcuni return/import usano `np.ndarray` generico invece di `NDArray[...]`.

## 8. Come estendere questa mappa

Quando aggiungi codice, aggiorna solo le sezioni pertinenti seguendo questo schema:

```markdown
### `path/to/module.py`

API pubblica:
- `funzione_o_classe(...) -> ...`

Responsabilità:
- ...

Contratti/assunzioni:
- shape array, unità, errori attesi, lazy/eager, ecc.

Dipendenze interne:
- usa `...`

Punti aperti:
- ...
```

Per nuovi workflow, aggiungi un diagramma breve in **Flussi principali**.
Per nuovi moduli I/O, specifica sempre:
- formato input/output
- unità lette/scritte
- conversioni a unità interne o da unità interne
- se legge/scrive `Atoms` o `Trajectory`
- colonne/sezioni supportate

## 9. Checklist rapida per un agente che riprende il lavoro

1. Leggere questa mappa.
2. Controllare `git status` per distinguere codice versionato, modificato e untracked.
3. Se si lavora sui workflow fononici, partire da:
   - `src/isse/project_velocities.py`
   - `src/isse/phonon_temperatures.py`
   - `src/isse/helpers/cell_mapping.py`
4. Se si lavora su RDF/PBC, partire da:
   - `src/isse/radial_distribution.py`
   - `src/isse/helpers/periodic.py`
5. Se si lavora su I/O, partire da:
   - `src/isse/io/parse_lammps.py`
   - `src/isse/io/parse_gpumddump.py`
   - `src/isse/io/parse_vasp.py`
   - `src/isse/convert.py`
6. Prima di refactor importanti, introdurre test minimi su `Atoms`, `Trajectory`, parser, shape degli array, RDF/PBC e output di `calculate_temperature`.


## RDF: kernel unificato

- `calculate_rdf` usa un solo `_histogram_rdf_numba` per accumulare totale e parziali nello stesso attraversamento delle coppie; le distanze minimum-image non vengono ricalcolate per le parziali.
- Il totale include anche coppie non selezionate in `species_pairs`. Il kernel restituisce `(counts, partial_counts)`; senza parziali, il secondo array ha shape `(nframes, 0, nbins)`.
- I frame mantengono istogrammi separati per il parallelismo Numba. Normalizzazione e API pubblica restano invariate.
- Test di regressione versionabili in `tests/test_radial_distribution.py`; esecuzione: `python -m unittest discover -s tests -v`.
