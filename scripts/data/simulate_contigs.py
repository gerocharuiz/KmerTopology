"""
simulate_contigs.py

Simulación de contigs de plásmido (secuencias sintéticas de ADN) para usarlas
como modelos nulos / datos de control frente a los contigs reales, integrado
con el pipeline KmerTopology (github.com/gerocharuiz/KmerTopology, fork de
hozumiyu/KmerTopology).

Motivación
----------
La pregunta de fondo no es solo "¿cómo genero ATGC al azar?" sino "¿qué tipo
de aleatoriedad es una comparación justa contra mis contigs reales?". Se
implementan tres niveles, de más débil a más fuerte como modelo nulo:

  0. uniform       -> A,C,G,T equiprobables (i.i.d.). Ignora composición real.
  1. iid_freq      -> i.i.d. según frecuencia empírica (orden 0). Preserva
                       %GC global pero no dependencias locales.
  2. markov        -> cadena de Markov de orden k entrenada sobre tus contigs
                       reales. Preserva estadística local de k-mers de bajo
                       orden; es el null model recomendado si quieres probar
                       que tu vector topológico capta algo MÁS que
                       composición local.
  3. dinuc_shuffle -> shuffle exacto (Altschul-Erikson) que preserva el
                       conteo EXACTO de dinucleótidos de cada contig real,
                       reordenando sus propias letras. Útil como control
                       "por secuencia" en vez de un modelo generativo.

Requiere: numpy, biopython (para leer fastas). La parte de topología requiere
tu paquete KmerTopology ya instalado (gudhi, ripser, etc.).
"""

from __future__ import annotations
import glob
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

BASES = ["A", "C", "G", "T"]


# ---------------------------------------------------------------------------
# 1. Lectura de contigs reales (salida de Platon, por ejemplo)
# ---------------------------------------------------------------------------

def read_fastas(pattern_or_paths) -> List[str]:
    """Lee una o varias rutas/fasta-glob y regresa lista de secuencias (str).

    pattern_or_paths: str con un glob (p.ej. 'platon_out/*/*.plasmid.fasta')
                       o una lista de rutas.
    """
    from Bio import SeqIO

    if isinstance(pattern_or_paths, str):
        paths = glob.glob(pattern_or_paths)
    else:
        paths = list(pattern_or_paths)

    seqs = []
    for p in paths:
        for rec in SeqIO.parse(p, "fasta"):
            seqs.append(str(rec.seq).upper())
    return seqs


# ---------------------------------------------------------------------------
# 2. Modelo 0: uniforme
# ---------------------------------------------------------------------------

def simulate_uniform(length: int = 1000, n: int = 100, seed: int | None = None,
                      verbose: bool = True) -> List[str]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        letters = rng.choice(BASES, size=length)
        out.append("".join(letters))
        if verbose:
            print(f"[uniform] secuencia {i + 1}/{n} generada", flush=True)
    return out


# ---------------------------------------------------------------------------
# 3. Modelo 1: i.i.d. por frecuencia empírica (orden 0)
# ---------------------------------------------------------------------------

def empirical_base_freqs(sequences: Sequence[str]) -> Dict[str, float]:
    """Frecuencia mononucleotídica agregada sobre una colección de contigs."""
    counts = defaultdict(int)
    total = 0
    for s in sequences:
        for ch in s:
            if ch in BASES:
                counts[ch] += 1
                total += 1
    return {b: counts[b] / total for b in BASES}


def simulate_iid_freq(freqs: Dict[str, float], length: int = 1000, n: int = 100,
                       seed: int | None = None, verbose: bool = True) -> List[str]:
    rng = np.random.default_rng(seed)
    p = [freqs[b] for b in BASES]
    out = []
    for i in range(n):
        letters = rng.choice(BASES, size=length, p=p)
        out.append("".join(letters))
        if verbose:
            print(f"[iid_freq] secuencia {i + 1}/{n} generada", flush=True)
    return out


# ---------------------------------------------------------------------------
# 4. Modelo 2: cadena de Markov de orden k (recomendado)
# ---------------------------------------------------------------------------

def fit_markov_chain(sequences: Sequence[str], order: int = 2,
                      pseudocount: float = 1.0, verbose: bool = True) -> Dict[str, Dict[str, float]]:
    """Entrena una cadena de Markov de orden `order` sobre contigs reales.

    Regresa un diccionario {contexto (str de longitud `order`): {base: prob}}.
    Se usa suavizado de Laplace para contextos poco vistos.
    """
    counts: Dict[str, Dict[str, float]] = defaultdict(lambda: {b: pseudocount for b in BASES})

    n_seqs = len(sequences)
    for i, s in enumerate(sequences):
        for j in range(len(s) - order):
            ctx = s[j:j + order]
            nxt = s[j + order]
            if nxt in BASES and all(c in BASES for c in ctx):
                counts[ctx][nxt] += 1
        if verbose:
            print(f"[fit_markov_chain] contig {i + 1}/{n_seqs} procesado", flush=True)

    model = {}
    for ctx, d in counts.items():
        total = sum(d.values())
        model[ctx] = {b: d[b] / total for b in BASES}
    if verbose:
        print(f"[fit_markov_chain] modelo entrenado con {len(model)} contextos distintos", flush=True)
    return model


def simulate_markov(model: Dict[str, Dict[str, float]], order: int,
                     length: int = 1000, n: int = 100,
                     seed: int | None = None, verbose: bool = True) -> List[str]:
    """Genera secuencias sintéticas a partir de la cadena de Markov entrenada."""
    rng = random.Random(seed)
    contexts = list(model.keys())
    out = []
    for i in range(n):
        # Semilla inicial: un contexto observado real, elegido al azar.
        seq = list(rng.choice(contexts))
        while len(seq) < length:
            ctx = "".join(seq[-order:])
            probs = model.get(ctx)
            if probs is None:
                # contexto nunca visto -> caer a distribución uniforme
                probs = {b: 0.25 for b in BASES}
            r, cum = rng.random(), 0.0
            for b in BASES:
                cum += probs[b]
                if r <= cum:
                    seq.append(b)
                    break
            else:
                seq.append(BASES[-1])
        out.append("".join(seq[:length]))
        if verbose:
            print(f"[markov] secuencia {i + 1}/{n} generada", flush=True)
    return out


# ---------------------------------------------------------------------------
# 5. Modelo 3: shuffle exacto de dinucleótidos (Altschul-Erikson)
#    Preserva el conteo EXACTO de dinucleótidos de cada secuencia real.
# ---------------------------------------------------------------------------

def _build_dinuc_graph(seq: str):
    counts = defaultdict(lambda: {b: 0 for b in BASES})
    adj = defaultdict(list)
    for i in range(len(seq) - 1):
        x, y = seq[i], seq[i + 1]
        counts[x][y] += 1
        adj[x].append(y)
    return counts, adj


def _random_last_edges(counts, nodes, rng):
    """Elige, para cada nodo != último, un arco de salida al azar (respetando
    los conteos) que se usará como 'arco final' para garantizar un camino
    euleriano hasta el último caracter."""
    last_edge = {}
    for x in nodes:
        total = sum(counts[x].values())
        if total == 0:
            continue
        r, cum = rng.random() * total, 0.0
        for y in BASES:
            cum += counts[x][y]
            if r <= cum:
                last_edge[x] = y
                counts[x][y] -= 1
                break
    return last_edge


def _connected_to_last(last_edge, nodes, last_ch):
    reach = {x: (x == last_ch) for x in nodes}
    changed = True
    while changed:
        changed = False
        for x in nodes:
            if not reach[x] and x in last_edge and reach.get(last_edge[x], False):
                reach[x] = True
                changed = True
    return all(reach[x] for x in nodes if x != last_ch)


def dinucleotide_shuffle_single(seq: str, rng: random.Random) -> str:
    """Una realización del shuffle de dinucleótidos de Altschul-Erikson."""
    if len(seq) < 3:
        return seq
    nodes = sorted(set(seq))
    last_ch = seq[-1]

    while True:
        counts, adj = _build_dinuc_graph(seq)
        last_edge = _random_last_edges(counts, [x for x in nodes if x != last_ch], rng)
        if _connected_to_last(last_edge, nodes, last_ch):
            break  # existe un camino euleriano válido; procedemos

    # baraja el resto de las listas de adyacencia (quitando el arco final ya elegido)
    for x, y in last_edge.items():
        adj[x].remove(y)
    for x in adj:
        rng.shuffle(adj[x])
    for x, y in last_edge.items():
        adj[x].append(y)

    out = [seq[0]]
    ptr = {x: 0 for x in nodes}
    cur = seq[0]
    for _ in range(len(seq) - 2):
        nxt = adj[cur][ptr[cur]]
        ptr[cur] += 1
        out.append(nxt)
        cur = nxt
    out.append(last_ch)
    return "".join(out)


def dinucleotide_shuffle(sequences: Sequence[str], n_per_seq: int = 1,
                          seed: int | None = None, verbose: bool = True) -> List[str]:
    rng = random.Random(seed)
    out = []
    n_seqs = len(sequences)
    for i, s in enumerate(sequences):
        for _ in range(n_per_seq):
            out.append(dinucleotide_shuffle_single(s, rng))
        if verbose:
            print(f"[dinuc_shuffle] contig {i + 1}/{n_seqs} barajeado "
                  f"({n_per_seq} versión(es))", flush=True)
    return out


# ---------------------------------------------------------------------------
# 6. Conexión con el pipeline de vectores topológicos
# ---------------------------------------------------------------------------

def _topology_worker(args):
    """Función a nivel de módulo (picklable) para usar con ProcessPoolExecutor."""
    seq, kmers_size, step_size, max_step, mode = args
    if mode == "homology":
        from KmerTopology.kmer_topology import KmerHomology
        return KmerHomology(seq, kmers_size, step_size, max_step), None
    from KmerTopology.kmer_topology import KmerTopology
    return KmerTopology(seq, kmers_size, step_size, max_step)


def compute_topology_features(sequences: Sequence[str], kmers_size: int,
                                step_size: int, max_step: int,
                                mode: str = "topology",
                                n_jobs: int = 1,
                                verbose: bool = True,
                                label: str = "") -> Tuple[np.ndarray, np.ndarray | None]:
    """Corre KmerHomology o KmerTopology sobre una lista de secuencias
    (reales o simuladas) y regresa una matriz de features apilada.

    mode:   'homology' -> solo Betti numbers (KmerHomology)
            'topology' -> Betti numbers + eigmin (KmerTopology)
    n_jobs: número de procesos en paralelo. 1 = serial (default, igual que
            antes). Cada secuencia es independiente de las demás, así que
            esto escala casi linealmente con el número de cores del
            servidor. Usa algo como os.cpu_count() - 1 para dejar un core
            libre para el sistema.
    label:  texto opcional para distinguir el origen en los prints, p.ej.
            'reales' o 'simuladas'.
    """
    tasks = [(seq, kmers_size, step_size, max_step, mode) for seq in sequences]
    n_total = len(tasks)
    prefix = f"[topology{f'/{label}' if label else ''}]"

    if n_jobs == 1:
        results = []
        for i, t in enumerate(tasks):
            results.append(_topology_worker(t))
            if verbose:
                print(f"{prefix} vector {i + 1}/{n_total} calculado", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = [None] * n_total
        done = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            future_to_idx = {ex.submit(_topology_worker, t): i for i, t in enumerate(tasks)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                done += 1
                if verbose:
                    print(f"{prefix} vector {done}/{n_total} calculado "
                          f"(secuencia #{idx + 1})", flush=True)

    if mode == "homology":
        return np.vstack([r[0] for r in results]), None

    betti_list = [r[0] for r in results]
    eigmin_list = [r[1] for r in results]
    return np.vstack(betti_list), np.vstack(eigmin_list)


def build_real_vs_simulated_matrix(real_sequences: Sequence[str],
                                    simulated_sequences: Sequence[str],
                                    kmers_size: int, step_size: int, max_step: int,
                                    mode: str = "topology", n_jobs: int = 1,
                                    verbose: bool = True):
    """Calcula vectores topológicos para reales + simulados y regresa
    (X, labels) donde labels es 1 para reales y 0 para simulados -- listo
    para alimentar single_scale_distance / multiscale_distance o un
    clasificador que valide si el método distingue estructura real de null."""
    betti_r, eig_r = compute_topology_features(
        real_sequences, kmers_size, step_size, max_step, mode, n_jobs, verbose, label="reales")
    betti_s, eig_s = compute_topology_features(
        simulated_sequences, kmers_size, step_size, max_step, mode, n_jobs, verbose, label="simuladas")

    X_betti = np.vstack([betti_r, betti_s])
    labels = np.array([1] * len(real_sequences) + [0] * len(simulated_sequences))

    if mode == "topology":
        X_eig = np.vstack([eig_r, eig_s])
        return X_betti, X_eig, labels
    return X_betti, None, labels


# ---------------------------------------------------------------------------
# 7. Ejemplo de uso end-to-end
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) contigs reales de plásmido (salida de Platon en tu pipeline)
    real = read_fastas("data/processed/selected_genomes/plasmid_contigs/*.plasmid.fasta")

    # 2) elige tu modelo nulo -- se recomienda 'markov' con order = kmers_size - 1
    kmers_size = 2  # ajusta al k que uses en tu pipeline
    order = kmers_size - 1

    model = fit_markov_chain(real, order=order)
    simulated = simulate_markov(model, order=order, length=1000, n=len(real), seed=42)

    # alternativas rápidas para comparar:
    # freqs = empirical_base_freqs(real)
    # simulated = simulate_iid_freq(freqs, length=1000, n=len(real), seed=42)
    # simulated = simulate_uniform(length=1000, n=len(real), seed=42)
    # simulated = dinucleotide_shuffle(real, n_per_seq=1, seed=42)

    # 3) conecta con el pipeline de vectores topológicos
    #    n_jobs: en un servidor con varios cores, sube esto (p.ej. os.cpu_count()-1)
    #    en vez de dejarlo en 1 (serial) -- cada secuencia es independiente.
    step_size, max_step = 10, 50  # ajusta a los valores que ya usas
    import os
    n_jobs = max(os.cpu_count() - 1, 1)
    #X_betti, X_eig, labels = build_real_vs_simulated_matrix(
    #    real, simulated, kmers_size, step_size, max_step, mode="topology", n_jobs=n_jobs
    #)
    from pathlib import Path
    from KmerTopology.matrix import create_matrix_vectors
    import pandas as pd
    # Ruta y nombres con los que se guadaran las matrices
    ruta_matB = Path("data/simulated/mat_B")
    ruta_matE = Path("data/simulated/mat_E")

    #Vectores topologicos de cada genoma
    mat_B, mat_e = create_matrix_vectors(
        secuences = simulated, 
        kmers_size = kmers_size, 
        step_size = step_size, 
        max_step = max_step
    )

    # Guardamos
    df_B = pd.DataFrame(mat_B)
    df_e = pd.DataFrame(mat_e)
    # Creamos carpetas en caso de ser necesario
    ruta_matB.parent.mkdir(parents=True, exist_ok=True)
    ruta_matE.parent.mkdir(parents=True, exist_ok=True)
    
    df_B.to_csv(ruta_matB, index=False)
    df_e.to_csv(ruta_matE, index=False)

    print(f"{X_betti.shape[0]} secuencias ({labels.sum()} reales, "
          f"{(labels == 0).sum()} simuladas) -> features Betti: {X_betti.shape}")

    # 4) con X_betti/X_eig + labels puedes:
    #    - correr KmerTopology.distance.single_scale_distance(X_betti) y ver si
    #      reales/simuladas forman clusters separados (validación del método)
    #    - entrenar un clasificador simple reales-vs-simuladas: si el accuracy
    #      es alto, tu vector topológico SÍ captura algo más allá del null model
