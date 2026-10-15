"""
kmer_frequency.py

Baseline para comparar contra KmerTopology: el vector de frecuencias de
k-mers de una secuencia. Para k-mers de longitud k el vector tiene
dimension 4^k (una entrada por cada k-mer posible sobre {A, C, G, T}),
en orden lexicografico: AAAA, AAAC, AAAG, AAAT, AACA, ...

Va en la misma carpeta que kmer_topology.py, matrix.py, etc.:
    KmerTopology/kmer_frequency.py

Uso:
    from KmerTopology.kmer_frequency import create_frequency_matrix
    mat = create_frequency_matrix(list_secuences, kmers_size=4)
    # mat.shape == (n_secuencias, 4**4)

Detalles:
- Mayusculas/minusculas se tratan igual.
- Cualquier k-mer que contenga un caracter que no sea A/C/G/T (N, IUPAC,
  etc.) se ignora, y la normalizacion usa solo los k-mers validos.
- normalize=True (default): frecuencias relativas (suman 1). Con False
  regresa los conteos crudos.
- canonical=True: cada k-mer se junta con su reverso complementario (util
  para ADN de doble cadena como plasmidos). OJO: en ese caso la dimension
  ya no es 4^k sino el numero de k-mers canonicos (~4^k / 2).

Autor: Gerardo Rocha Ruiz Jr
"""
import numpy as np

_BASES = "ACGT"
MAX_K = 12  # 4^12 ~ 16.7M entradas por vector; mas alla no es practico

# tabla de busqueda byte -> codigo de base (A=0, C=1, G=2, T=3, otro=-1)
_LUT = np.full(256, -1, dtype=np.int8)
for _i, _b in enumerate(_BASES):
    _LUT[ord(_b)] = _i
    _LUT[ord(_b.lower())] = _i


def _check_k(kmers_size):
    if not isinstance(kmers_size, (int, np.integer)) or kmers_size < 1:
        raise ValueError(f"kmers_size debe ser un entero >= 1, se recibio {kmers_size!r}")
    if kmers_size > MAX_K:
        raise ValueError(f"kmers_size={kmers_size} es demasiado grande (maximo {MAX_K}): "
                         f"el vector tendria 4^{kmers_size} entradas")


def _encode(sequence):
    """Secuencia (str, Bio.Seq, bytes) -> arreglo de codigos 0-3 (o -1 si no es ACGT)."""
    if isinstance(sequence, (bytes, bytearray)):
        raw = bytes(sequence)
    else:
        raw = str(sequence).encode("ascii", errors="replace")
    return _LUT[np.frombuffer(raw, dtype=np.uint8)].astype(np.int64)


def _revcomp_index(kmers_size):
    """Para cada indice de k-mer, el indice de su reverso complementario."""
    idx = np.arange(4 ** kmers_size, dtype=np.int64)
    rc = np.zeros_like(idx)
    tmp = idx.copy()
    for _ in range(kmers_size):
        rc = rc * 4 + (3 - (tmp % 4))  # complemento: A<->T (0<->3), C<->G (1<->2)
        tmp //= 4
    return rc


def kmer_counts(sequence, kmers_size):
    """Conteo crudo de los 4^k k-mers de la secuencia (vector de longitud 4^k)."""
    _check_k(kmers_size)
    k = int(kmers_size)
    codes = _encode(sequence)
    n = codes.shape[0]
    counts = np.zeros(4 ** k, dtype=np.int64)
    if n < k:
        return counts

    n_windows = n - k + 1
    valid_base = codes >= 0

    # una ventana es valida si no contiene ninguna base invalida
    n_bad = np.concatenate(([0], np.cumsum(~valid_base)))
    window_valid = (n_bad[k:] - n_bad[:-k]) == 0

    # indice de cada k-mer en base 4 (las bases invalidas se reemplazan por
    # 0 solo para poder calcular el indice; esas ventanas se descartan abajo)
    safe = np.where(valid_base, codes, 0)
    idx = np.zeros(n_windows, dtype=np.int64)
    for j in range(k):
        idx = idx * 4 + safe[j:j + n_windows]

    return np.bincount(idx[window_valid], minlength=4 ** k).astype(np.int64)


def kmer_frequency_vector(sequence, kmers_size, normalize=True, canonical=False):
    """
    Vector de frecuencias de k-mers de UNA secuencia.

    Regresa un np.ndarray float64 de dimension 4^k (o la de k-mers
    canonicos si canonical=True).
    """
    counts = kmer_counts(sequence, kmers_size).astype(np.float64)

    if canonical:
        rc = _revcomp_index(int(kmers_size))
        folded = np.zeros_like(counts)
        np.add.at(folded, np.minimum(np.arange(counts.shape[0]), rc), counts)
        keep = np.arange(counts.shape[0]) <= rc  # un representante por clase
        counts = folded[keep]

    if normalize:
        total = counts.sum()
        if total > 0:
            counts = counts / total
    return counts


def kmer_labels(kmers_size, canonical=False):
    """Nombres de las columnas del vector, en el mismo orden (ej. 'AAAA', 'AAAC', ...)."""
    _check_k(kmers_size)
    k = int(kmers_size)
    labels = []
    for index in range(4 ** k):
        digits = []
        n = index
        for _ in range(k):
            digits.append(_BASES[n % 4])
            n //= 4
        labels.append("".join(reversed(digits)))
    if canonical:
        rc = _revcomp_index(k)
        keep = np.arange(4 ** k) <= rc
        labels = [lab for lab, flag in zip(labels, keep) if flag]
    return labels


def create_frequency_matrix(secuences, kmers_size, normalize=True, canonical=False):
    """
    Matriz de frecuencias de k-mers: una fila por secuencia, en el mismo
    orden de `secuences` (analogo a create_matrix_vectors de KmerTopology.matrix).

    Parameters
    ----------
    secuences : list
        Lista de secuencias (str o Bio.Seq).
    kmers_size : int
        Longitud k del k-mer.

    Returns
    -------
    np.ndarray de forma (len(secuences), 4^k)
    """
    rows = [kmer_frequency_vector(s, kmers_size, normalize=normalize, canonical=canonical)
            for s in secuences]
    if not rows:
        raise ValueError("La lista de secuencias esta vacia")
    return np.vstack(rows)
