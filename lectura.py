import numpy as np
from KmerTopology.kmer_topology import KmerTopology, KmerHomology
from Bio import SeqIO

secuencias = list(SeqIO.parse("ERZ3081675.fna", "fasta"))

print(f"Número de secuencias: {len(secuencias)}")

long = 0
for secuencia in secuencias:
    print(secuencia.id, f"Longitud de la secuencia: {len(secuencia.seq)}")
    long += len(secuencia.seq)
#print(secuencias[0].seq)
print(long)
