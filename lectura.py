import numpy as np
from KmerTopology.kmer_topology import KmerTopology, KmerHomology
from Bio import SeqIO
#Libreria para leer carpetas
from pathlib import Path
import statistics
import matplotlib.pyplot as plt

secuencias = list(SeqIO.parse("ERZ3081675.fna", "fasta"))

#print(f"Número de secuencias: {len(secuencias)}")

long = 0
for secuencia in secuencias:
    #print(secuencia.id, f"Longitud de la secuencia: {len(secuencia.seq)}")
    long += len(secuencia.seq)
#print(secuencias[0].seq)
#print(long)

"""
Este es un análisis exploratorio para ver cuantos de nuestros archivos .fna están
divididos en contigs
"""
carpeta = Path("/files2/generic-amr/data/genomes")
lista_contigs = []

for archivo in carpeta.glob("*.fna"):
    contigs = list(SeqIO.parse(archivo, "fasta"))
    lista_contigs.append(len(contigs))
    #print(archivo.name, f"Número de contigs: {len(contigs)}")

print(f"Promedio: {statistics.mean(lista_contigs)}")
print(f"Varianza: {statistics.variance(lista_contigs)}")

plt.hist(lista_contigs, bins = 20)
plt.xlabel("No. Contigs")
plt.ylabel("Frecuencia")
plt.show()
