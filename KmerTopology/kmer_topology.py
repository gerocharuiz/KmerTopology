# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:19:56 2025

@author: yutah
"""
import numpy as np
from KmerTopology._extract_kmers import find_kmers_position
from KmerTopology._compute_topology import compute_kmers_persistent_diagram, compute_kmers_betti, compute_filtration_topology

def KmerHomology(sequence, kmers_size, step_size, max_step):
    
    kmers_list, kmers_positions = find_kmers_position(sequence, kmers_size)
    
    bc_kmers = {}
    for kmers in kmers_list:
        positions = kmers_positions[kmers]
        pd_kmer = compute_kmers_persistent_diagram(positions, step_size, max_step)
        
        betti_curve_kmer = compute_kmers_betti(pd_kmer, step_size, max_step)
        bc_kmers[kmers] = betti_curve_kmer
        
    betti_features = []
    for kmers in kmers_list:
        betti_features.append(bc_kmers[kmers])
    betti_features = np.concatenate(betti_features)
    return betti_features

"""
Para una secuencia, un K y una filtracion, obtiene sus vectores topolicos
B y Lambda
"""
def KmerTopology(sequence, kmers_size, step_size, max_step):
    
    #Obtenemos la lista de listas de posciones donde cada L aparece en sequence
    kmers_list, kmers_positions = find_kmers_position(sequence, kmers_size)
    
    bc_kmers = {}
    eigmin_kmers = {}
    #Para cada L en { L_1, ... , L_{4^k} }
    for kmers in kmers_list:
        #Obtiene S_Lp, las posciones donde aparece en sequence
        positions = kmers_positions[kmers]
        #Obtiene su vector de carazteristicas con la filtracion dada
        betti_curve_kmer, eig_min_curve_kmer = compute_filtration_topology(positions, step_size, max_step)
        #Los guarda
        bc_kmers[kmers] = betti_curve_kmer
        eigmin_kmers[kmers] = eig_min_curve_kmer
        
    betti_features = []
    eigmin_features = []
    for kmers in kmers_list:
        betti_features.append(bc_kmers[kmers])
        eigmin_features.append(eigmin_kmers[kmers])
    betti_features = np.concatenate(betti_features)
    eigmin_features = np.concatenate(eigmin_features)
    #Al final cada uno de los vectores B_lp los concatena
    return betti_features, eigmin_features