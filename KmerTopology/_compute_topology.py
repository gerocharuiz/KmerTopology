# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 15:03:58 2025

@author: yutah
"""

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.space.linalg import eigsh
from gudhi.representations.vector_methods import BettiCurve
import ripser

"""
Se da una lista de posiciones en los que aparece el kmer L, una filtracion de 0 a r_max
Obtiene aristas. Y para cada r en 0 a r_max calcula L y sus B y Lambda asociados.
Al final nos entrega los vecotres de caracteristicas topologicas para ese L y esa
filtracion
"""
def compute_filtration_topology(positions, step_size, max_step):
    '''
        Compute all the eigenvalues associated with the filtration (step)

    Parameters
    ----------
    D : np.array
        distance matrix for kmers

    Returns
    -------
    eigenval_list : list
        list all the eigenvalues. The first few should be 0, and the number corresponds to the Betti 0, or the number of connected components
        

    '''

    """
    Lista de posiciones en las que aparece el kmer dado en el genoma,
    notemos que esta ordenado
    """
    positions = positions[positions> 0]  #only keep the nonzero entry
    positions = np.sort(positions)
    #Numero de posiciones
    n = positions.shape[0]

    #Obtenemos todas las filtraciones
    filtration = np.linspace(0, step_size*max_step, max_step)
    betti = np.zeros(max_step, dtype=int)
    eig_min = np.zeros(max_step, dtype=int)

    if n == 0:
        return betti, eig_min
    if n == 1:
        return betti, eig_min

    #La mayor filtracion posible
    max_threshold = filtration[-1]
    
        # --- Construccion de aristas sin matriz de distancias densa ---
    upper = np.searchsorted(positions, positions + max_threshold, side='right')
    upper = np.minimum(upper, n)

    # Para guardar las aristas
    rows_chunks, cols_chunks, w_chunks = [], [], []
    # Recorremos cada posicion
    for i in range(n):
        # Para la posicion i nos dice hasta donde podemos llegar
        j_hi = upper[i]
        #Si exiten posiciones a las que podemos llegar
        if j_hi > i + 1:
            # Agarra a las posciones posibles
            js = np.arange(i + 1, j_hi)
            # Guarda las aristas
            rows_chunks.append(np.full(js.shape[0], i, dtype=np.int64))
            cols_chunks.append(js.astype(np.int64))
            # Guardamos las distancias
            w_chunks.append(positions[js] - positions[i])

    if not rows_chunks:
        # Ninguna posicion queda dentro del umbral maximo de nadie mas:
        # todos los nodos son su propia componente en toda la filtracion
        betti[:] = n
        return betti, eig_min

    # Gardamos todas las aristas y pesos
    rows = np.concatenate(rows_chunks)
    cols = np.concatenate(cols_chunks)
    weights = np.concatenate(w_chunks)

    for k, s in enumerate(filtration):
        # Si no existen filtraciones
        if s == 0:
            betti[k] = n
            eig_min[k] = 0
            continue

        #Nos fijamos en que aristas cumplen con estar a distancia menor a s
        mask = weights <= s
        if not np.any(mask):
            betti[k] = n
            eig_min[k] = 0
            continue

        # Matriz de adayacencia con mascaras booleanas
        r, c = rows[mask], cols[mask]

        # Matriz de adyacencia dispersa y simetrica (nunca se materializa D)
        A = sparse.coo_matrix(
            (np.ones(2 * r.shape[0]),
             (np.concatenate([r, c]), np.concatenate([c, r]))),
            shape=(n, n)
        ).tocsr()

        n_comp, labels = connected_components(A, directed=False)
        betti[k] = n_comp

        # Laplaciano
        deg = np.asarray(A.sum(axis=1)).flatten()
        L = (sparse.diags(deg) - A).tocsr()

        # eig_min = menor eigenvalor positivo entre TODAS las componentes
        # conexas. Pedir globalmente los (n_comp+1) eigenvalores mas chicos
        # es inviable cuando hay muchas componentes (n_comp puede ser miles).
        # En cambio, como no hay aristas entre componentes distintas, el
        # Laplaciano es block-diagonal: basta resolver, POR COMPONENTE, un
        # problema de eigenvalores chico (k=2: el 0 y el siguiente), y
        # tomar el minimo de esos "Fiedler values" sobre todas las
        # componentes no triviales. Esto es exacto, no una aproximacion.
        best = None
        for comp_id in np.unique(labels):
            idx = np.where(labels == comp_id)[0]
            if idx.shape[0] < 2:
                continue  # nodo aislado: no aporta eigenvalor positivo
            Lc = L[idx][:, idx]
            if idx.shape[0] <= 300:
                vals_c = np.linalg.eigvalsh(Lc.toarray())
            else:
                try:
                    vals_c = eigsh(Lc.asfptype(), k=2, sigma=-1e-8,
                                   which='LM', return_eigenvectors=False)
                except Exception:
                    vals_c = eigsh(Lc.asfptype(), k=2, which='SM',
                                   return_eigenvectors=False)
            vals_c = np.sort(vals_c)
            vals_c = vals_c[vals_c > 1e-8]
            if vals_c.shape[0] > 0:
                cand = vals_c[0]
                if best is None or cand < best:
                    best = cand
        eig_min[k] = best if best is not None else 0

    return betti, eig_min

def compute_kmers_persistent_diagram(positions, step_size, max_step):
    '''
        Compute the persistnet diagram for a given kmer
        
    Parameters
    ----------
    kmers : str
        The k-mer you want. This needs to match the kmers_size you initialize with

    Returns
    -------
    pd_kmers : np.array
        persistent diagram. the matrix is M by 2, where M is the number of kmer.
        pd_kmers[:, 0] is the birth time, pd_kmers[:,1] is the death time. 
        Note that pd_kmers[:, 0] are all 0 because we are only doing 0-th order.
    '''
        
    
    positions = positions[positions> 0]  #only keep the nonzero entry
    
    #dis = sparse.coo_array((pos.shape[0], pos.shape[0]))
    row = []; col = []; val = []
    if positions.shape[0] > 1:
        for i in range(positions.shape[0]): #compute the neighboring positions' distance
            if i ==0:
                row.append(i); col.append(i+1); val.append(positions[i+1] - positions[i])
            elif i == positions.shape[0] - 1:
                row.append(i); col.append(i-1); val.append(positions[i] - positions[i-1])
            else:
                row.append(i); col.append(i-1); val.append(positions[i] - positions[i-1])
                row.append(i); col.append(i+1); val.append(positions[i+1] - positions[i])
        row = np.array(row); col = np.array(col); val = np.array(val)
        dis = sparse.coo_array( (val, (row, col)), shape = (positions.shape[0], positions.shape[0]))
        pd_kmers = ripser.ripser(dis, thresh = step_size*max_step, distance_matrix = True)['dgms'][0]
    elif positions.shape[0] == 1:
        pd_kmers = np.array([[0, np.inf]])
    elif positions.shape[0] == 0:
        pd_kmers = np.array([[0, 0]])
    return pd_kmers



    
def compute_kmers_betti(pd_kmers, step_size, max_step):
    '''
    

    Parameters
    ----------
    pd_kmers : np.array
        Persistent diagram

    Returns
    -------
    betti_curve : 1d vector of np.array
        Betti curve.

    '''
    if pd_kmers.ndim == 1:
        if pd_kmers[1] == 0:
            betti_curve_kmers = np.array([0,0])
        else:
            betti_curve_kmers = np.array([1,1])
    else:
        sorted_kmers = pd_kmers[:, 1].copy()
        sorted_kmers[sorted_kmers == np.inf] = 0
        filtration = np.linspace(0, step_size*max_step, max_step)
        
        bc = BettiCurve(predefined_grid = filtration)
        betti_curve_kmers = bc.fit_transform([pd_kmers]).reshape(-1)
    return betti_curve_kmers