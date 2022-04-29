import numpy as np

def dumpERIs(fcidump, header, int1e=None, int2e=None, ecore=0., tol=1e-9):
    with open(fcidump,'w') as f:
        f.writelines(header)
        n = int1e.shape[0]
        if int2e is not None:
            for i in range(n):
                for j in range(i + 1):
                    for k in range(i + 1):
                        if k == i: 	
                            lmax = j + 1
                        else:
                            lmax = k + 1
                        for l in range(lmax):
                            line = str(int2e[i, j, k, l])\
                            + ' ' + str(i + 1) \
                            + ' ' + str(j + 1) \
                            + ' ' + str(k + 1) \
                            + ' ' + str(l + 1) + '\n'
                            if np.abs(int2e[i, j, k, l]) > tol:
                                f.writelines(line)
        if int1e is not None:
            for i in range(n):
                for j in range(i + 1):
                    line = str(int1e[i, j])\
                    + ' ' + str(i + 1) \
                    + ' ' + str(j + 1) \
                    + ' 0 0\n'
                    if np.abs(int1e[i, j]) > tol:
                        f.writelines(line)
        line = str(ecore) + ' 0 0 0 0\n'
        f.writelines(line)
    return 0

def loadERIs(fcidump):
    with open(fcidump,'r') as f:
        line = f.readline().split(',')
        norb = int(line[0].split(' ')[-1])
        nelec= int(line[1].split('=')[-1])
        ms2  = int(line[2].split('=')[-1])
        f.readline()
        f.readline()
        f.readline()
        n = norb 
        e = 0.0
        int1e = np.zeros((n,n))
        int2e = np.zeros((n,n,n,n))
        for line in f.readlines():
            data = line.split()
            ind = [int(x)-1 for x in data[1:]]
            if ind[2] == -1 and ind[3]== -1:
                if ind[0] == -1 and ind[1] ==-1:
                    e = float(data[0])
                else :
                    int1e[ind[0],ind[1]] = float(data[0])
                    int1e[ind[1],ind[0]] = float(data[0])
            else:
                int2e[ind[0], ind[1], ind[2], ind[3]] = float(data[0])
                int2e[ind[1], ind[0], ind[2], ind[3]] = float(data[0])
                int2e[ind[0], ind[1], ind[3], ind[2]] = float(data[0])
                int2e[ind[1], ind[0], ind[3], ind[2]] = float(data[0])
                int2e[ind[2], ind[3], ind[0], ind[1]] = float(data[0])
                int2e[ind[3], ind[2], ind[0], ind[1]] = float(data[0])
                int2e[ind[2], ind[3], ind[1], ind[0]] = float(data[0])
                int2e[ind[3], ind[2], ind[1], ind[0]] = float(data[0])
    return e, int1e, int2e, norb, nelec, ms2

def gen_small_cas_Ham(info_cas, ncas):
    """
        generating effective Hamiltonian for small active (interacting) space
        including two 3p S orbitals and ncas 3d Fe orbitals around HOMO 
        from the CAS(14e,12o) Fe2S2 model Hamiltonian

        e.g., when ncas = 3, orbital space is divided by 3 core, 5 active, 4 external as
        act  act  core  core  core  act   act   act   ext   ext   ext   ext
        3p_S 3p_S 3d_Fe 3d_Fe 3d_Fe 3d_Fe 3d_Fe 3d_Fe 3d_Fe 3d_Fe 3d_Fe 3d_Fe
                                          HOMO  LUMO
    """
    ecore_c, int1e_c, int2e_c, _, nelec_c, _ = info_cas
    nocc = nelec_c // 2 
    assert nelec_c % 2 == 0
 
    if ncas % 2 == 0:
        nc_cas = nocc - ncas//2
        noc_cas = nocc + ncas//2
    else:
        nc_cas = nocc - ncas//2 - 1
        noc_cas = nocc + ncas//2
    assert noc_cas - nc_cas == ncas
 
    int1e = np.zeros((ncas+2, ncas+2))
    int2e = np.zeros((ncas+2, ncas+2, ncas+2, ncas+2))
 
    # effective int1e in small cas 
    int1e[:2,:2] = int1e_c[:2,:2].copy()
    int1e[:2,2:] = int1e_c[:2,nc_cas:noc_cas].copy()
    int1e[2:,:2] = int1e_c[nc_cas:noc_cas,:2].copy()
    int1e[2:,2:] = int1e_c[nc_cas:noc_cas,nc_cas:noc_cas].copy()
 
    int1e[:2,:2] += 2. * np.einsum('ijkk->ij', int2e_c[:2,:2,2:nc_cas,2:nc_cas])
    int1e[:2,:2] -= 1. * np.einsum('ikkj->ij', int2e_c[:2,2:nc_cas,2:nc_cas,:2])
    int1e[:2,2:] += 2. * np.einsum('ijkk->ij', int2e_c[:2,nc_cas:noc_cas,2:nc_cas,2:nc_cas])
    int1e[:2,2:] -= 1. * np.einsum('ikkj->ij', int2e_c[:2,2:nc_cas,2:nc_cas,nc_cas:noc_cas])
    int1e[2:,:2] += 2. * np.einsum('ijkk->ij', int2e_c[nc_cas:noc_cas,:2,2:nc_cas,2:nc_cas])
    int1e[2:,:2] -= 1. * np.einsum('ikkj->ij', int2e_c[nc_cas:noc_cas,2:nc_cas,2:nc_cas,:2])
    int1e[2:,2:] += 2. * np.einsum('ijkk->ij', int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,2:nc_cas,2:nc_cas])
    int1e[2:,2:] -= 1. * np.einsum('ikkj->ij', int2e_c[nc_cas:noc_cas,2:nc_cas,2:nc_cas,nc_cas:noc_cas])

    # int2e in small cas 
    int2e[:2,:2,:2,:2] = int2e_c[:2,:2,:2,:2].copy()
 
    int2e[2:,:2,:2,:2] = int2e_c[nc_cas:noc_cas,:2,:2,:2]
    int2e[:2,2:,:2,:2] = int2e_c[:2,nc_cas:noc_cas,:2,:2]
    int2e[:2,:2,2:,:2] = int2e_c[:2,:2,nc_cas:noc_cas,:2]
    int2e[:2,:2,:2,2:] = int2e_c[:2,:2,:2,nc_cas:noc_cas]
 
    int2e[:2,2:,2:,2:] = int2e_c[:2,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas]
    int2e[2:,:2,2:,2:] = int2e_c[nc_cas:noc_cas,:2,nc_cas:noc_cas,nc_cas:noc_cas]
    int2e[2:,2:,:2,2:] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,:2,nc_cas:noc_cas]
    int2e[2:,2:,2:,:2] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,:2]
 
    int2e[2:,2:,:2,:2] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,:2,:2]
    int2e[2:,:2,2:,:2] = int2e_c[nc_cas:noc_cas,:2,nc_cas:noc_cas,:2]
    int2e[2:,:2,:2,2:] = int2e_c[nc_cas:noc_cas,:2,:2,nc_cas:noc_cas]
    int2e[:2,2:,2:,:2] = int2e_c[:2,nc_cas:noc_cas,nc_cas:noc_cas,:2]
    int2e[:2,2:,:2,2:] = int2e_c[:2,nc_cas:noc_cas,:2,nc_cas:noc_cas]
    int2e[:2,:2,2:,2:] = int2e_c[:2,:2,nc_cas:noc_cas,nc_cas:noc_cas]
 
    int2e[2:,2:,2:,2:] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas]
 
    # core energy 
    ecore  = ecore_c
    ecore += 2. * np.einsum('ii->', int1e_c[2:nc_cas,2:nc_cas])
    ecore += 2. * np.einsum('iikk->', int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
    ecore -=      np.einsum('ikki->', int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])

    n = ncas+2 
    nelec = n+2 if n % 2 == 0 else n+3 
    twos = 0
    return ecore, int1e, int2e, n, nelec, twos 

def gen_Dyall_Ham(info_cas, ncas, rdm1):
    ecore_c, int1e_c, int2e_c, ncas_c, nelec_c, twos_c = info_cas
    nocc = nelec_c // 2 
    assert nelec_c % 2 == 0  

    if ncas % 2 == 0:
        nc_cas = nocc - ncas//2
        noc_cas = nocc + ncas//2
    else:
        nc_cas = nocc - ncas//2 - 1
        noc_cas = nocc + ncas//2
    assert noc_cas-nc_cas == ncas

    int1e = np.zeros(int1e_c.shape)
    int2e = np.zeros(int2e_c.shape)

    # Fock
    fock  = np.zeros(int1e_c.shape)
    fock  = int1e_c.copy()
    # J and K from core 
    fock += 2. * np.einsum('ijkk->ij',    int2e_c[:,:,2:nc_cas,2:nc_cas])
    fock -=      np.einsum('ikkj->ij',    int2e_c[:,2:nc_cas,2:nc_cas,:])
    # J and K from active
    fock +=      np.einsum('ijkl,kl->ij', int2e_c[:,:,:2,:2], rdm1[:2,:2])
    fock -= .5 * np.einsum('iklj,kl->ij', int2e_c[:,:2,:2,:], rdm1[:2,:2])
    fock +=      np.einsum('ijkl,kl->ij', int2e_c[:,:,:2,nc_cas:noc_cas], rdm1[:2,2:])
    fock -= .5 * np.einsum('iklj,kl->ij', int2e_c[:,:2,nc_cas:noc_cas,:], rdm1[:2,2:])
    fock +=      np.einsum('ijkl,kl->ij', int2e_c[:,:,nc_cas:noc_cas,:2], rdm1[2:,:2])
    fock -= .5 * np.einsum('iklj,kl->ij', int2e_c[:,nc_cas:noc_cas,:2,:], rdm1[2:,:2])
    fock +=      np.einsum('ijkl,kl->ij', int2e_c[:,:,nc_cas:noc_cas,nc_cas:noc_cas], rdm1[2:,2:])
    fock -= .5 * np.einsum('iklj,kl->ij', int2e_c[:,nc_cas:noc_cas,nc_cas:noc_cas,:], rdm1[2:,2:])
 
    # Dyall hamiltonian
    # mean-field CORE
    int1e[2:nc_cas,2:nc_cas] = fock[2:nc_cas,2:nc_cas].copy()
 
    # interacting ACTIVE
    #   effective int1e
    int1e[:2,:2] = int1e_c[:2,:2].copy()
    int1e[:2,nc_cas:noc_cas] = int1e_c[:2,nc_cas:noc_cas].copy()
    int1e[nc_cas:noc_cas,:2] = int1e_c[nc_cas:noc_cas,:2].copy()
    int1e[nc_cas:noc_cas,nc_cas:noc_cas] = int1e_c[nc_cas:noc_cas,nc_cas:noc_cas].copy()
    int1e[:2,:2] += 2. * np.einsum('ijkk->ij', int2e_c[:2,:2,2:nc_cas,2:nc_cas])
    int1e[:2,:2] -= 1. * np.einsum('ikkj->ij', int2e_c[:2,2:nc_cas,2:nc_cas,:2])
    int1e[:2,nc_cas:noc_cas] += 2. * np.einsum('ijkk->ij', int2e_c[:2,nc_cas:noc_cas,2:nc_cas,2:nc_cas])
    int1e[:2,nc_cas:noc_cas] -= 1. * np.einsum('ikkj->ij', int2e_c[:2,2:nc_cas,2:nc_cas,nc_cas:noc_cas])
    int1e[nc_cas:noc_cas,:2] += 2. * np.einsum('ijkk->ij', int2e_c[nc_cas:noc_cas,:2,2:nc_cas,2:nc_cas])
    int1e[nc_cas:noc_cas,:2] -= 1. * np.einsum('ikkj->ij', int2e_c[nc_cas:noc_cas,2:nc_cas,2:nc_cas,:2])
    int1e[nc_cas:noc_cas,nc_cas:noc_cas] += 2. * np.einsum('ijkk->ij', int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,2:nc_cas,2:nc_cas])
    int1e[nc_cas:noc_cas,nc_cas:noc_cas] -= 1. * np.einsum('ikkj->ij', int2e_c[nc_cas:noc_cas,2:nc_cas,2:nc_cas,nc_cas:noc_cas])
 
    int2e[:2,:2,:2,:2] = int2e_c[:2,:2,:2,:2].copy()
 
    int2e[:2,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[:2,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas]
    int2e[nc_cas:noc_cas,:2,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,:2,nc_cas:noc_cas,nc_cas:noc_cas]
    int2e[nc_cas:noc_cas,nc_cas:noc_cas,:2,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,:2,nc_cas:noc_cas]
    int2e[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,:2] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,:2]
 
    int2e[nc_cas:noc_cas,nc_cas:noc_cas,:2,:2] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,:2,:2]
    int2e[nc_cas:noc_cas,:2,nc_cas:noc_cas,:2] = int2e_c[nc_cas:noc_cas,:2,nc_cas:noc_cas,:2]
    int2e[nc_cas:noc_cas,:2,:2,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,:2,:2,nc_cas:noc_cas]
    int2e[:2,nc_cas:noc_cas,nc_cas:noc_cas,:2] = int2e_c[:2,nc_cas:noc_cas,nc_cas:noc_cas,:2]
    int2e[:2,nc_cas:noc_cas,:2,nc_cas:noc_cas] = int2e_c[:2,nc_cas:noc_cas,:2,nc_cas:noc_cas]
    int2e[:2,:2,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[:2,:2,nc_cas:noc_cas,nc_cas:noc_cas]
 
    int2e[nc_cas:noc_cas,:2,:2,:2] = int2e_c[nc_cas:noc_cas,:2,:2,:2]
    int2e[:2,nc_cas:noc_cas,:2,:2] = int2e_c[:2,nc_cas:noc_cas,:2,:2]
    int2e[:2,:2,nc_cas:noc_cas,:2] = int2e_c[:2,:2,nc_cas:noc_cas,:2]
    int2e[:2,:2,:2,nc_cas:noc_cas] = int2e_c[:2,:2,:2,nc_cas:noc_cas]
 
    int2e[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas] = int2e_c[nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas,nc_cas:noc_cas]
 
    # mean-field VIRTUAL
    int1e[noc_cas:,noc_cas:]  = fock[noc_cas:,noc_cas:].copy()
 
    # core energy
    ecore  = ecore_c
    ecore += 2. * np.einsum('ii->',      int1e_c[2:nc_cas,2:nc_cas])
    ecore += 2. * np.einsum('iikk->',    int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
    ecore -=      np.einsum('ikki->',    int2e_c[2:nc_cas,2:nc_cas,2:nc_cas,2:nc_cas])
    ecore -= 2. * np.einsum('ii->',      fock[2:nc_cas,2:nc_cas])
    return ecore, int1e, int2e, ncas_c, nelec_c, twos_c

