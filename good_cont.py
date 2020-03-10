import numpy as np
from scipy.spatial.distance import cdist
import networkx as nx
from scipy import sparse
import matplotlib.pyplot as plt
import os

def mirror_points(points, W):

    x = points[:, 0]
    y = points[:, 1]

    p1 = np.transpose(np.array([-x, -y]))
    p2 = np.transpose(np.array([-x, y]))
    p3 = np.transpose(np.array([-x, 2*W-y]))
    p4 = np.transpose(np.array([x, -y]))
    p5 = np.transpose(np.array([x, y]))
    p6 = np.transpose(np.array([x, 2*W-y]))
    p7 = np.transpose(np.array([2*W-x, -y]))
    p8 = np.transpose(np.array([2*W-x, y]))
    p9 = np.transpose(np.array([2*W-x, 2*W-y]))
    a = np.concatenate((p1, p2, p3, p4, p5, p6, p7, p8, p9))
    return a

def pdist2(a, b):
    # d = np.sqrt(np.sum((a-b) ** 2))
    # print(d)
    # d = np.linalg.norm(a-b, axis=-1)
    # d = distance_matrix(a, b)
    d = cdist(a, b)
    return d

def calc_triplet_prob(p3,pideal, R,n, MIN_PRECISION):
    r = pdist2(p3, pideal)

    r = max(r, MIN_PRECISION)

    p = 1 - (1-r**2/R**2)**n
    return p


def build_triplets(points, b, EPSILON, W, MIN_PRECISION, lamb):
    N = len(points)
    if len(points) > 200:
        Emax = (EPSILON/(b*N*np.sqrt(N))) ** (1/(np.round(np.sqrt(N))-2))
    else:
        Emax = (EPSILON/(b*N*np.sqrt(N))) ** (1/(N-2))

    dists = pdist2(points, points)
    np.fill_diagonal(dists, np.inf)
    I = np.argsort(dists, axis=1)
    I = np.transpose(I)
    points_mirrored = mirror_points(points, W)
    points_mirrored = np.asarray(points_mirrored)
    triplets = np.full([N*b, 5], 1.0)
    # triplets = np.ones(N * b, 5) * -1
    count = 0
    test1 = 0
    for i in range(0, N):
        Ii = I[:, i]
        p1 = points[i, :]
        for jj in range(0, min(N, b)):
            j = Ii[jj]
            p2 = points[j, :]

            p2 = np.asmatrix(p2)
            p1 = np.asmatrix(p1)
            # p2 = np.asarray(p2)

            # d12 = np.linalg.norm(p2-p1, axis=-1)
            d12 = pdist2(p2, p1)
            if d12 < MIN_PRECISION:
                continue

            R = d12 * lamb

            dists_mirror = pdist2(p2, points_mirrored)
            test = np.squeeze(np.asarray(dists_mirror))
            N_local_points = 0
            N_local_points = sum(1 for i in test if i <= R) - 2

            pideal = 2*p2-p1

            dists_to_symmetric = pdist2(pideal, points)
            dists_to_symmetric = np.squeeze(np.asarray(dists_to_symmetric))
            Isym = np.argsort(dists_to_symmetric)

            Isym = np.delete(Isym, np.where(Isym == i))
            Isym = np.delete(Isym, np.where(Isym == j))


            # Isym[Isym == i] = []
            # Isym[Isym == j] = []

            for kk in range(0, min(2, len(Isym) + 1)):
                k = Isym[kk]
                if dists_to_symmetric[k] > R:
                    continue

                p3 = np.asmatrix(points[k, :])
                p_triplet = calc_triplet_prob(p3,pideal, R,N_local_points, MIN_PRECISION)
                if p_triplet > Emax:
                    continue
                else:
                    if count < N*b:
                        triplets[count, :] = [i, j, k, p_triplet, N_local_points]
                        count += 1
                    else:
                        triplets = np.vstack((triplets, [i, j, k, p_triplet[0, 0], N_local_points]))
    # triplets = np.asmatrix(triplets)
    # deleteValue = np.where(triplets[:, 1] == -1)
    # triplets = np.delete(triplets, deleteValue)
    # triplets[triplets[:, 1]==-1, :] = []
    T = len(triplets)

    I = np.argsort(triplets[:, 3])
    triplets = triplets[I, :]

    D = pdist2(triplets[:, [1, 2]], triplets[:, [0, 1]])
    D = D == 0
    # print(np.transpose(np.sum(D, 1)))
    value1 = np.asmatrix((np.sum(D, 0) == 0))
    value2 = np.asmatrix((np.sum(D, 1) == 0))
    # deleteValue = np.where(value1 and value2)
    # oneValue = value1 and value2
    mask1 = np.logical_and(value2, value1)
    mask1 = (np.array(mask1).flatten())
    mask1 = np.where(mask1==True)
    triplets = np.delete(triplets, mask1, axis=0)

    D = pdist2(triplets[:, [1, 2]], triplets[:, [0, 1]])
    D = D == 0
    P = pdist2(np.transpose(np.matrix(triplets[:, 3])), np.transpose(np.matrix(-triplets[:, 3])))
    D = np.multiply(D, P)
    D = sparse.csr_matrix(D)
    return triplets, D


def chain_nfa(chain, N, b):
    # print(chain)
    max_p = np.max(chain)
    L = np.max(chain.size)
    K = L+2

    Ntests = b * N * np.sqrt(N)

    NFA = Ntests * (max_p ** (K-2))

    return NFA, max_p


def get_fw_path(l, m, P):
    p = []
    while m>=0:
        p = np.append(p, m)
        m = P[l, m]

    p = np.fliplr(np.matrix(p))
    return p


def compute_paths_NFAs(PathDists, P, triplets, N, b):
    T = triplets.shape[0]
    NFAs = np.full_like(PathDists, np.inf)
    count = 0
    for x in range(0, T):
        for y in range(0, T):

            if PathDists[x, y] == np.inf:
                count+=1
                continue
            fw_path = get_fw_path(x, y, P)
            fw_path = np.squeeze(np.asarray(fw_path)).astype(int)
            probs = triplets[fw_path, 3]
            NFAs[x, y], _ = chain_nfa(probs, N, b)

    return NFAs


def ind2sub(array_shape, ind):
    # Gives repeated indices, replicates matlabs ind2sub
    rows = (ind.astype("int32") // array_shape[1])
    cols = (ind.astype("int32") % array_shape[1])
    return rows, cols


def masking_principle(NFAs, z, P, triplets, Debug=True):

    # I, J = np.unravel_index(NFAs.shape, z)
    z = z[z[:, 1].argsort()]
    I = z[:, 0]
    J = z[:, 1]

    nNFAs = NFAs[(z[:, 0], z[:, 1])]
    VV = np.sort(nNFAs)
    II = np.argsort(nNFAs)

    C = len(I)
    final_curves = [[]]

    for x in range(0, C):
        xx = II[x]
        i = I[xx]
        j = J[xx]
        fw_path = get_fw_path(i, j, P)
        # [triplets(fw_path, 1); triplets(fw_path(end), 2: 3)']
        fw_path = np.squeeze(np.asarray(fw_path)).astype(int)
        points_in_path = triplets[fw_path, 0]
        points_in_path = np.append(points_in_path, triplets[fw_path[-1], 1:3])

        masked = 0

        for c in range(0, len(final_curves[0])):
            existing_curve_points = final_curves[0][c]
            if np.sum(np.isin(points_in_path, existing_curve_points)) > 1:
                masked = 1
                break

        if masked:
            continue
        else:
            if len(points_in_path) > (len(np.unique(points_in_path)) + 1):
                continue

        # final_curves = np.append(final_curves, points_in_path)
        final_curves[-1].append(points_in_path)
        if Debug:
            print(len(final_curves))
            print(points_in_path)

    if VV.size != 0:
        minNFA = VV[0]
    else:
        minNFA = np.inf

    return final_curves, minNFA


def final_curve(points, b, lamb, outfname, Debug=True):
    EPSILON = 1
    W = 1
    MIN_PRECISION = W/256
    N = len(points)
    plt.scatter(points[:, 0], points[:, 1], s=1, c='g')
    maxPoint = np.max(points)
    if np.max(points) > 1:
        points = points / maxPoint

    print(points)

    if N <= 3:
        if Debug:
            print("Points are less. Cannot evaluate")
        return

    triplets, D = build_triplets(points, b, EPSILON, W, MIN_PRECISION, lamb)
    if Debug:
        print("Starting Floyd Warshall")
    dist_matrix, predecessors = sparse.csgraph.floyd_warshall(D, directed=True, return_predecessors=True)
    np.fill_diagonal(dist_matrix, np.inf)


    if Debug:
        print("Starting NFA calculation:")
    NFA = compute_paths_NFAs(dist_matrix, predecessors, triplets, N, b)

    if Debug:
        print("NFA Shape= ", NFA.shape)

    z = np.argwhere(NFA < EPSILON)

    final_curves, minNFA = masking_principle(NFA, z, predecessors, triplets, Debug=False)

    if Debug:
        print("Minimal NFA: ", minNFA)
        print("Final Curves: ", final_curves)

    return final_curves


if __name__ == "__main__":
    points = []
    f = open("examples\\simple_alignment.txt", 'r')
    x, y = zip(*[l.split() for l in f])
    points = np.zeros((len(x), 2))
    points[:, 0] = np.array(x)
    points[:, -1] = np.array(y)
    b = 5
    lamb = 4
    final_curves = final_curve(points, b, lamb, '')
    for k in range(0, len(final_curves[0])):
        curve = np.squeeze(final_curves[0][k]).astype(int)
        plt.plot(points[curve, 0], points[curve, 1], 'r,-', linewidth=1.5)
    plt.show()
