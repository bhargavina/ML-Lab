import csv

with open('candidate-elimination-training-examples.csv') as file:
    data = [tuple(line) for line in csv.reader(file)]

D = []
for i in range(len(data[0])):
    D.append(list(set(ele[i] for ele in data)))

def consistent(h1, h2):
    for x, y in zip(h1, h2):
        if not (x == '?' or (x != '$' and (x == y or y == '$'))):
            return False
    return True

def candidateElimination():
    G = {('?', ) * (len(data[0]) - 1), }
    S = ['$'] * (len(data[0]) - 1)
    num = 0
    print('G[{0}]: '.format(num), G)
    print('S[{0}]: '.format(num), S)
    for item in data:
        inp, res = item[: -1], item[-1]
        num += 1
        if res == 'Y':
            G = {g for g in G if consistent(g, inp)}
            i = 0
            for s, x in zip(S, inp):
                if s != x:
                    S[i] = '?' if s != '$' else x
                i += 1
        else:
            S = S
            Gprev = G.copy()
            for g in Gprev:
                if g not in G:
                    continue
                for i in range(len(g)):
                    if g[i] == '?':
                        for val in D[i]:
                            if val == S[i] and val != inp[i]:
                                G.add(g[:i] + (val, ) + g[i + 1:])
                    else:
                        G.add(g)
                G.difference_update([h for h in G if any([consistent(h, g1) for g1 in G if h != g1])])
        print('\nG[{0}]: '.format(num), G)
        print('S[{0}]: '.format(num), S)

candidateElimination()
