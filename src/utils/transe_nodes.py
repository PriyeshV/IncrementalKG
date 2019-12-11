import pandas as pd
import pickle
import numpy as np
# df = pd.read_csv('train1000.txt', sep=' ', header=None)


with open('/home/nishant/Downloads/_map_entnew-id_b.pkl', 'rb') as inp:
    ent = pickle.load(inp)
# with open('train_map_rel-id_b.pkl', 'rb') as inp:
#     rel = pickle.load(inp)


# df[0] = df[0].map(ent)
# df[1] = df[1].map(ent)
# df[2] = df[2].map(rel)
# df.to_csv('train_fb20k.txt',sep=' ',index=False)
print(len(ent))
keys = []
values = []
for key, value in ent.items():
    keys.append(key)
    values.append(value)

df2 = pd.DataFrame()
df2['key'] = keys
df2['value']= values

# keys = []
# values = []
# for key, value in rel.items():
#     keys.append(key)
#     values.append(value)

# df3 = pd.DataFrame()
# df3['key'] = keys
# df3['value']= values


df2.to_csv('entity2id.txt',sep='\t',index=False,header=False)
# df2.to_csv('relation2id.txt',sep='\t',index=False, header=False)


print(len(df2))

# def new_matrix(nodes):

#     for n in nodes:
#         df.drop(df.loc[df[0] == n].index, inplace=True)
#         df.drop(df.loc[df[1] == n].index, inplace=True)
#     return df




# with open('train_3951iter1removed_nodes.pkl', 'rb') as inp:
#     nodes = pickle.load(inp)

# df10 = new_matrix(nodes)

# print(len(df10))
# df10.to_csv('train_3951iter1removed_nodes.txt', sep=' ',index=False,header=False)