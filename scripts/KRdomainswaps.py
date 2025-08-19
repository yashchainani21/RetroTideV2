from retrotide import retrotide, structureDB
import bcs
from collections import OrderedDict
from similarity_scores import fingerprints, similarity
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt

#Define KR types
KR_types_branched = ['A1', 'A2', 'B1', 'B2'] #When loading methylmalonyl coA

target_product = 'CC[C@@H]1[C@H](C)[C@H](O)[C@@H](C)C([C@H](C)C[C@H](C)[C@H](O)[C@@H](C)[C@H](O)[C@@H](C)C(O1)=O)=O'

#Check the AT substrate
#Check if a KR domain is present
#Enumerate over all KR domains
#Make KR type swaps and test all possible combinations
#For each combination store the product string in a list
#Compare the product string with the target product string
#Use mapchiral-jaccard combination to calculate a similarity score

#Dictionary: Substrates, KR types present, Product String, Similarity Score
PKS_product_str = []
KR_combinations = []
for i in KR_types_branched:
    for j in KR_types_branched:
        for k in KR_types_branched:
            for l in KR_types_branched:
                current_KR_comb = f"{i},{j},B1,{k},{l}"
                #current_KR_comb = [i, j, 'B1', k, l]
                KR_combinations.append(current_KR_comb)

                #Construct the Erythromycin PKS
                #Loading Module
                AT_domain_with_propionylCoA = bcs.AT(active = True, substrate = 'prop')
                domains_dict = OrderedDict({bcs.AT: AT_domain_with_propionylCoA})
                loading_module = bcs.Module(domains = domains_dict, loading = True)

                #Module 1: Add methylmalonyl-CoA
                AT_domain_mod1 = bcs.AT(active = True, substrate = 'Methylmalonyl-CoA')
                KR_domain_mod1 = bcs.KR(active = True, type = i)
                module1_domains_dict = OrderedDict({bcs.AT: AT_domain_mod1,
                                                    bcs.KR: KR_domain_mod1})
                module1 = bcs.Module(domains = module1_domains_dict, loading = False)

                #Module 2: Add methylmalonyl-CoA
                AT_domain_mod2 = bcs.AT(active = True, substrate = 'Methylmalonyl-CoA')
                KR_domain_mod2 = bcs.KR(active = True, type = j)
                module2_domains_dict = OrderedDict({bcs.AT: AT_domain_mod2,
                                                    bcs.KR: KR_domain_mod2})
                module2 = bcs.Module(domains = module2_domains_dict, loading = False)

                #Module 3: Add methylmalonyl-CoA
                AT_domain_mod3 = bcs.AT(active = True, substrate = 'Methylmalonyl-CoA')
                module3_domains_dict = OrderedDict({bcs.AT: AT_domain_mod3})
                module3 = bcs.Module(domains = module3_domains_dict, loading = False)

                #Module 4: Add methylmalonyl-CoA
                AT_domain_mod4 = bcs.AT(active = True, substrate = 'Methylmalonyl-CoA')
                KR_domain_mod4 = bcs.KR(active = True, type = "B1") #has to be B1 because of the mmal substrate AND the DH and ER
                DH_domain_mod4 = bcs.DH(active = True)
                ER_domain_mod4 = bcs.ER(active = True)
                module4_domains_dict = OrderedDict({bcs.AT: AT_domain_mod4,
                                                    bcs.KR: KR_domain_mod4,
                                                    bcs.DH: DH_domain_mod4,
                                                    bcs.ER: ER_domain_mod4})
                module4 = bcs.Module(domains = module4_domains_dict, loading = False)

                #Module 5: Add methylmalonyl-CoA
                AT_domain_mod5 = bcs.AT(active = True, substrate = 'Methylmalonyl-CoA')
                KR_domain_mod5 = bcs.KR(active = True, type = k)
                module5_domains_dict = OrderedDict({bcs.AT: AT_domain_mod5,
                                                    bcs.KR: KR_domain_mod5})
                module5 = bcs.Module(domains = module5_domains_dict, loading = False)

                #Module 6: Add methylmalonyl-CoA
                AT_domain_mod6 = bcs.AT(active = True, substrate = 'Methylmalonyl-CoA')
                KR_domain_mod6 = bcs.KR(active = True, type = l)
                module6_domains_dict = OrderedDict({bcs.AT: AT_domain_mod6,
                                                    bcs.KR: KR_domain_mod6})
                module6 = bcs.Module(domains = module6_domains_dict, loading = False)

                #Terminal Domain: Hydrolysis to cleave thioesterbond
                TE_domain_mod7 = bcs.TE(active = True, cyclic = True, ring = 3)
                module7_domains_dict = OrderedDict({bcs.TE: TE_domain_mod7})
                module7 = bcs.Module(domains = module7_domains_dict, loading = False)

                cluster = bcs.Cluster(modules=[loading_module, module1, module2, module3, module4, module5, module6, module7])
                mol = cluster.computeProduct(structureDB)
                PKS_product = Chem.MolToSmiles(mol)
                PKS_product_str.append(PKS_product)

scores = []
for product in PKS_product_str:
    fp_1 = fingerprints.get_fingerprint(target_product, 'mapchiral')
    fp_2 = fingerprints.get_fingerprint(product, 'mapchiral')

    score = similarity.get_similarity(fp_1, fp_2, 'jaccard')
    scores.append(score)

print(len(scores))

plt.bar(KR_combinations[185: 200], scores[185: 200], color = 'skyblue')
plt.xlabel('KR Types')
plt.ylabel('Mapchiral-Jaccard Similarity Score')
plt.ylim(0, 1)
plt.xticks(rotation = 45, ha = 'right', fontsize = 8)

for idx, score in enumerate(scores[185: 200]):
    plt.text(idx, score + 0.01, f"{score:.3f}", ha = 'center', fontsize = 7)

plt.tight_layout()
plt.show()

'''
for idx, i in enumerate(scores):
    count = 0
    if i == 1:
        count += 1
        print('True')
        print(idx)
        print(count)
'''