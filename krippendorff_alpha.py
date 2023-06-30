# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


##main source of this code of krippendorff's alpha is from the following link
###https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py
##krippendroff's alpha


import pandas as pd
import re
import xlwt
from xlwt import Workbook
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')

#read our CR dataset
#df=pd.read_csv('models/CR_full_span_dataset.csv')
df=pd.read_excel('models/CR_full_span_dataset.xlsx')

def label_attention_list(text, annotator_label,label_mat):
    #Code Rrevie message processing
    text = re.sub(r"[\([{})\]]", "", text)
    text=text.replace('"','')
    text=text.replace(':','')
    text=text.replace(",",'')
    text=text.replace("-"," ")
    text=text.replace("/"," ")
    sp_ls=text.split()

    #label processing
    pattn=annotator_label
    patn = re.sub(r"[\([{})\]]", "", pattn)
    patn=patn.replace('"','')
    patn=patn.replace(':','')
    patn=patn.replace(",",'')
    patn=patn.replace("-"," ")

    ls=patn.split()


    ##label preprocess
    ##add which tokens are selected
    list_selected_tokens=[]
    i=0
    for i in range(0,len(ls)-1):
        start=0
        if(ls[i]=='text'):
            start=i+1
            while(ls[start]!='labels'):
                #print(" ", i)

                list_selected_tokens.append(ls[start])
                start=start+1
                i=i+1


    for k in range(0,len(sp_ls)):
        label_mat.append(0)

    #count_len=0
    for p in range(0,len(list_selected_tokens)):
        for k in range(0, len(sp_ls)):
            if(sp_ls[k]==list_selected_tokens[p]):
                label_mat[k]=1

    return label_mat




try:
    import numpy as np
except ImportError:
    np = None


def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a - b) ** 2


def ratio_metric(a, b):
    return ((a - b) / (a + b)) ** 2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    # number of coders
    m = len(data)

    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)

    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)

        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)

    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n * (n - 1))

    return 1. - Do / De if (Do and De) else 1.
###

cnt=0
total_k=0
lb_rater1 = []
lb_rater2 = []
total_len=3742
for c in range(0,total_len):

    if(pd.isnull(df['rater1_label'][c]) or pd.isnull(df['rater2_label'][c])):
        print(c, df['id'][c])
    else:
        lb_rater1.append(label_attention_list(df['text'][c], df['rater1_label'][c], []))
        lb_rater2.append(label_attention_list(df['text'][c], df['rater2_label'][c], []))


from itertools import chain
lb_rater1=list(chain.from_iterable(lb_rater1))
lb_rater2=list(chain.from_iterable(lb_rater2))
print(len(lb_rater1))
print(len(lb_rater2))
array = [lb_rater1, lb_rater2]
missing = '*'
alpha=krippendorff_alpha(array, nominal_metric, missing_items=missing)

print("Krippendorff alpha score is: ", alpha)