# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


##this code is to aggregate the results from validation set and finding best threshold.
import  pandas as pd
import  numpy as np

import matplotlib.pyplot as plt
path="./threshold_variation_results_with_validation_set/"
#path="./"



files =[
         path+"cross-validation-BERT-bert.csv",
         path+"cross-validation-XLNET-xlnet.csv",
         path+"cross-validation-DBERT-dbert.csv",
         path+"cross-validation-ALBERT-albert.csv",
         path+"cross-validation-ROBERTA-roberta.csv"
]

for datafile in files:

    dataframe =pd.read_csv(datafile)
    aggregateDF=dataframe.groupby("Threshold").mean()
    plt.plot(aggregateDF["precision_1"], label ="Precision toxic", linestyle="-.")
    plt.plot(aggregateDF["recall_1"], label ="Recall toxic", linestyle="--")
    plt.plot(aggregateDF["f-score_1"], label ="F1-score toxic", linestyle=":")


    xmax = np.argmax(aggregateDF["f-score_1"])
    xmax=(float)(xmax/100)
    y_max=max(aggregateDF["f-score_1"])
    print(y_max)
    print(xmax)
    #print(aggregateDF["f-score_1"].shape)
    plt.legend()
    #plt.title("")

    text = "Threshold={:.2f}, F1={:.2f}".format(xmax, y_max)
    #if not ax:
        #ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    plt.annotate(text, xy=(xmax, y_max), xytext=(0.6, 0.93), **kw)
    plt.xlabel('Threshold' )
    plt.ylabel('Score')

    #plt.annotate('max',xy=(xmax,y_max))
    plt.show()
    aggregateDF.to_excel(datafile+".xlsx")



