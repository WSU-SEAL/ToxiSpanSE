# Copyright Software Engineering Analytics Lab (SEAL), Wayne State University, 2023
# Authors: Jaydeb Sarker <jaydebsarker@wayne.edu> and Amiangshu Bosu <abosu@wayne.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# version 3 as published by the Free Software Foundation.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.


##this model is for getting the character spans from the final_label from label studio exported file
import pandas as pd
import re

import xlwt
from xlwt import Workbook
wb = Workbook()
sheet1 = wb.add_sheet('Sheet 1')

df=pd.read_excel('models/CR_full_span_dataset.xlsx')

total_len=len(df)
#total_len=3757

def span_offset_cal(pattn,sp):
    #print(pattn)
    patn = re.sub(r"[\([{})\]]", "", pattn)
    patn=patn.replace('"','')
    patn=patn.replace(':','')
    patn=patn.replace(",",'')
    #print(patn)
    ls=patn.split()

    for i in range(0,len(ls)-2):

        if(ls[i]=="start"):
            start=ls[i+1]
            end=ls[i+3]
            for k in range(int(start),int(end)):
                sp.append(k)

    return sp

max_len=0
max_len_pos=0
sheet1.write(0, 0, "id")
sheet1.write(0,1,"spans")

for j in range(0,total_len):
    span = []
    if(pd.isnull(df['Final_label'][j])):
        sheet1.write(j+1, 0, str(df['id'][j]))
        sheet1.write(j+1, 1, str("[]"))
        #print(df['id'][j])
        #continue
    else:
        spt=span_offset_cal(df['Final_label'][j], span)
        spt.sort()
        dt=str(spt)
        sheet1.write(j+1,0,str(df['id'][j]))
        sheet1.write(j+1, 1, dt)

wb.save('final_span_offset.xlsx')