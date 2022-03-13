#


#import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#        0                                                                                           


f1 = [0.7259672837692231, 0.7385002233453621, 0.7840485074626866, 0.8037900564249973, 0.6405228758169934, 0.9407433033507957, 0.9040241448692152, 0.7138727191811304, 0.9617039964866052, 0.7906153630501822, 0.7316166531494999, 0.7346059113300492, 0.9334031048316762, 0.8244443316756318, 0.9353213507625272, 0.8041429002872313, 0.7795621650812113, 0.7349505840071877, 0.8438229897156513, 0.9410868994486743]

ins = {"accordion": 0, "banjo": 1, "bass": 2, "cello": 3, "clarinet": 4, "cymbals": 5, "drums": 6, "flute": 7, "guitar": 8, "mallet_percussion": 9, "mandolin": 10, "organ": 11, "piano": 12, "saxophone": 13, "synthesizer": 14, "trombone": 15, "trumpet": 16, "ukulele": 17, "violin": 18, "voice": 19}
ins_name = ["accordion", "banjo", "bass", "cello", "clarinet", "cymbals", "drums", "flute", "guitar", "mallet_percussion", "mandolin", "organ", "piano", "saxophone", "synthesizer", "trombone", "trumpet", "ukulele", "violin", "voice"]

#print(ins["accordion"])
'''
plt.bar(ins_name, f1)
#plt.grid(True)
plt.title('FixMatch F1 score over the instruments')
#plt.legend(['90%/10%', '80%/20%', '95%/5%'])
#plt.xlabel('Instrument')
plt.xticks(rotation=90)
plt.ylim((0.4, 1)) 
plt.ylabel('F1 score')
plt.savefig('Ins_f1.png')
plt.show()
'''

f1_bs = [0.6942613016299292, 0.7308396178984414, 0.7856481481481481, 0.806, 0.6116432345637521, 0.9313459346085773, 0.9029476843256441, 0.7098458406050029, 0.9614493624997789, 0.7781805446108045, 0.737407058931522, 0.6746002676095293, 0.944985173556602, 0.8342018289097508, 0.9417892156862744, 0.8058734411243591, 0.782384856045305, 0.7497422377677139, 0.839793500338524, 0.9436831802439392]
'''
index = np.arange(20)
bar_width = 0.4
plt.bar(index, f1, bar_width)
plt.bar(index+bar_width, f1_bs, bar_width)

#plt.grid(True)
plt.title('Baseline/FixMatch F1 score over the instruments')
plt.legend(['FixMatch','Baseline'])
plt.xticks((index + bar_width / 2), ins_name)
#plt.xticklabels(ins_name)
#plt.xlabel('Instrument')
plt.xticks(rotation=90)
plt.ylim((0.4, 1)) 
plt.ylabel('F1 score')
plt.tight_layout()
plt.savefig('./plot/Ins_f1_bs.pdf')
plt.show()
'''
pos_label = [ 489.,  732.,  549.,  824.,  533., 1111., 1106.,  647., 1138., 733.,  845.,  603., 1170., 1135., 1091.,  863., 1146.,  738., 1173.,  988.]

neg_label = [1582., 1486., 1339., 1125., 1852.,  624.,  641., 1437.,  512., 1069., 1619., 1287.,  550., 1230.,  511., 1897., 1770., 1687., 860.,  576.]

sum_pos = sum(pos_label)
pos_label_norm = [num/sum_pos for num in pos_label]

#f1_dif = f1 - f1_bs
f1_dif = []
zip_object = zip(f1, f1_bs)
for list1_i, list2_i in zip_object:
    f1_dif.append(list1_i-list2_i)

index = np.arange(20)
bar_width = 0.4
plt.bar(index, f1_dif, bar_width)
plt.bar(index+bar_width, pos_label_norm, bar_width)
#plt.grid(True)
plt.title('FixMatch F1 improvement vs pos label number')
plt.legend(['F1 difference','Pos label'])
plt.xticks((index + bar_width / 2), ins_name)
#plt.xticklabels(ins_name)
#plt.xlabel('Instrument')
plt.xticks(rotation=90)
#plt.ylim((0, 0.07)) 
plt.ylabel('F1 score improvement')
plt.tight_layout()
plt.savefig('./plot/Ins_f1_pos.pdf')
plt.show()




