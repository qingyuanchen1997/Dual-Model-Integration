import os
root = os.path.dirname(os.path.dirname(__file__))
final_add = root+'/classification_result/result.txt'
path = root+'/user_data/tmp_data/results/'
txt1_add=path+'result_1.txt'
txt2_add=path+'result_2.txt'

with open(final_add, 'w',encoding='utf-8') as f:
    with open(txt1_add,'r') as f1:
        with open(txt2_add, 'r') as f2:
            result1 = f1.read().split('\n')
            result2 = f2.read().split('\n')
            for i in range(1172):
                idx=result1[i].split(' ')[0]
                f.write(idx)
                f.write(' ')
                if result2[i].split(' ')[1]=='3':
                    f.write(result2[i].split(' ')[1])
                elif result1[i].split(' ')[1]=='3':
                    f.write('2')
                else:
                    f.write(result1[i].split(' ')[1])
                f.write('\n')
            