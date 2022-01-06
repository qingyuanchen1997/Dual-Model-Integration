def save_pred(y_pred, res_path):
    f = open(res_path, mode='w')
    for i in range(len(y_pred)):
        content = str(i+1).zfill(6) + ' ' + str(int(y_pred[i]+1)) + '\n'
        f.write(content)