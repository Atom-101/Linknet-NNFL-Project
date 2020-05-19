import numpy as np

def gen_cls_wts(train_dl):
    class_counts = [0]*12
    for batch in train_dl:
        unique,counts = np.unique(batch['mask'],return_counts=True)
        for i,u in enumerate(unique):
            class_counts[int(u.item())] += counts[i]

    class_counts = np.array(class_counts)/sum(class_counts)
    print(class_counts)
    class_weights = 1/np.log(1.02 + class_counts)
    print(class_weights)
    np.save('cls_wts.npy',class_weights)