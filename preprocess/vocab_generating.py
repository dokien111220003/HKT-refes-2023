# import cPickle
import pickle as cPickle
if __name__ == "__main__":
    f_train = open("data/train_data", "r")
    user_dict = {}
    item_dict = {}
    cat_dict = {}

    print("vocab generating...")
    for line in f_train:
        arr = line.strip("\n").split("\t")
        clk = arr[0] #negative/positve samples (label)
        uid = arr[1] #userid
        mid = arr[2] #movieid
        cat = arr[3] #category
        timestepnow = arr[4] #unixtime
        mid_list = arr[5] #list of movie
        cat_list = arr[6] #list of cate
        timestep_list = arr[7] #list of unixtime

        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1 #dem so luong review cua user
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1 #dem so lan phim xuat hien
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1 #dem so lan category xuat hien
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(""):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1
        for c in cat_list.split(""):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

    sorted_user_dict = sorted(user_dict.items(),
                              key=lambda x: x[1], reverse=True) #nguoi nao review cang nhieu thi cang dc xep len tren, x[1] nghia la sap xep theo value
    sorted_item_dict = sorted(item_dict.items(),
                              key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(),
                             key=lambda x: x[1], reverse=True)

    uid_voc = {} #user id vocab, gom key la userid va value la index 
    index = 0
    for key, value in sorted_user_dict:
        uid_voc[key] = index #bay gio value se la index chu k phai so luong review nua 
        index += 1

    mid_voc = {} #movie id vocab
    mid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_item_dict:
        mid_voc[key] = index
        index += 1

    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    cPickle.dump(uid_voc, open("data/user_vocab.pkl", "wb"))
    cPickle.dump(mid_voc, open("data/item_vocab.pkl", "wb"))
    cPickle.dump(cat_voc, open("data/category_vocab.pkl", "wb"))
