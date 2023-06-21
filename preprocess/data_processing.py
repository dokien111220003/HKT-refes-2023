import sys
import random
import time

#meta_readfile: meta_Movies_and_TV.json, meta_writefile: meta_information bao gom asin | category
def meta_preprocessing(meta_readfile, meta_writefile):
    meta_r = open(meta_readfile, "r")
    meta_w = open(meta_writefile, "w")
    for line in meta_r:
        line_new = eval(line)
        meta_w.write(line_new["asin"] + "\t" +
                     line_new["categories"][0][-1] + "\n")
    meta_r.close()
    meta_w.close()

#input: file reviews ban dau, output: reviews_information bao gom reviewerID, asin, unixReviewTime
def reviews_preprocessing(reviews_readfile, reviews_writefile):
    reviews_r = open(reviews_readfile, "r")
    reviews_w = open(reviews_writefile, "w")
    for line in reviews_r:
        line_new = eval(line)
        reviews_w.write(str(line_new["reviewerID"]) + "\t" + str(
            line_new["asin"]) + "\t" + str(line_new["unixReviewTime"]) + "\n")
    reviews_r.close()
    reviews_w.close()

#reviews_file: reviews_information, meta_file: meta_information, output_file: ns_file
def negative_sampling(reviews_file, meta_file, output_file, negative_sampling_value=1):
    f_reviews = open(reviews_file, "r") #mo file reviews_information
    user_dict = {}
    item_list = []
    for line in f_reviews:
        line = line.strip()
        reviews_things = line.split("\t") #reviews_things: revierID, asin, unixTime
        if reviews_things[0] not in user_dict: #xem co reviewerID trong user_dict k
            user_dict[reviews_things[0]] = [] #tao ra 1 dictionary voi key la reviewerID
        user_dict[reviews_things[0]].append((line, float(reviews_things[-1]))) #luu gia tri cua ca dong (line) vao key con unixTime vao value 
        item_list.append(reviews_things[1]) #them asin vao item list

    f_meta = open(meta_file, "r") #mo file meta_information
    meta_dict = {}
    for line in f_meta:
        line = line.strip()
        meta_things = line.split("\t")
        if meta_things[0] not in meta_dict: #check asin
            meta_dict[meta_things[0]] = meta_things[1] #add value=category, key la asin
    
    f_output = open(output_file, "w") #viet du lieu vao ns_file
    for user_behavior in user_dict: #user_dict: bao gom gia tri cua cac line trong reviews information va unixTime
        sorted_user_behavior = sorted(
            user_dict[user_behavior], key=lambda x: x[1]) #sap xep theo reviewerID
        for line, _ in sorted_user_behavior:
            user_things = line.split("\t") #cung la lay reviewerID, asin, unix
            asin = user_things[1]
            negative_sample = 0
            while True:
                asin_neg_index = random.randint(0, len(item_list) - 1) #tao asin index ngau nhien
                asin_neg = item_list[asin_neg_index] #lay asin dua vao index vua boc ngau nhien o tren
                if asin_neg == asin: #check xem minh co boc trung asin cua dong minh dang xet k, neu trung thi boc lai
                    continue
                user_things[1] = asin_neg #neu k thi gan asin vua boc dc vao user_things[1]
                f_output.write("0" + "\t" + "\t".join(user_things) +
                               "\t" + meta_dict[asin_neg] + "\n") #f_output la ns_data, meta_dict[asin_neg] la category
                negative_sample += 1
                if negative_sample == negative_sampling_value:
                    break
            if asin in meta_dict:
                f_output.write("1" + "\t" + line + "\t" +
                               meta_dict[asin] + "\n")
            else:
                f_output.write("1" + "\t" + line + "\t" + "default_cat" + "\n")

    f_reviews.close()
    f_meta.close()
    f_output.close()

#input_file: ns_data, output_file: pre_processed data
def data_processing(input_file, output_file, negative_sampling_value=1):
    f_input = open(input_file, "r")
    f_output = open(output_file, "w")
    user_count = {}
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1] #(ReviewerID)
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1 #dem so review cua user
    f_input.seek(0) #dem xong r thi file pointer se quay tro lai dong dau tien
    i = 0
    last_user = None
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user == last_user:
            if i < user_count[user] - 1 - negative_sampling_value:  # 1 + negative samples
                f_output.write("train" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 1 - negative_sampling_value:
                f_output.write("train" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        i += 1


if __name__ == "__main__":
    meta_readfile = "meta_Movies_and_TV.json"
    meta_writefile = "data/meta_information"
    reviews_readfile = "reviews_Movies_and_TV_5.json"
    reviews_writefile = "data/reviews_information"
    ns_file = "data/ns_data"
    output_file = "data/preprocessed_data"

    print("meta preprocessing...")
    meta_preprocessing(meta_readfile, meta_writefile) #input la "meta_Movies_and_TV.json" output la meta_information, gom asin (movie id) va category
    print("reviews preprocessing...")
    reviews_preprocessing(reviews_readfile, reviews_writefile)#input la reviews_5 output la reviewerID, asin, unixreviewtime
    print("data processing...")
    negative_sampling(reviews_writefile, meta_writefile,
                      ns_file, negative_sampling_value=1) #ns_file gom negative_sampling, reviewerID, asin, unix, category
    data_processing(ns_file, output_file)
