import random

def generate_label_dictionary(data_feat, data_labels):
    label_dict = {}
    for i in range(len(data_labels)):
        if data_labels[i] not in label_dict.keys():
            label_dict[data_labels[i]] = [data_feat[i]]
        else:
            label_dict[data_labels[i]].append(data_feat[i])

    return label_dict

def balance_data(label_dict, data_len):
    class_num = len(label_dict.keys())
    avg_len = int(data_len / class_num)

    new_data_feat = []
    new_data_labels = []
    for k in label_dict.keys():
        if len(label_dict[k]) < avg_len:
            multiplier = int(avg_len / len(label_dict[k])) + 1
            label_dict[k] = label_dict[k] * multiplier
        random.shuffle(label_dict[k])
        new_data_feat.extend(label_dict[k][:avg_len])
        new_data_labels.extend([k] * avg_len)
    return new_data_feat, new_data_labels

