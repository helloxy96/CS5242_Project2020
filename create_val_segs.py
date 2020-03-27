import random


def get_labels_segs_dict(data_labels):
    '''
    get the segs list that contain each label
    :param data_labels:
    :return: {label1:[seg_idx1, seg_idx2,...], label2:[...]}
    '''

    labels_segments_dict = {}
    for i in range(len(data_labels)):
        seg_labels = data_labels[i]
        if seg_labels in labels_segments_dict:
            if i not in labels_segments_dict[seg_labels]:
                labels_segments_dict[seg_labels].append(i)
        else:
            labels_segments_dict[seg_labels] = [i]
    return labels_segments_dict


def random_choose_val_segs(labels_segments_dict, n_seg_per_label, total_val_seg_num, rand_seed):
    """
    random choose 4-5 segments for each label, totally 200 segments for all labels
    :param labels_segments_dict:
    :param n_seg_per_label:
    :param total_val_seg_num:
    :return:
    """
    chosen_seg_lists = []
    random.seed(rand_seed)
    while len(chosen_seg_lists) < total_val_seg_num:
        for label, seg_idx in labels_segments_dict.items():
            if len(seg_idx) <= n_seg_per_label:
                chosen_seg_lists.extend(seg_idx)
            else:
                chosen_seg_lists.extend(random.sample(seg_idx, n_seg_per_label))
        n_seg_per_label += 1

        chosen_seg_lists = list(set(chosen_seg_lists))
    chosen_seg_lists.sort()
    return chosen_seg_lists


def get_train_val_feat_label(origin_data_feat, origin_data_labels, train_or_val='training', n_seg_per_label=5, total_val_seg_num=200, rand_seed=1):
    labels_segments_dict = get_labels_segs_dict(origin_data_labels)
    val_seg_lists = random_choose_val_segs(labels_segments_dict, n_seg_per_label, total_val_seg_num, rand_seed)

    if train_or_val == 'training':
        train_seg_lists = [idx for idx in range(len(origin_data_labels)) if idx not in val_seg_lists]
        train_data_feat = []
        train_data_labels = []
        for idx in train_seg_lists:
            train_data_feat.append(origin_data_feat[idx])
            train_data_labels.append(origin_data_labels[idx])
        return train_data_feat, train_data_labels

    elif train_or_val == 'validation':
        val_data_feat = []
        val_data_labels = []
        for idx in val_seg_lists:
            val_data_feat.append(origin_data_feat[idx])
            val_data_labels.append(origin_data_labels[idx])
        return val_data_feat, val_data_labels