import numpy as np
import logging

# two objects in LM are not used such as object 2 (bow)
query_name_to_real_id = {
    "ape": 0,
    "benchvise": 1,
    "cam": 3,
    "can": 4,
    "cat": 5,
    "driller": 7,
    "duck": 8,
    "eggbox": 9,
    "glue": 10,
    "holepuncher": 11,
    "iron": 12,
    "lamp": 13,
    "phone": 14,
}
query_names = np.asarray([k for k in query_name_to_real_id.keys()])
query_real_ids = np.asarray([v for v in query_name_to_real_id.values()])

occlusion_query_name_to_real_id = {
    "ape": 0,
    "can": 4,
    "cat": 5,
    "driller": 7,
    "duck": 8,
    "eggbox": 9,
    "glue": 10,
    "holepuncher": 11,
}
occlusion_query_names = np.asarray([k for k in occlusion_query_name_to_real_id.keys()])
occlusion_real_ids = np.asarray([v for v in occlusion_query_name_to_real_id.values()])


def get_list_id_obj_from_split_name(split_name):
    assert split_name in ["split1", "split2", "split3"], print(
        "Split_name is not correct!!!"
    )
    list_id_obj = query_real_ids
    if split_name == "split1":
        seen_id_obj, seen_names = list_id_obj[4:], query_names[4:]
        seen_occ_id_obj, seen_occ_names = (
            occlusion_real_ids[2:],
            occlusion_query_names[2:],
        )
        unseen_id_obj, unseen_names = list_id_obj[:4], query_names[:4]
        unseen_occ_id_obj, unseen_occ_names = (
            occlusion_real_ids[:2],
            occlusion_query_names[:2],
        )
    elif split_name == "split2":
        seen_id_obj, seen_names = np.concatenate(
            (list_id_obj[:4], list_id_obj[8:])
        ), np.concatenate((query_names[:4], query_names[8:]))
        seen_occ_id_obj, seen_occ_names = np.concatenate(
            (occlusion_real_ids[:2], occlusion_real_ids[6:])
        ), np.concatenate((occlusion_query_names[:2], occlusion_query_names[6:]))
        unseen_id_obj, unseen_names = list_id_obj[4:8], query_names[4:8]
        unseen_occ_id_obj, unseen_occ_names = (
            occlusion_real_ids[2:6],
            occlusion_query_names[2:6],
        )
    elif split_name == "split3":
        seen_id_obj, seen_names = list_id_obj[:8], query_names[:8]
        seen_occ_id_obj, seen_occ_names = (
            occlusion_real_ids[:6],
            occlusion_query_names[:6],
        )
        unseen_id_obj, unseen_names = list_id_obj[8:], query_names[8:]
        unseen_occ_id_obj, unseen_occ_names = (
            occlusion_real_ids[6:],
            occlusion_query_names[6:],
        )
    logging.info(
        f"Seen: {seen_names}, Occluded Seen: {seen_occ_names}, Unseen: {unseen_names}, Occluded Unseen: {unseen_occ_names}"
    )
    return (
        seen_id_obj.tolist(),
        seen_names,
        seen_occ_id_obj.tolist(),
        seen_occ_names,
        unseen_id_obj.tolist(),
        unseen_names,
        unseen_occ_id_obj.tolist(),
        unseen_occ_names,
    )
