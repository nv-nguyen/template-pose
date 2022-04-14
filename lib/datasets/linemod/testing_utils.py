import os, time
import torch
import torch.nn.functional as F
from lib.utils.metrics import AverageValueMeter
from lib.datasets.linemod import inout


def calculate_score(pred_location, gt_location, id_symmetry, id_obj, pred_id_obj):
    unique_ids, inverse_indices = torch.unique(id_obj, sorted=True, return_inverse=True)
    cosine_sim = F.cosine_similarity(pred_location, gt_location)
    angle_err = torch.rad2deg(torch.arccos(cosine_sim.clamp(min=-1, max=1)))

    # for symmetry
    gt_location_opposite = gt_location
    gt_location_opposite[:, :2] *= -1  # rotation 180 in Z axis
    cosine_sim_sym = F.cosine_similarity(gt_location_opposite, gt_location_opposite)
    angle_err_sym = torch.rad2deg(torch.arccos(cosine_sim_sym.clamp(min=-1, max=1)))
    angle_err[id_symmetry == 1] = torch.minimum(angle_err[id_symmetry == 1], angle_err_sym[id_symmetry == 1])

    list_err, list_pose_acc, list_class_acc, list_class_and_pose_acc15 = {}, {}, {}, {}
    for i in range(len(unique_ids)):
        err = angle_err[id_obj == unique_ids[i]]
        recognition_acc = (pred_id_obj[id_obj == unique_ids[i]] == unique_ids[i])

        class_and_pose_acc15 = torch.logical_and(err <= 15, recognition_acc).float().mean()
        err = err.mean()
        recognition_acc = recognition_acc.float().mean()
        pose_acc = (err <= 15).float().mean()

        list_err[unique_ids[i].item()] = err
        list_pose_acc[unique_ids[i].item()] = pose_acc
        list_class_acc[unique_ids[i].item()] = recognition_acc
        list_class_and_pose_acc15[unique_ids[i].item()] = class_and_pose_acc15

    list_err["mean"] = torch.mean(angle_err)
    list_pose_acc["mean"] = (angle_err <= 15).float().mean()
    list_class_acc["mean"] = (pred_id_obj == id_obj).float().mean()
    list_class_and_pose_acc15["mean"] = torch.logical_and(angle_err <= 15, pred_id_obj == id_obj).float().mean()

    return list_err, list_pose_acc, list_class_acc, list_class_and_pose_acc15


def test(query_data, template_data, model, epoch, logger, tb_logger, split_name, list_id_obj, is_master):
    print(split_name)
    start_time = time.time()
    list_id_obj.append("mean")
    meter_error = {id_obj: AverageValueMeter() for id_obj in list_id_obj}
    meter_accuracy = {id_obj: AverageValueMeter() for id_obj in list_id_obj}
    meter_recognition = {id_obj: AverageValueMeter() for id_obj in list_id_obj}
    meter_accuracy_class_and_pose = {id_obj: AverageValueMeter() for id_obj in list_id_obj}

    query_size, query_dataloader = len(query_data), iter(query_data)
    template_size, template_dataloader = len(template_data), iter(template_data)

    monitoring_text = "Epoch-{}, {} -- Mean err: {:.2f}, Acc: {:.2f}, Rec : {:.2f}, Class and Pose  : {:.2f}"
    timing_text = "Validation time for epoch {}: {:.02f} minutes"

    model.eval()
    with torch.no_grad():
        list_feature_template, list_synthetic_pose, list_id_obj_template, list_mask = [], [], [], []
        for i in range(template_size):
            # read all templates and its poses
            miniBatch = template_dataloader.next()

            template = miniBatch["template"].cuda()
            obj_pose = miniBatch["obj_pose"].cuda()
            id_obj = miniBatch["id_obj"].cuda()
            mask = miniBatch["mask"].cuda().float()
            feature_template = model(template)

            list_synthetic_pose.append(obj_pose)
            list_id_obj_template.append(id_obj)
            list_mask.append(mask)
            list_feature_template.append(feature_template)

        list_feature_template = torch.cat(list_feature_template, dim=0)
        list_synthetic_pose = torch.cat(list_synthetic_pose, dim=0)
        list_id_obj_template = torch.cat(list_id_obj_template, dim=0)
        list_mask = torch.cat(list_mask, dim=0)

        for i in range(query_size):
            miniBatch = query_dataloader.next()

            query = miniBatch["query"].cuda()
            obj_pose = miniBatch["obj_pose"].cuda()
            id_obj = miniBatch["id_obj"].cuda()
            id_symmetry = miniBatch["id_symmetry"].cuda()
            feature_query = model(query)

            # get best template
            matrix_sim = model.calculate_similarity_for_search(feature_query, list_feature_template, list_mask,
                                                               training=False)
            weight_sim, pred_index = matrix_sim.topk(k=1)
            pred_pose = list_synthetic_pose[pred_index.reshape(-1)]
            pred_id_obj = list_id_obj_template[pred_index.reshape(-1)]

            err, acc, class_score, class_and_pose = calculate_score(pred_location=pred_pose,
                                                                    gt_location=obj_pose,
                                                                    id_symmetry=id_symmetry,
                                                                    id_obj=id_obj,
                                                                    pred_id_obj=pred_id_obj)
            for key in err.keys():
                meter_error[key].update(err[key].item())
                meter_accuracy[key].update(acc[key].item())
                meter_recognition[key].update(class_score[key].item())
                meter_accuracy_class_and_pose[key].update(class_and_pose[key].item())

        scores = [meter_error, meter_accuracy, meter_recognition, meter_accuracy_class_and_pose]
        if is_master:
            results = {}
            for idx_metric, metric_name in enumerate(["error", "accuracy", "recognition", "recognition and pose"]):
                for id_obj in list_id_obj:
                    if id_obj == "mean":
                        obj_name = "mean"
                    else:
                        obj_name = inout.LINEMOD_real_id_to_name[id_obj]
                    key_name = "{}, {}, {}".format(split_name, metric_name, obj_name)
                    results[key_name] = scores[idx_metric][id_obj].avg
        tb_logger.add_scalar_dict_list("results per object", [results], epoch)
        filled_monitoring_text = monitoring_text.format(epoch, split_name,
                                                        meter_error["mean"].avg,
                                                        meter_accuracy["mean"].avg,
                                                        meter_recognition["mean"].avg,
                                                        meter_accuracy_class_and_pose["mean"].avg)
        logger.info(filled_monitoring_text)
        logger.info(timing_text.format(epoch, (time.time() - start_time) / 60))
    return [meter_error["mean"].avg, meter_accuracy["mean"].avg, meter_recognition["mean"].avg,
            meter_accuracy_class_and_pose["mean"].avg]
