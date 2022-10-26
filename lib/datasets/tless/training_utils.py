import os, time
import torch
from lib.utils.metrics import AverageValueMeter


def train(train_data, model, optimizer, warm_up_config, epoch, logger, tb_logger, log_interval, is_master):
    start_time = time.time()
    meter_positive_similarity = AverageValueMeter()
    meter_negative_similarity = AverageValueMeter()
    meter_train_loss = AverageValueMeter()

    timing_text = "Training time for epoch {}: {:.02f} minutes"
    monitoring_text = 'Epoch-{} -- Iter [{}/{}] loss: {:.2f}, (pos: {:.2f}, neg: {:.2f})'

    model.train()
    train_size, train_loader = len(train_data), iter(train_data)
    with torch.autograd.set_detect_anomaly(True):
        for i in range(train_size):
            # update learning rate with warm up
            if warm_up_config is not None:
                [nb_iter_warm_up, lr] = warm_up_config
                nb_iter = epoch * train_size + i
                if nb_iter <= nb_iter_warm_up:
                    lrUpdate = nb_iter / float(nb_iter_warm_up) * lr
                    for g in optimizer.param_groups:
                        g['lr'] = lrUpdate

            # load data and label
            miniBatch = train_loader.next()
            query = miniBatch["query"].cuda()
            template = miniBatch["template"].cuda()
            mask = miniBatch["mask"].cuda().float()

            feature_query = model(query)
            feature_template = model(template)
            pos_sim = model.calculate_similarity(feature_query, feature_template, mask)  # B x 1
            # from dataloader but taking others img in current batch
            neg_sim = model.calculate_similarity_for_search(feature_query, feature_template, mask)  # B x B
            avg_pos_sim, avg_neg_sim, loss = model.calculate_global_loss(positive_pair=pos_sim,
                                                                         negative_pair=neg_sim)

            meter_positive_similarity.update(avg_pos_sim.item())
            meter_negative_similarity.update(avg_neg_sim.item())
            meter_train_loss.update(loss.item())

            # back prop
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            # monitoring
            if ((i + 1) % log_interval == 0 or i == 0) and is_master:
                if warm_up_config is not None and nb_iter <= nb_iter_warm_up:
                    text = "Learning rate: {}".format(lrUpdate)
                    logger.info(text)
                filled_monitoring_text = monitoring_text.format(epoch, i + 1, train_size,
                                                                meter_train_loss.val,
                                                                meter_positive_similarity.val,
                                                                meter_negative_similarity.val)
                logger.info(filled_monitoring_text)
    logger.info(timing_text.format(epoch, (time.time() - start_time) / 60))
    if is_master:
        tb_logger.add_scalar_dict_list('loss', [{'train_loss': meter_train_loss.avg,
                                                 'positive similarity': meter_positive_similarity.avg,
                                                 'negative similarity': meter_negative_similarity.avg}], epoch)
    return meter_train_loss.avg
