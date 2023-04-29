import os

import torch


def save_model(logger, model_path, experiment_name, epoch, val_loss, model, optimizer, arguments, states, best_model=True,
               clear_history=True, data_parallel=False):
    if best_model:
        model_path = os.path.join(model_path, experiment_name, 'best_model')
    else:
        model_path = os.path.join(model_path, experiment_name, 'check_point')
    if not os.path.isdir(model_path):
        logger.info('[CHECKPOINT] Creating model file:%s' % model_path)
        os.makedirs(model_path)
    model_name = '%s/model_%d_%d.pt' % (model_path, epoch, int(val_loss * 100))

    arguments_dict = {}
    if isinstance(arguments, dict) is False:
        arguments = vars(arguments)
        for key, value in arguments.items():
            arguments_dict[key] = value
    else:
        arguments_dict = arguments

    model_state = {
        'arguments': arguments_dict,
        'model': model.state_dict() if not data_parallel else model.module.state_dict(),
        'optimizer': optimizer.state_dict(),
        'states': states
    }
    torch.save(model_state, model_name)

    logger.info('[CHECKPOINT] Model has been saved to :%s' % model_name)
    if clear_history:
        logger.info('[CHECKPOINT] Removing old checkpoints')
        files = os.listdir(model_path)
        for file in files:
            file_name = '%s/%s' % (model_path, file)
            if file_name != model_name:
                logger.info('[CHECKPOINT] Removing %s' % file_name)
                try:
                    os.remove(file_name)
                except Exception as e:
                    print(e)