import numpy as np
import logging
import time
import os
import socket


def set_up_log(args, sys_argv):
    args.data = args.data.lower()
    log_dir = args.log_dir
    data_log_dir = os.path.join(log_dir, args.data)
    for _ in [log_dir, data_log_dir]:
        if not os.path.exists(_):
            os.mkdir(_)
    file_path = os.path.join(data_log_dir, '{}.log'.format(str(time.time())))
    args.file_path = file_path

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(file_path)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('Create log file at {}'.format(file_path))
    logger.info('Command line executed: python ' + ' '.join(sys_argv))
    logger.info('Full args parsed:')
    logger.info(args)
    return logger


def save_performance_result(args, logger, model):
    summary_file = os.path.join(args.log_dir, args.data, args.data+'_'+args.summary_file)
    log_name = os.path.split(logger.handlers[1].baseFilename)[-1]
    server = socket.gethostname()

    line = '\t'.join([str(args.seed),
                      args.data,
                      str(args.num_layers),
                      str(args.lr),
                      str(args.weight_decay),
                      str(model.max_acc_se),
                      str(model.max_acc_re),
                      str(model.max_epoch_se),
                      str(model.max_epoch_re),
                      # str(int(model.time_se)),
                      # str(int(model.time_re)),
                      str(np.round(model.time_se, 4)),
                      str(np.round(model.time_re, 4)),
                      time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      log_name,
                      server,
                      f'gpu_{args.gpu}']) + '\n'

    with open(summary_file, 'a') as f:
        f.write(line)  # WARNING: process unsafe!
    f.close()

    # shut_down_log to avoid writing all in the first log
    # refer to https://blog.csdn.net/weixin_41956627/article/details/125784000?utm_medium=distribute.pc_aggpage_search_result.none-task-blog-2
    logger.warning(f'Shut down {log_name}')
    logger.handlers.clear()