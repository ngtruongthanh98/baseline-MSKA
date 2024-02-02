# *torch
from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from torch.utils.data import DataLoader

from Tokenizer import GlossTokenizer_S2G
from model import SignLanguageModel
# *user-defined

import utils as utils
from datasets import S2T_Dataset

# *basic
import os
import time
import argparse, json, datetime
import numpy as np
from collections import defaultdict
import yaml
import random
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger

# *metric
from metrics import wer_list, bleu, rouge
import torch.distributed as dist
from timm.utils import NativeScaler

# global definition
from definition import *
from optimizer import build_optimizer, build_scheduler
from phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014


def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=40, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)
    # parser.add_argument('--gpu', default=[0], type=int)
    # parser.add_argument('--distributed', default=True, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=[0.9, 0.998], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * Baise params
    parser.add_argument('--output_dir', default='out/vlp_v2',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/SingleStream/phoenix-2014t_s2g.yaml')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)

    # * wandb params
    parser.add_argument("--log_all", action="store_true",
                        help="flag to log in all processes, otherwise only in rank0",
                        )
    parser.add_argument("--entity", type=str,
                        help="wandb entity",
                        )
    parser.add_argument("--project", type=str, default='VLP',
                        help="wandb project",
                        )

    # * Noise params
    parser.add_argument('--training-refurbish', default=True, type=bool)
    parser.add_argument('--noise-rate', default=0.15, type=float)
    parser.add_argument('--noise-type', default='omit_last', type=str, choices=['omit', 'omit_last'])
    parser.add_argument('--random-shuffle', default=False, type=bool)

    parser.add_argument('--loss-lambda', type=float, default=1.0, metavar='RATE',
                        help='lambda param')

    return parser


def init_ddp(local_rank):
    # 有了这一句之后，在转换device的时候直接使用 a=a.cuda()即可，否则要用a=a.cuda(local+rank)
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')


def main(args, config):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False

    print(f"Creating dataset:")
    tokenizer = GlossTokenizer_S2G(config['gloss'])
    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer=tokenizer, config=config, args=args,
                             phase='train', training_refurbish=True)
    print(train_data)
    # torch.distributed.init_process_group(backend='nccl', init_method='tcp://127.0.0.11:3806', rank=0, world_size=1)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data, shuffle=True)
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  shuffle=True,
                                  # sampler=train_sampler,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)

    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer=tokenizer, config=config, args=args,
                           phase='val', training_refurbish=True)
    print(dev_data)
    # dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data,shuffle=False)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                # sampler=dev_sampler,
                                pin_memory=args.pin_mem)

    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer=tokenizer, config=config, args=args,
                            phase='test', training_refurbish=True)
    print(test_data)
    # test_sampler = torch.utils.data.distributed.DistributedSampler(test_data, shuffle=False)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 # sampler=test_sampler,
                                 pin_memory=args.pin_mem)

    print(f"Creating model:")
    model = SignLanguageModel(cfg=config, args=args)
    model.to(device)
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret = model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    n_parameters = utils.count_parameters_in_MB(model)
    print(f'number of params: {n_parameters}M')
    optimizer = build_optimizer(config=config['training']['optimization'], model=model)
    scheduler, scheduler_type = build_scheduler(config=config['training']['optimization'], optimizer=optimizer)
    loss_scaler = GradScaler()
    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, tokenizer, config, args.start_epoch, UNK_IDX, SPECIAL_SYMBOLS,
                             PAD_IDX, device)
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, tokenizer, config, args.start_epoch, UNK_IDX,
                              SPECIAL_SYMBOLS, PAD_IDX, device)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 200
    bleu_4 = 0
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        train_stats = train_one_epoch(args, config, model, tokenizer, train_dataloader, optimizer, device, epoch,
                                      config, PAD_IDX, loss_scaler)
        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        test_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch, beam_size=1,
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])

        if config['task'] == "S2T":
            if bleu_4 < test_stats["bleu4"]:
                bleu_4 = test_stats["bleu4"]
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            # 'args': args,
                        }, checkpoint_path)

            print(f"* DEV BLEU-4 {test_stats['bleu-4']:.3f} Max DEV BLEU-4 {bleu_4}")
        else:
            if min_loss > test_stats["wer"]:
                min_loss = test_stats["wer"]
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch,
                            # 'args': args,
                        }, checkpoint_path)
            print(f"* DEV wer {test_stats['wer']:.3f} Min DEV WER {min_loss}")
        if args.run:
            args.run.log(
                {'epoch': epoch + 1, 'training/train_loss': train_stats['loss'], 'dev/dev_loss': test_stats['loss'],
                 'dev/min_loss': min_loss})

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir:
        checkpoint = torch.load(args.output_dir + '/best_checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        dev_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch, beam_size=5,
                             do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")
        test_stats = evaluate(args, config, test_dataloader, model, tokenizer, epoch, beam_size=5,
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        if config['do_translation']:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({'Dev WER:': dev_stats['wer'],
                                    'Test WER:': test_stats['wer']}) + "\n")
        if config['do_translation']:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({'Dev WER:': dev_stats['wer'],
                                    'Test WER:': test_stats['bleu-4']}) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(args, config, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, PAD_IDX, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    for step, (src_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        # w#ith torch.cuda.amp.autocast():
        output = model(src_input)
        with torch.autograd.set_detect_anomaly(True):
            output['total_loss'].backward()
        optimizer.step()
        model.zero_grad()
        # 调用 GradScaler 的 `step` 方法来执行优化器的更新，并清除梯度
        loss_value = output['total_loss'].item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    if args.run:
        args.run.log({'epoch': epoch + 1, 'epoch/train_loss': loss_value})
    # gather the stats from all processes
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate(args, config, dev_dataloader, model, tokenizer, epoch, beam_size=1, do_translation=True,
             do_recognition=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    results = defaultdict(dict)

    with torch.no_grad():
        for step, (src_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):
            output = model(src_input)
            if do_recognition:
                for k, gls_logits in output.items():
                    if not 'gloss_logits' in k:
                        continue
                    logits_name = k.replace('gloss_logits', '')
                    ctc_decode_output = model.recognition_network.decode(gloss_logits=gls_logits,
                                                                         beam_size=beam_size,
                                                                         input_lengths=output['input_lengths'])
                    batch_pred_gls = tokenizer.convert_ids_to_tokens(ctc_decode_output)
                    for name, gls_hyp, gls_ref in zip(src_input['name'], batch_pred_gls, src_input['gloss']):
                        results[name][f'{logits_name}gls_hyp'] = \
                            ' '.join(gls_hyp).upper() if tokenizer.lower_case \
                                else ' '.join(gls_hyp)
                        results[name]['gls_ref'] = gls_ref.upper() if tokenizer.lower_case \
                            else gls_ref
            if do_translation:
                generate_output = model.generate_txt(
                    transformer_inputs=output['transformer_inputs'],
                    generate_cfg={'length_penalty': 1, 'max_length': 100, 'num_beams': 5})
                # decoded_sequences
                for name, txt_hyp, txt_ref in zip(src_input['name'], generate_output['decoded_sequences'],
                                                  src_input['text']):
                    results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref
            metric_logger.update(loss=output['total_loss'].item())
        if do_recognition:
            evaluation_results = {}
            evaluation_results['wer'] = 200
            for hyp_name in results[name].keys():
                if not 'gls_hyp' in hyp_name:
                    continue
                k = hyp_name.replace('gls_hyp', '')
                if config['data']['dataset_name'].lower() == 'phoenix-2014t':
                    gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results]
                elif config['data']['dataset_name'].lower() == 'phoenix-2014':
                    gls_ref = [clean_phoenix_2014(results[n]['gls_ref']) for n in results]
                    gls_hyp = [clean_phoenix_2014(results[n][hyp_name]) for n in results]
                wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
                evaluation_results[k + 'wer_list'] = wer_results
                evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
            metric_logger.update(wer=evaluation_results['wer'])

        if do_translation:
            txt_ref = [results[n]['txt_ref'] for n in results]
            txt_hyp = [results[n]['txt_hyp'] for n in results]
            bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level='word')
            rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level='word')
            for k, v in bleu_dict.items():
                print('{} {:.2f}'.format(k, v))
            print('ROUGE: {:.2f}'.format(rouge_score))
            evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
            wandb.log({'eval/BLEU4': bleu_dict['bleu4']})
            wandb.log({'eval/ROUGE': rouge_score})
            metric_logger.update(bleu1=bleu_dict['bleu1'])
            metric_logger.update(bleu2=bleu_dict['bleu2'])
            metric_logger.update(bleu3=bleu_dict['bleu3'])
            metric_logger.update(bleu4=bleu_dict['bleu4'])
            metric_logger.update(rouge=rouge_score)

    if args.run:
        args.run.log(
            {'epoch': epoch + 1, 'epoch/dev_loss': output['recognition_loss'].item(), 'wer': evaluation_results['wer']})
    print("* Averaged stats:", metric_logger)
    print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False
    return run


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    args = parser.parse_args()
    with open(args.config, 'r+', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)