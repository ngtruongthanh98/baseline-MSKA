import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from Tokenizer import GlossTokenizer_S2G
from model import SignLanguageModel
import utils as utils
from datasets import S2T_Dataset
import os
import time
import argparse, json, datetime
import numpy as np
from collections import defaultdict
import yaml
import random
import wandb
from pathlib import Path
import math
import sys
from typing import Iterable
from loguru import logger
import gc
import re
import ast

# *metric
from metrics import wer_list, bleu, rouge
import torch.distributed as dist

# global definition
from optimizer import build_optimizer, build_scheduler
from phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014

# gTTS
from gtts import gTTS
from IPython.display import Audio

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=2, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)

    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')
    # * Baise params
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='configs/csl-daily_s2g.yaml')

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

    return parser


def init_ddp(local_rank):
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
    train_dataloader = DataLoader(train_data,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  collate_fn=train_data.collate_fn,
                                  shuffle=True,
                                  pin_memory=args.pin_mem,
                                  drop_last=True)

    dev_data = S2T_Dataset(path=config['data']['dev_label_path'], tokenizer=tokenizer, config=config, args=args,
                           phase='val', training_refurbish=True)
    print(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=dev_data.collate_fn,
                                pin_memory=args.pin_mem)

    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer=tokenizer, config=config, args=args,
                            phase='test', training_refurbish=True)
    print(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 collate_fn=test_data.collate_fn,
                                 pin_memory=args.pin_mem)

    # get keypoint data from ../01April_2010_Thursday_heute-6697/src_input.txt
    # keypoints_data get from the above file

    file_path = '../01April_2010_Thursday_heute-6697/src_input.txt'
    src_input = load_keypoints_data(file_path)

    if src_input is not None:
        # Process the keypoints_data
        print('Keypoints data loaded successfully.')
    else:
        print('Failed to load keypoints data.')


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
    output_dir = Path(config['training']['model_dir'])
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
        # dev_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch=0, beam_size=5,
        #                       generate_cfg=config['training']['validation']['translation'],
        #                       do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        # print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        # test_stats = evaluate(args, config, test_dataloader, model, tokenizer, epoch=0, beam_size=5,
        #                       generate_cfg=config['testing']['translation'],
        #                       do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        # print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")


        custom_starts = evaluate_one_item(args, config, src_input, model, tokenizer, epoch=0, beam_size=5,
                              generate_cfg=config['testing']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {custom_starts['loss']:.3f}")

        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_loss = 200
    bleu_4 = 0
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()
        train_stats = train_one_epoch(args, model, tokenizer, train_dataloader, optimizer, device, epoch)
        checkpoint_paths = [output_dir / f'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
            }, checkpoint_path)
        # test_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch,
        #                       beam_size=config['training']['validation']['recognition']['beam_size'],
        #                       generate_cfg=config['training']['validation']['translation'],
        #                       do_translation=config['do_translation'], do_recognition=config['do_recognition'])

        custom_starts = evaluate_one_item(args, config, src_input, model, tokenizer, epoch,
                              beam_size=config['training']['validation']['recognition']['beam_size'],
                              generate_cfg=config['training']['validation']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])

        if config['task'] == "S2T":
            if bleu_4 < test_stats["bleu4"]:
                bleu_4 = test_stats["bleu4"]
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)

            print(f"* DEV BLEU-4 {test_stats['bleu4']:.3f} Max DEV BLEU-4 {bleu_4}")
        else:
            if min_loss > test_stats["wer"]:
                min_loss = test_stats["wer"]
                checkpoint_paths = [output_dir / 'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
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

        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

        # Last epoch
    test_on_last_epoch = True
    if test_on_last_epoch:
        checkpoint = torch.load(str(output_dir) + '/best_checkpoint.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=True)
        # dev_stats = evaluate(args, config, dev_dataloader, model, tokenizer, epoch=0, beam_size=config['testing']['recognition']['beam_size'],
        #                      generate_cfg=config['training']['validation']['translation'],
        #                      do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        # print(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        # test_stats = evaluate(args, config, test_dataloader, model, tokenizer, epoch=0, beam_size=config['testing']['recognition']['beam_size'],
        #                       generate_cfg=config['testing']['translation'],
        #                       do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        # print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")

        custom_starts = evaluate_one_item(args, config, src_input, model, tokenizer, epoch=0, beam_size=config['testing']['recognition']['beam_size'],
                              generate_cfg=config['testing']['translation'],
                              do_translation=config['do_translation'], do_recognition=config['do_recognition'])
        print(f"Test loss of the network on the {len(test_dataloader)} custom videos: {custom_starts['loss']:.3f}")

        if config['do_recognition']:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({'Dev WER:': dev_stats['wer'],
                                    'Test WER:': test_stats['wer']}) + "\n")
        if config['do_translation']:
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps({'Dev Bleu-4:': dev_stats['bleu4'],
                                    'Test Bleu-4:': test_stats['bleu4']}) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

# def load_keypoints_data(file_path):
#     if not os.path.exists(file_path):
#         print(f'File not found: {file_path}')
#         return None

#     keypoints_data = torch.load(file_path)
#     return keypoints_data

def load_keypoints_data(file_path):
    if not os.path.exists(file_path):
        print(f'File not found: {file_path}')
        return None

    try:
        # Try to load as a PyTorch file
        keypoints_data = torch.load(file_path)
    except (torch.serialization.pickle.UnpicklingError, AttributeError, EOFError):
        try:
            # Try to load as a JSON file
            with open(file_path, 'r') as f:
                keypoints_data = json.load(f)
        except json.JSONDecodeError:
            try:
                # Try to load as a plain text file
                with open(file_path, 'r') as f:
                    keypoints_data = f.read()
            except Exception as e:
                print(f'Error loading file: {e}')
                return None

    return keypoints_data

# def load_keypoints_data(file_path):
#     if not os.path.exists(file_path):
#         print(f'File not found: {file_path}')
#         return None

#     try:
#         # Try to load as a PyTorch file
#         keypoints_data = torch.load(file_path)
#     except (torch.serialization.pickle.UnpicklingError, AttributeError, EOFError):
#         try:
#             # Try to load as a JSON file
#             with open(file_path, 'r') as f:
#                 keypoints_data = json.load(f)
#         except json.JSONDecodeError:
#             try:
#                 # Try to load as a plain text file
#                 keypoints_data = {}
#                 with open(file_path, 'r') as f:
#                     content = f.read()
#                     try:
#                         # Safely evaluate the entire content to handle complex data structures
#                         keypoints_data = ast.literal_eval(content)
#                     except (ValueError, SyntaxError):
#                         print(f"Error evaluating content: {content}")
#                         return None
#             except Exception as e:
#                 print(f'Error loading file: {e}')
#                 return None

#     return keypoints_data


def train_one_epoch(args, model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    for step, (src_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        optimizer.zero_grad()
        output = model(src_input)
        with torch.autograd.set_detect_anomaly(True):
            output['total_loss'].backward()
        optimizer.step()
        model.zero_grad()
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


def evaluate(args, config, dev_dataloader, model, tokenizer, epoch, beam_size=1, generate_cfg={}, do_translation=True,
             do_recognition=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    results = defaultdict(dict)

    with torch.no_grad():
        for step, (src_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):
            print('test src_input: ', src_input)

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

            result_dir = f'../result'
            os.makedirs(result_dir, exist_ok=True)

            if do_translation:
                last_result = []

                generate_output = model.generate_txt(
                    transformer_inputs=output['transformer_inputs'],
                    generate_cfg=generate_cfg)

                for idx, (name, txt_hyp, txt_ref) in enumerate(zip(src_input['name'], generate_output['decoded_sequences'], src_input['text']), start=1):
                    print('name: ', name)
                    results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref

                    match = re.match(r'^(test|dev)/(.+)$', name)
                    if match:
                        prefix, rest_of_name = match.groups()
                        temp_name = rest_of_name.replace("/", "-")
                        sub_dir = os.path.join(result_dir, prefix)
                    else:
                        temp_name = name.replace("/", "-")
                        sub_dir = result_dir

                    print('txt_hyp: ', txt_hyp)

                    # Create directory for the sample inside the result directory
                    sample_dir = os.path.join(sub_dir, f'{temp_name}')
                    os.makedirs(sample_dir, exist_ok=True)

                    # Save txt_hyp as an mp3 file
                    tts_hyp = gTTS(text=txt_hyp, lang='de')
                    hyp_path = os.path.join(sample_dir, f'txt_hyp_{temp_name}.mp3')
                    tts_hyp.save(hyp_path)

                    # Save txt_ref as an mp3 file
                    ref_path = os.path.join(sample_dir, f'txt_ref_{temp_name}.mp3')
                    tts_ref = gTTS(text=txt_ref, lang='de')
                    tts_ref.save(ref_path)

                    # save src_input as a txt file
                    src_input_path = os.path.join(sample_dir, f'src_input.txt')
                    with open(src_input_path, 'w') as src_input_file:
                        src_input_file.write(f"{src_input}")

                    # Create a text file to store txt_hyp and txt_ref
                    text_file_path = os.path.join(sample_dir, f'{temp_name}.txt')
                    with open(text_file_path, 'w') as text_file:
                        text_file.write(f"txt_hyp: {txt_hyp}\n")
                        text_file.write(f"txt_ref: {txt_ref}\n")

                    # Optionally, play the audio (comment out if not needed)
                    # Audio(hyp_path, autoplay=True)
                    # Audio(ref_path, autoplay=True)

                    print('txt_ref: ', txt_ref)

                    print(
                        f'Name: {name}' + '\n' +
                        f'txt_hyp: {txt_hyp}' + '\n' +
                        f'txt_ref: {txt_ref}' + '\n' +
                        f'type: {prefix}'
                    )

                    last_result.append(
                        {
                            'name': name,
                            'txt_hyp': txt_hyp,
                            'txt_ref': txt_ref,
                            'prefix': prefix,
                        }
                    )

                    # Clear variables and call garbage collection
                    del tts_hyp, tts_ref
                    gc.collect()

                print('last_result: ', last_result)

                os.makedirs('../result/json', exist_ok=True)

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                # store data to json file
                with open(f'../result/json/last_result_{timestamp}.json', 'w') as f:
                    json.dump(last_result, f, indent=4)

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
                elif config['data']['dataset_name'].lower() == 'csl-daily':
                    gls_ref = [results[n]['gls_ref'] for n in results]
                    gls_hyp = [results[n][hyp_name] for n in results]
                wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
                evaluation_results[k + 'wer_list'] = wer_results
                evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
            metric_logger.update(wer=evaluation_results['wer'])

        if do_translation:
            txt_ref = [results[n]['txt_ref'] for n in results]
            txt_hyp = [results[n]['txt_hyp'] for n in results]
            bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
            rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
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

def evaluate_one_item(args, config, src_input, model, tokenizer, epoch, beam_size=1, generate_cfg={}, do_translation=True, do_recognition=True):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    results = defaultdict(dict)

    with torch.no_grad():
        print('test src_input: ', src_input)
        # print type of src_input
        print('type of src_input: ', type(src_input))

        # Check if src_input is a string and convert it to a dictionary
        if isinstance(src_input, str):
            try:
                # Define the context for eval to include the tensor function
                context = {'tensor': torch.tensor}
                src_input = eval(src_input, context)
            except (ValueError, SyntaxError, NameError) as e:
                print(f"Error converting src_input to dict: {e}")
                return {"loss": float('inf')}  # Return a default value to avoid NoneType errors

        print('type of src_input after: ', type(src_input))

        output = model(src_input)

        print('output: ', output)

    return {"loss": 0.0}  # Return a default value to avoid NoneType errors


    #     if do_recognition:
    #         for k, gls_logits in output.items():
    #             if not 'gloss_logits' in k:
    #                 continue
    #             logits_name = k.replace('gloss_logits', '')
    #             ctc_decode_output = model.recognition_network.decode(gloss_logits=gls_logits,
    #                                                                  beam_size=beam_size,
    #                                                                  input_lengths=output['input_lengths'])
    #             batch_pred_gls = tokenizer.convert_ids_to_tokens(ctc_decode_output)
    #             for name, gls_hyp, gls_ref in zip(src_input['name'], batch_pred_gls, src_input['gloss']):
    #                 results[name][f'{logits_name}gls_hyp'] = \
    #                     ' '.join(gls_hyp).upper() if tokenizer.lower_case \
    #                         else ' '.join(gls_hyp)
    #                 results[name]['gls_ref'] = gls_ref.upper() if tokenizer.lower_case \
    #                     else gls_ref

    #     result_dir = f'../result-one-item'
    #     os.makedirs(result_dir, exist_ok=True)

    #     if do_translation:
    #         last_result = []

    #         generate_output = model.generate_txt(
    #             transformer_inputs=output['transformer_inputs'],
    #             generate_cfg=generate_cfg)

    #         for idx, (name, txt_hyp, txt_ref) in enumerate(zip(src_input['name'], generate_output['decoded_sequences'], src_input['text']), start=1):
    #             print('name: ', name)
    #             results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref

    #             match = re.match(r'^(test|dev)/(.+)$', name)
    #             if match:
    #                 prefix, rest_of_name = match.groups()
    #                 temp_name = rest_of_name.replace("/", "-")
    #                 sub_dir = os.path.join(result_dir, prefix)
    #             else:
    #                 temp_name = name.replace("/", "-")
    #                 sub_dir = result_dir

    #             print('txt_hyp: ', txt_hyp)

    #             # Create directory for the sample inside the result directory
    #             sample_dir = os.path.join(sub_dir, f'{temp_name}')
    #             os.makedirs(sample_dir, exist_ok=True)

    #             # Save txt_hyp as an mp3 file
    #             tts_hyp = gTTS(text=txt_hyp, lang='de')
    #             hyp_path = os.path.join(sample_dir, f'txt_hyp_{temp_name}.mp3')
    #             tts_hyp.save(hyp_path)

    #             # Save txt_ref as an mp3 file
    #             ref_path = os.path.join(sample_dir, f'txt_ref_{temp_name}.mp3')
    #             tts_ref = gTTS(text=txt_ref, lang='de')
    #             tts_ref.save(ref_path)

    #             # save src_input as a txt file
    #             src_input_path = os.path.join(sample_dir, f'src_input.txt')
    #             with open(src_input_path, 'w') as src_input_file:
    #                 for key, value in src_input.items():
    #                     src_input_file.write(f"{key}: {value}\n")

    #             # Create a text file to store txt_hyp and txt_ref
    #             text_file_path = os.path.join(sample_dir, f'{temp_name}.txt')
    #             with open(text_file_path, 'w') as text_file:
    #                 text_file.write(f"txt_hyp: {txt_hyp}\n")
    #                 text_file.write(f"txt_ref: {txt_ref}\n")

    #             # Optionally, play the audio (comment out if not needed)
    #             # Audio(hyp_path, autoplay=True)
    #             # Audio(ref_path, autoplay=True)

    #             print('txt_ref: ', txt_ref)

    #             print(
    #                 f'Name: {name}' + '\n' +
    #                 f'txt_hyp: {txt_hyp}' + '\n' +
    #                 f'txt_ref: {txt_ref}' + '\n' +
    #                 f'type: {prefix}'
    #             )

    #             last_result.append(
    #                 {
    #                     'name': name,
    #                     'txt_hyp': txt_hyp,
    #                     'txt_ref': txt_ref,
    #                     'prefix': prefix,
    #                 }
    #             )

    #             # Clear variables and call garbage collection
    #             del tts_hyp, tts_ref
    #             gc.collect()

    #         print('last_result: ', last_result)

    #         os.makedirs('../result/json', exist_ok=True)

    #         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    #         # store data to json file
    #         with open(f'../result/json/last_result_{timestamp}.json', 'w') as f:
    #             json.dump(last_result, f, indent=4)

    #     metric_logger.update(loss=output['total_loss'].item())

    #     if do_recognition:
    #         evaluation_results = {}
    #         evaluation_results['wer'] = 200
    #         for hyp_name in results[name].keys():
    #             if not 'gls_hyp' in hyp_name:
    #                 continue
    #             k = hyp_name.replace('gls_hyp', '')
    #             if config['data']['dataset_name'].lower() == 'phoenix-2014t':
    #                 gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
    #                 gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results]
    #             elif config['data']['dataset_name'].lower() == 'phoenix-2014':
    #                 gls_ref = [clean_phoenix_2014(results[n]['gls_ref']) for n in results]
    #                 gls_hyp = [clean_phoenix_2014(results[n][hyp_name]) for n in results]
    #             elif config['data']['dataset_name'].lower() == 'csl-daily':
    #                 gls_ref = [results[n]['gls_ref'] for n in results]
    #                 gls_hyp = [results[n][hyp_name] for n in results]
    #             wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
    #             evaluation_results[k + 'wer_list'] = wer_results
    #             evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
    #         metric_logger.update(wer=evaluation_results['wer'])

    #     if do_translation:
    #         txt_ref = [results[n]['txt_ref'] for n in results]
    #         txt_hyp = [results[n]['txt_hyp'] for n in results]
    #         bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
    #         rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
    #         for k, v in bleu_dict.items():
    #             print('{} {:.2f}'.format(k, v))
    #         print('ROUGE: {:.2f}'.format(rouge_score))
    #         evaluation_results['rouge'], evaluation_results['bleu'] = rouge_score, bleu_dict
    #         wandb.log({'eval/BLEU4': bleu_dict['bleu4']})
    #         wandb.log({'eval/ROUGE': rouge_score})
    #         metric_logger.update(bleu1=bleu_dict['bleu1'])
    #         metric_logger.update(bleu2=bleu_dict['bleu2'])
    #         metric_logger.update(bleu3=bleu_dict['bleu3'])
    #         metric_logger.update(bleu4=bleu_dict['bleu4'])
    #         metric_logger.update(rouge=rouge_score)

    # if args.run:
    #     args.run.log(
    #         {'epoch': epoch + 1, 'epoch/dev_loss': output['recognition_loss'].item(), 'wer': evaluation_results['wer']})
    # print("* Averaged stats:", metric_logger)
    # print('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    # return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


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
    Path(config['training']['model_dir']).mkdir(parents=True, exist_ok=True)
    main(args, config)
