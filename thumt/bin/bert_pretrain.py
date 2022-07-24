# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import glob
import logging
import os
import re
import socket
import time

import six
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import thumt.models as models
import thumt.optimizers as optimizers
import thumt.utils as utils
import thumt.utils.summary as summary

import thumt.data as data


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model.",
        usage="trainer.py [<args>] [-h | --help]"
    )
    parser.add_argument("--input", type=str,
                        help="Path to source and target corpus.")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to load/store checkpoints.")
    parser.add_argument("--vocabulary", type=str,
                        help="Path to source and target vocabulary.")
    parser.add_argument("--validation", type=str,
                        help="Path to validation file.")
    parser.add_argument("--references", type=str,
                        help="Pattern to reference files.")
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training.")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process.")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed-precision training.")
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper-parameter set.")

    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")
    return parser.parse_args(args)


def default_params():
    params = utils.HParams(
        input="",
        output="",
        model="bert",
        vocab="",
        pad="[PAD]",
        bos="[EOS]",
        eos="[EOS]",
        unk="[UNK]",
        # Dataset
        batch_size=512,
        fixed_batch_size=False,
        min_length=1,
        max_length=512,
        buffer_size=10000,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        initial_step=0,
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-7,
        pattern="",
        clipping="global_norm",
        clip_grad_norm=5.0,
        learning_rate=1.0,
        initial_learning_rate=0.0,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        device_list=[0],
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_summary=True,
        save_checkpoint_secs=0,
        save_checkpoint_steps=2,
        # Validation
        eval_steps=2000,
        eval_secs=0,
        top_beams=1,
        beam_size=4,
        decode_batch_size=32,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        validation="",
        references="",
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if os.path.exists(p_name):
        with open(p_name) as fd:
            logging.info("Restoring hyper parameters from %s" % p_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    if os.path.exists(m_name):
        with open(m_name) as fd:
            logging.info("Restoring model parameters from %s" % m_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    return params


def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    with open(filename, "w") as fd:
        fd.write(params.to_json())


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters.lower())

    params.vocabulary = {
        "source": data.Vocabulary(params.vocab)
    }

    return params


def collect_params(all_params, params):
    collected = utils.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def print_variables(model, pattern, log=True):
    flags = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name):
            flags.append(True)
        else:
            flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable variables size: %d" % total_size)

    return flags


def exclude_variables(flags, grads_and_vars):
    idx = 0
    new_grads = []
    new_vars = []

    for grad, (name, var) in grads_and_vars:
        if flags[idx]:
            new_grads.append(grad)
            new_vars.append((name, var))

        idx += 1

    return zip(new_grads, new_vars)


def save_checkpoint(step, epoch, model, optimizer, params):
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def get_learning_rate_schedule(params):
    if params.learning_rate_schedule == "linear_warmup_rsqrt_decay":
        schedule = optimizers.LinearWarmupRsqrtDecay(
            params.learning_rate, params.warmup_steps,
            initial_learning_rate=params.initial_learning_rate,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "piecewise_constant_decay":
        schedule = optimizers.PiecewiseConstantDecay(
            params.learning_rate_boundaries, params.learning_rate_values,
            summary=params.save_summary)
    elif params.learning_rate_schedule == "linear_exponential_decay":
        schedule = optimizers.LinearExponentialDecay(
            params.learning_rate, params.warmup_steps,
            params.start_decay_step, params.end_decay_step,
            dist.get_world_size(), summary=params.save_summary)
    elif params.learning_rate_schedule == "constant":
        schedule = params.learning_rate
    else:
        raise ValueError("Unknown schedule %s" % params.learning_rate_schedule)

    return schedule


def get_clipper(params):
    if params.clipping.lower() == "none":
        clipper = None
    elif params.clipping.lower() == "adaptive":
        clipper = optimizers.adaptive_clipper(0.95)
    elif params.clipping.lower() == "global_norm":
        clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
    else:
        raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_optimizer(params, schedule, clipper):
    if params.optimizer.lower() == "adam":
        optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                             beta_1=params.adam_beta1,
                                             beta_2=params.adam_beta2,
                                             epsilon=params.adam_epsilon,
                                             clipper=clipper,
                                             summaries=params.save_summary)
    elif params.optimizer.lower() == "adadelta":
        optimizer = optimizers.AdadeltaOptimizer(
            learning_rate=schedule, rho=params.adadelta_rho,
            epsilon=params.adadelta_epsilon, clipper=clipper,
            summaries=params.save_summary)
    elif params.optimizer.lower() == "sgd":
        optimizer = optimizers.SGDOptimizer(
            learning_rate=schedule, clipper=clipper,
            summaries=params.save_summary)
    else:
        raise ValueError("Unknown optimizer %s" % params.optimizer)

    return optimizer


def load_references(pattern):
    if not pattern:
        return None

    files = glob.glob(pattern)
    references = []

    for name in files:
        ref = []
        with open(name, "rb") as fd:
            for line in fd:
                items = line.strip().split()
                ref.append(items)
        references.append(ref)

    return list(zip(*references))


def evaluate(model, eval_loader):
    # # 将模型放到服务器上
    # model.to(device)
    # 设定模式为验证模式
    model.eval()
    # 设定不会有梯度的改变仅作验证
    mlm_l_avg = 0
    nsp_l_avg = 0
    l_avg = 0
    mlm_precision_avg = 0
    nsp_precision_avg = 0
    with torch.no_grad():
        for step, inputs in enumerate(eval_loader):
            print("Dev Step[{}/{}]".format(step + 1, len(eval_loader)))
            # input_ids, token_type_ids, attention_mask, labels = input_ids.to(device), token_type_ids.to(
            #     device), attention_mask.to(device), labels.to(device)

            train_data = inputs[0]
            segments = inputs[1]
            pred_positions = inputs[3]
            mlm_weights = inputs[4]
            nsp_labels = inputs[5]
            mlm_labels = inputs[6]

            bias = [example[-1] for example in inputs]

            mlm_l, nsp_l, l, mlm_precision, nsp_precision = model(train_data, segments, bias, mlm_weights,
                                                                  nsp_labels, mlm_labels,
                                                                  pred_positions=pred_positions)
            mlm_l_avg += mlm_l
            nsp_l_avg += nsp_l
            l_avg += l
            mlm_precision_avg += mlm_precision
            nsp_precision_avg += nsp_precision

        mlm_l_avg /= len(eval_loader)
        nsp_l_avg /= len(eval_loader)
        l_avg /= len(eval_loader)
        mlm_precision_avg /= len(eval_loader)
        nsp_precision_avg /= len(eval_loader)

        return mlm_l_avg, nsp_l_avg, l_avg, mlm_precision_avg, nsp_precision_avg


def main(args):
    model_cls = models.get_model(args.model)

    params = default_params()
    params = merge_params(params, model_cls.default_params(args.hparam_set))
    params = import_params(args.output, args.model, params)

    # 词表（word2id,id2word)
    params = override_params(params, args)

    if args.distributed:
        params.device = args.local_rank
        dist.init_process_group("gloo")
        torch.cuda.set_device(args.local_rank)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        params.device = params.device_list[args.local_rank]
        dist.init_process_group("gloo", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Export parameters
    if dist.get_rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(params.output, "%s.json" % params.model,
                      collect_params(params, model_cls.default_params()))

    model = model_cls(params).cuda()

    if args.half:
        model = model.half()
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model.train()

    summary.init(params.output, params.save_summary)

    schedule = get_learning_rate_schedule(params)
    clipper = get_clipper(params)
    optimizer = get_optimizer(params, schedule, clipper)

    if args.half:
        optimizer = optimizers.LossScalingOptimizer(optimizer)

    optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

    trainable_flags = print_variables(model, params.pattern,
                                      dist.get_rank() == 0)

    t = time.time()

    train_dataset = data.FilmDataset(input_path=params.input, max_len=512, params=params,
                                     vocab_path=params.vocab)  # vocab 需要绝对路径
    train_loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size, shuffle=True)
    t = time.time() - t
    print("train_data load successfully  (%.3f sec)" % t)

    if params.validation:
        t = time.time()
        eval_dataset = data.FilmDataset(input_path=params.validation, max_len=512, params=params,
                                        vocab_path=params.vocab)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=params.batch_size, shuffle=False)
        t = time.time() - t
        print("dev_data load successfully  (%.3f sec)" % t)

    # Load checkpoint
    checkpoint = utils.latest_checkpoint(params.output)

    if args.checkpoint is not None:
        # Load pre-trained models
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"])
        step = params.initial_step
        epoch = 0
        broadcast(model)
    elif checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0
        epoch = 0
        broadcast(model)

    def train_fn(inputs):
        """
        inputs: batch_size * dataset的__getitem__
        inputs:
                batch_size * self.all_token_ids[idx], batch_size * self.all_segments[idx],\
                batch_size * self.valid_lens[idx],batch_size * self.all_pred_positions[idx],\
                batch_size * self.all_mlm_weights[idx],batch_size * self.all_mlm_labels[idx],\
                batch_size * self.nsp_labels[idx], batch_size * self.attention_mask[idx]
        """

        # train_data [batch_size,len(sentence)]
        train_data = inputs[0]
        segments = inputs[1]
        pred_positions = inputs[3]
        mlm_weights = inputs[4]
        mlm_labels = inputs[5]
        nsp_labels = inputs[6]

        bias = inputs[-1]

        mlm_l, nsp_l, l, mlm_precision, nsp_precision = model(train_data, segments, bias, mlm_weights,
                                                              nsp_labels, mlm_labels,
                                                              pred_positions=pred_positions)

        return mlm_l, nsp_l, l, mlm_precision, nsp_precision

    counter = 0

    while True:
        for features in train_loader:

            if counter % params.update_cycle == 0:
                step += 1
                utils.set_global_step(step)

            counter += 1
            t = time.time()

            mlm_l, nsp_l, loss, mlm_precision, nsp_precision = train_fn(features)

            gradients = optimizer.compute_gradients(loss,
                                                    list(model.parameters()))
            grads_and_vars = exclude_variables(
                trainable_flags,
                zip(gradients, list(model.named_parameters())))
            optimizer.apply_gradients(grads_and_vars)

            t = time.time() - t

            summary.scalar("loss", loss, step, write_every_n_steps=1)
            summary.scalar("global_step/sec", t, step)

            print("epoch = %d, step = %d, loss = %.3f mlm_precision = %.3f, nsp_precision = %.3f (%.3f sec)" %
                  (epoch + 1, step, float(loss), float(mlm_precision), float(nsp_precision), t))

            if counter % params.update_cycle == 0:
                if params.validation and step >= params.train_steps:
                    mlm_l_avg, nsp_l_avg, l_avg, mlm_precision_avg, nsp_precision_avg = evaluate(model, eval_loader)
                    save_checkpoint(step, epoch, model, optimizer, params)

                    print(
                        "validation mlm_l_avg = %.3f, nsp_l_avg = %.3f, l_avg = %.3f, mlm_precision_avg = %.3f, nsp_precision_avg =%.3f " %
                        (float(mlm_l_avg), float(nsp_l_avg), float(l_avg), float(mlm_precision_avg),
                         float(nsp_precision_avg)))

                    if dist.get_rank() == 0:
                        summary.close()

                    return

                if params.validation and step % params.eval_steps == 0:
                    mlm_l_avg, nsp_l_avg, l_avg, mlm_precision_avg, nsp_precision_avg = evaluate(model, eval_loader)
                    print(
                        "validation mlm_l_avg = %.3f, nsp_l_avg = %.3f, l_avg = %.3f, mlm_precision_avg = %.3f, nsp_precision_avg =%.3f " %
                        (float(mlm_l_avg), float(nsp_l_avg), float(l_avg), float(mlm_precision_avg),
                         float(nsp_precision_avg)))

                if step % params.save_checkpoint_steps == 0:
                    save_checkpoint(step, epoch, model, optimizer, params)

        epoch += 1


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        # 显卡数量
        world_size = infer_gpu_num(parsed_args.parameters)

        if world_size > 1:
            torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                        nprocs=world_size)
        else:
            process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
