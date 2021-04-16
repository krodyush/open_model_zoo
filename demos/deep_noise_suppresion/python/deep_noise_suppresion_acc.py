import logging as log
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
import soundfile

from openvino.inference_engine import IECore

FREQ = 16000

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
                      required=True, type=Path)
    args.add_argument("-i", "--input", help="Required. Path to a dataset file",
                      required=True, type=str)
    args.add_argument("--delay", help="Required. signal delay by network in samples. specific for each network",
                      required=True, type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on."
                           "Default value is CPU",
                      default="CPU", type=str)
    return parser

def sisdr_metric(y, target):

    dim_agregate = -1
    EPS = np.finfo(float).eps

    target = target - np.mean(target, axis=dim_agregate, keepdims=True)
    y = y - np.mean(y, axis=dim_agregate, keepdims=True)

    y_by_target = np.sum(y * target, axis=dim_agregate, keepdims=True)
    t2 = np.sum(target ** 2, axis=dim_agregate, keepdims=True)
    y_target = y_by_target * target / (t2 + EPS)
    y_noise = y - y_target

    target_pow = np.sum(y_target ** 2, axis=dim_agregate)
    noise_pow = np.sum(y_noise ** 2, axis=dim_agregate)

    sisdr = 10 * np.log10(target_pow + EPS) - 10 * np.log10(noise_pow + EPS)
    return sisdr


def main():

    #sample that outputs are delayed
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Initializing Inference Engine")
    ie = IECore()
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = model_xml.with_suffix(".bin")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)

    # check input and output names
    input_names = list(i.strip() for i in ie_encoder.inputs.keys())
    output_names = list(o.strip() for o in ie_encoder.outputs.keys())
    assert "input" in input_names, "'input' is not presented in model"
    assert "output" in output_names, "'output' is not presented in model"
    input_size = ie_encoder.inputs["input"].shape[1]

    #get names for states
    state_inp_names = list(sorted([n for n in input_names if "state" in n]))
    state_param_num = sum(np.prod(ie_encoder.inputs[n].shape) for n in state_inp_names)
    log.info("state_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))


    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)

    #create dataset
    dataset = []
    with open(args.input) as inp:
        base_name = os.path.split(args.input)[0]
        for l in inp.readlines():
            dataset.append([os.path.join(base_name,f) for f in l.strip().split()])

    #iterate over dataset
    sisdr_inps = []
    sisdr_outs = []
    for f_clean, f_noisy in dataset:
        #read clean and noisy signals
        def read(f):
            sample, freq = soundfile.read(f, start=0, stop=None, dtype='float32')
            assert freq == FREQ, "file {} sample rate {} != {}".format(f, freq, FREQ)
            if len(sample.shape) == 2:
                sample = sample.mean(0)
            return sample

        x_clean, x_noisy = read(f_clean), read(f_noisy)
        assert x_clean.shape[0] == x_noisy.shape[0]

        #make both signals divisable by input_size
        patch_num, rest_size = divmod(x_noisy.shape[0], input_size)
        if rest_size>0:
            pad = input_size - rest_size
            x_clean = np.pad(x_clean, ((pad,0),), mode='constant')
            x_noisy = np.pad(x_noisy, ((pad,0),), mode='constant')
            patch_num += 1

        #split noisy signal into patches to feed network
        samples_inp = np.split(x_noisy, patch_num)
        samples_out = []
        samples_times = []
        res = None
        for input in samples_inp:
            #forms input
            inputs = {"input": input[None,:]}

            #add states to input
            for n in state_inp_names:
                if res:
                    inputs[n] = res[n.replace('inp', 'out')]
                else:
                    shape = ie_encoder.inputs[n].shape
                    inputs[n] = np.zeros(shape, dtype=np.float32)

            t0 = time.perf_counter()
            res = ie_encoder_exec.infer(inputs=inputs)
            t1 = time.perf_counter()
            samples_times.append(t1-t0)
            samples_out.append(res["output"].squeeze(0))

        y = np.concatenate(samples_out, 0)

        #shift output by delay to align with input
        y = y[args.delay:]
        x_clean = x_clean[:-args.delay]
        x_noisy = x_noisy[:-args.delay]

        #calc metrics for input and output
        sisdr_inps.append(sisdr_metric(x_noisy, x_clean))
        sisdr_outs.append(sisdr_metric(y,x_clean))

        sample_len = sum(s.shape[0] for s in samples_out)/FREQ
        sample_time = sum(samples_times)
        log.info("Sample of length {:0.2f}s is processed by {:0.2f}s x{:0.2f} sisdr_inp,out,diff {:0.2f} {:0.2f} {:0.2f}".format(
            sample_len,
            sample_time,
            sample_time / sample_len,
            sisdr_inps[-1],
            sisdr_outs[-1],
            sisdr_outs[-1] - sisdr_inps[-1]
        ))

    N = len(sisdr_inps)
    sisdr_inp = sum(sisdr_inps)/N
    sisdr_out = sum(sisdr_outs)/N
    print("aver sisdr_inp,out,diff {:0.2f} {:0.2f} {:0.2f} for {} samples".format(
        sisdr_inp,
        sisdr_out,
        sisdr_out-sisdr_inp,
        N
    ))

if __name__ == '__main__':
    sys.exit(main() or 0)
