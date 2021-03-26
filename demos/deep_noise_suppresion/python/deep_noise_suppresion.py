#!/usr/bin/env python3

"""
 Copyright (c) 2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
import soundfile

from openvino.inference_engine import IECore

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
                      required=True, type=Path)
    args.add_argument("-i", "--input", help="Required. Path to a 16kHz sound file with speech+noise",
                      required=True, type=str)
    args.add_argument("-o", "--output", help="Required. Path to output sound file with speech",
                      required=True, type=str)
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on."
                           "Default value is CPU",
                      default="CPU", type=str)
    return parser

def main():
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
    print(ie_encoder.inputs)


    # check input and output names
    input_names = list(i.strip() for i in ie_encoder.inputs.keys())
    output_names = list(o.strip() for o in ie_encoder.outputs.keys())

    assert "input" in input_names, "'input' is not presented in model"
    assert "output" in output_names, "'output' is not presented in model"
    state_inp_names = list(sorted([n for n in input_names if "state" in n]))
    state_param_num = sum(np.prod(ie_encoder.inputs[n].shape) for n in state_inp_names)
    log.info("state_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))

    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)

    sample_inp, freq = soundfile.read(args.input, start=0, stop=None, dtype='float32')
    assert freq==16000
    if len(sample_inp.shape)==2:
        sample_inp = sample_inp.mean(0)

    input_size = ie_encoder.inputs["input"].shape[1]
    res = None

    samples_out = []
    samples_times = []
    while sample_inp is not None and sample_inp.shape[0]>0:
        if sample_inp.shape[0] > input_size:
            input = sample_inp[:input_size]
            sample_inp = sample_inp[input_size:]
        else:
            input = np.pad(sample_inp, ((0,input_size - sample_inp.shape[0]),), mode='constant')
            sample_inp = None


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
        # infer by IE
        res = ie_encoder_exec.infer(inputs=inputs)
        t1 = time.perf_counter()
        samples_times.append(t1-t0)
        samples_out.append(res["output"].squeeze(0))

    log.info("Sequence of length {:0.2f}s is processed by {:0.2f}s".format(
        sum(s.shape[0] for s in samples_out)/freq,
        sum(samples_times)
    ))
    sample_out = np.concatenate(samples_out,0)
    soundfile.write(args.output, sample_out, freq)


if __name__ == '__main__':
    sys.exit(main() or 0)
