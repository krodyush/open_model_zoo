# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import platform
import re
import shlex

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = PACKAGE_DIR / 'models'
DATASET_DEFINITIONS = PACKAGE_DIR / 'data/dataset_definitions.yml'

if not MODEL_ROOT.exists() or not DATASET_DEFINITIONS.exists():
    # We are run directly from OMZ rather than from an installed environment.
    _OMZ_ROOT = PACKAGE_DIR.parents[4]
    MODEL_ROOT = _OMZ_ROOT / 'models'
    DATASET_DEFINITIONS = _OMZ_ROOT / 'data/dataset_definitions.yml'

# make sure to update the documentation if you modify these
KNOWN_FRAMEWORKS = {
    'caffe': None,
    'caffe2': 'caffe2_to_onnx.py',
    'dldt': None,
    'mxnet': None,
    'onnx': None,
    'pytorch': 'pytorch_to_onnx.py',
    'tf': None,
}
KNOWN_PRECISIONS = {
    'FP16', 'FP16-INT1', 'FP16-INT8',
    'FP32', 'FP32-INT1', 'FP32-INT8',
}
KNOWN_TASK_TYPES = {
    'action_recognition',
    'classification',
    'colorization',
    'detection',
    'face_recognition',
    'feature_extraction',
    'head_pose_estimation',
    'human_pose_estimation',
    'image_inpainting',
    'image_processing',
    'image_translation',
    'instance_segmentation',
    'machine_translation',
    'monocular_depth_estimation',
    'named_entity_recognition',
    'object_attributes',
    'optical_character_recognition',
    'place_recognition',
    'question_answering',
    'salient_object_detection',
    'semantic_segmentation',
    'sound_classification',
    'speech_recognition',
    'style_transfer',
    'token_recognition',
    'text_to_speech',
}

KNOWN_QUANTIZED_PRECISIONS = {p + '-INT8': p for p in ['FP16', 'FP32']}
assert KNOWN_QUANTIZED_PRECISIONS.keys() <= KNOWN_PRECISIONS

def quote_arg_windows(arg):
    if not arg: return '""'
    if not re.search(r'\s|"', arg): return arg
    # On Windows, only backslashes that precede a quote or the end of the argument must be escaped.
    return '"' + re.sub(r'(\\+)$', r'\1\1', re.sub(r'(\\*)"', r'\1\1\\"', arg)) + '"'

if platform.system() == 'Windows':
    quote_arg = quote_arg_windows
else:
    quote_arg = shlex.quote

def command_string(args):
    return ' '.join(map(quote_arg, args))