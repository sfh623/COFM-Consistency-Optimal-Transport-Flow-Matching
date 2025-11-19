# Copyright 2019-2020 Stanislav Pidhorskyi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from torch import nn
import torch
import ALAE.utils


def get_model_dict(x):
    if x is None:
        return None
    if isinstance(x, nn.DataParallel):
        return x.module.state_dict()
    else:
        return x.state_dict()


def load_model(x, state_dict):
    if isinstance(x, nn.DataParallel):
        x.module.load_state_dict(state_dict)
    else:
        x.load_state_dict(state_dict)


class Checkpointer(object):
    def __init__(self, cfg, models, auxiliary=None, logger=None, save=True):
        self.models = models
        self.auxiliary = auxiliary
        self.cfg = cfg
        self.logger = logger
        self._save = save

    def save(self, _name, **kwargs):
        if not self._save:
            return
        data = dict()
        data["models"] = dict()
        data["auxiliary"] = dict()
        for name, model in self.models.items():
            data["models"][name] = get_model_dict(model)

        if self.auxiliary is not None:
            for name, item in self.auxiliary.items():
                data["auxiliary"][name] = item.state_dict()
        data.update(kwargs)

        @utils.async_func
        def save_data():
            save_file = os.path.join(self.cfg.OUTPUT_DIR, "%s.pth" % _name)
            self.logger.info("Saving checkpoint to %s" % save_file)
            torch.save(data, save_file)
            self.tag_last_checkpoint(save_file)

        return save_data()
    
    def load(self, ignore_last_checkpoint=False, file_name=None):
        save_file = os.path.join(self.cfg.OUTPUT_DIR, "last_checkpoint")
    
        f = None
        try:
            with open(save_file, "r") as last_checkpoint:
                f = last_checkpoint.read().strip()
        except IOError:
            self.logger.info("No checkpoint found at %s" % save_file)
            if file_name is None:
                self.logger.info("Initializing model from scratch")
                return {}
    
        if ignore_last_checkpoint:
            self.logger.info("Forced to initialize model from scratch")
            return {}
    
        if file_name is not None:
            f = file_name
    
        candidates = []
    
        if f is not None and os.path.isabs(f):
            candidates.append(f)
        else:
            if f is not None:
                candidates.append(f)
            if f is not None:
                candidates.append(os.path.join(self.cfg.OUTPUT_DIR, f))
            if f is not None:
                candidates.append(os.path.join(self.cfg.OUTPUT_DIR, os.path.basename(f)))
    
            proj_root = os.path.abspath(os.path.join(self.cfg.OUTPUT_DIR, os.pardir, os.pardir, os.pardir))
            if f is not None:
                candidates.append(os.path.join(proj_root, f))
    
        ckpt_path = None
        for c in candidates:
            if c and os.path.exists(c):
                ckpt_path = c
                break
    
        if ckpt_path is None:
            msg = "Checkpoint not found. Tried:\n" + "\n".join(["  - " + str(c) for c in candidates])
            self.logger.error(msg)
            raise FileNotFoundError(msg)
    
        self.logger.info("Loading checkpoint from {}".format(ckpt_path))
        checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    
        for name, model in self.models.items():
            if name in checkpoint.get("models", {}):
                try:
                    model_dict = checkpoint["models"].pop(name)
                    if model_dict is not None:
                        self.models[name].load_state_dict(model_dict, strict=False)
                    else:
                        self.logger.warning('State dict for model "%s" is None' % name)
                except RuntimeError as e:
                    self.logger.warning('%s\nFailed to load: %s\n%s' % ('!' * 160, name, '!' * 160))
                    self.logger.warning('\nFailed to load: %s' % str(e))
            else:
                self.logger.warning("No state dict for model: %s" % name)
        checkpoint.pop('models', None)
    
        return checkpoint


    def tag_last_checkpoint(self, last_filename):
        save_file = os.path.join(self.cfg.OUTPUT_DIR, "last_checkpoint")
        with open(save_file, "w") as f:
            f.write(last_filename)
