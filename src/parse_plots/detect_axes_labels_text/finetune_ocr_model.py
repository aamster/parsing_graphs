import os

import yaml
from parse_plots.detect_axes_labels_text._ocr_trainer.train import train

from parse_plots.detect_axes_labels_text._ocr_trainer.utils import AttrDict


def main():
    with open('/allen/aibs/informatics/aamster/benetech-making-graphs-accessible/ocr_config.yml', 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)

    opt = AttrDict(opt)
    opt.character = opt.number + opt.symbol + opt.lang_char
    os.makedirs(f'{opt.model_dir}/{opt.experiment_name}', exist_ok=True)

    train(opt, amp=False)


if __name__ == '__main__':
    main()
