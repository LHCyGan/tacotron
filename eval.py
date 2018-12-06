import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  'ni3 xu1 yao4 zhei4 yangr4 de5 yi2 ge4 li4 zi5', # 你需要这样儿的一个例子。
  'zhong1 wen2 ta1 ke5 chuang4', # 中文 tacotron
  'duo1 can1 jia1 hu4 wai4 huo2 dong4 hu1 xi1 qing1 xin1 kong1 qi4 la5', # 多参加户外活动，呼吸清新空气啦。
  'qi1 yue4 qi1 hao4 di4 yi1 chang2 you3 xiao2 yu3 yue4 dui4 ou1', # 7.7 第一场有小雨乐队哦！
]


def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
