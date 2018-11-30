# SageMaker PyTorch ハンズオン

このハンズオンは [Amazon SageMaker](https://aws.amazon.com/jp/sagemaker/) 上で [PyTorch](https://pytorch.org/) を使った機械学習/深層学習を学ぶことを目的としています。

## 学べること
このコースを終えたあと、以下のような概念/方法を習得することができます。
1. Amazon SageMaker を使って PyTorch のモデルを構築・学習・デプロイする方法
1. Amazon SageMaker を使った分散学習 (複数GPU、あるいはマルチノード)
1. Amazon SageMaker を使ったハイパーパラメータのチューニング

なお、以下の知識を前提とします。
1. 機械学習/深層学習の概念と一般的な理解
1. Python/PyTorch を用いたプログラミング
1. AWS の基礎的な用語と操作方法

## コンテンツ
3つのコンテンツの間に5分ずつの休憩をはさみ、2時間のハンズオンを通して学習できるよう構成されています。
1. SageMaker で Torchvision の転移学習 (30分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/handson/pytorch/finetuning_torchvision_models_tutorial.ipynb)]
1. SageMaker で PyTorch の分散学習 (40分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb)]
1. Hyper Parameter Optimization (40分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/pytorch_mnist/hpo_pytorch_mnist.ipynb)]
