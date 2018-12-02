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
1. AWS の基礎的な知識と操作方法

## コンテンツ
3つのコンテンツの間に5分ずつの休憩をはさみ、2時間のハンズオンを通して学習できるよう構成されています。なお、これらのコンテンツは SageMaker を動かすノートブックインスタンスは `ml.c5.xlarge` を推奨します (`ml.t2.medium` でも動きますが少し遅くなります)。
1. SageMaker で Torchvision の転移学習 (30分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/handson/pytorch/finetuning_torchvision_models_tutorial.ipynb)]
1. SageMaker で PyTorch の分散学習 (40分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb)]
1. ベイズ最適化による Hyper Parameter Optimization (40分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/pytorch_mnist/hpo_pytorch_mnist.ipynb)]

以下は必須ではありませんが追加のコンテンツです。
- (optional) Local mode スクリプトの書き換え [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb)]
- (optional) コンテナ作成 [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb)]

### 1. 転移学習
- Torchvision で学習済みの Squeezenet を読み込んで、アリとハチのデータセットを用いて2値分類のモデルを学習させます。
    - `feature_extract` 変数により、 finetune / feature extract の2種類の方法を試すことができます。
- 未学習のモデルを学習させて、上記の手順との学習速度・精度を比較します。
- (optional) 他のモデルやサイズの違うモデル (Alexnet や Resnet34 など) を使って試します。

### 2. 分散学習
- MNIST を使った学習スクリプト `mnist.py` が用意されているので、これをエントリーポイントとした SageMaker の学習を行います。
    - デフォルト `train_instance_count=2, train_instance_type='ml.c4.xlarge'` では2台の `ml.c4.xlarge (4 vCPUs)` で分散学習が行われます。
    - 出力を見て複数ノードで学習が分散されていることを確認します。
- (optional) インスタンスタイプ・インスタンス(ノード)数を変えて学習ジョブを走らせてみましょう。
- (optional) 出力されたモデルを S3 から取得しノートブックインスタンス上の Jupyter Notebook で読み込ん推論を行ってみましょう。

### 3. ベイズ最適化
- ベイズ最適化を用いてハイパーパラメータの最適化を行うことができます。
- (optional) 新たなパラメータを最適化対象として追加してみましょう。