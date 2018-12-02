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
3つのコンテンツの間に5分ずつの休憩をはさみ、2時間のハンズオンを通して学習できるよう構成されています。なお、これらのコンテンツを動かす SageMaker ノートブックインスタンスは `ml.c5.xlarge` を推奨します (`ml.t2.medium` でも動きますがローカルでの計算が少し遅くなります)。
1. SageMaker で PyTorch の分散学習 (40分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb "SAGEMAKER PYTHON SDK > pytorch_local_mode_cifar10.ipynb")]
1. ベイズ最適化による Hyper Parameter Optimization (HPO) (40分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/pytorch_mnist/hpo_pytorch_mnist.ipynb "HYPERPARAMETER TUNING > hpo_pytorch_mnist.ipynb")]
1. SageMaker で Torchvision の転移学習 (30分) [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/handson/pytorch/finetuning_torchvision_models_tutorial.ipynb "ADDITIONAL EXAMPLES > finetuning_torchvision_models_tutorial.ipynb")]

以下は必須ではありませんが追加のコンテンツです。
- (optional) HPO ジョブの可視化 [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb "HYPERPARAMETER TUNING > HPO_Analyze_TuningJob_Result.ipynb")]
- (optional) MNIST版の分散学習 [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_mnist/pytorch_mnist.ipynb "SAGEMAKER PYTHON SDK > pytorch_mnist.ipynb")]
- (optional) コンテナ作成 [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/advanced_functionality/pytorch_extending_our_containers/pytorch_extending_our_containers.ipynb "ADVANCED FUNCTIONALITY > pytorch_extending_our_containers.ipynb")]

### 1. [分散学習](https://github.com/hariby/amazon-sagemaker-examples/blob/master/sagemaker-python-sdk/pytorch_cnn_cifar10/pytorch_local_mode_cifar10.ipynb "SAGEMAKER PYTHON SDK > pytorch_local_mode_cifar10.ipynb")
- Cifar10 を使った学習スクリプト `source/cifar10.py` が用意されているので、これをエントリーポイントとした SageMaker の学習を行います。
    - デフォルトではローカルモードを用いて学習を行うようになっているので、ノートブックを書き換えて分散学習のジョブを発行します (実はスクリプト自体は元から対応しているので書き換えなくていい)。
    - ここでは `PyTorch 0.4.0` ビルド済みコンテナを呼び出しています。対応バージョンは[こちら](https://github.com/aws/sagemaker-python-sdk#pytorch-sagemaker-estimators)参照、なお 2018-12-02 時点の対応バージョンは `0.4.0`, `1.0.0.dev` ("Preview") です。
    - 出力を見て複数ノードで学習が分散されていることを確認します。
- [SageMaker PyTorch Estimator](https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/pytorch/README.rst) `sagemaker.pytorch.PyTorch` について: 
    - `hyperparameters={'epochs': 6}` でハイパーパラメータを渡すことができます。
    - (optional) `metric_definitions` で CloudWatch メトリクスとして結果を出力することができます [[ドキュメント](https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html)]。
    ```python
    metric_definitions=[
        {'Name': 'train:loss', 'Regex': 'loss: ([0-9\.]+)'},
    ]
    ```
- (optional) 学習スクリプト `source/cifar10.py` 68行目 `transforms.Compose([])` の中に以下の操作などを書き足して Data augumentation するようにして精度を比較 [[ドキュメント](https://pytorch.org/docs/stable/torchvision/transforms.html)]。
    ```python
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip()
    ```
- (optional) 出力されたモデルを S3 から取得しノートブックインスタンス上の Jupyter Notebook で読み込んで推論を行ってみましょう。

### 2. [ベイズ最適化 (HPO)](https://github.com/hariby/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/pytorch_mnist/hpo_pytorch_mnist.ipynb "HYPERPARAMETER TUNING > hpo_pytorch_mnist.ipynb")
- SageMaker ではベイズ最適化を用いて、正規表現でパースされたメトリクスに対してハイパーパラメータの最適化 (HPO) を行うことができます。
- (optional) HPO ジョブの結果を可視化しましょう [[notebook](https://github.com/hariby/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb "HYPERPARAMETER TUNING > HPO_Analyze_TuningJob_Result.ipynb")]。
- (optional) 新たなパラメータを最適化対象として追加してみましょう。
    - `'momentum': ContinuousParameter(0.1, 0.5)` など。
- (optional) Warm Start を使って最適化ジョブを継続するよう書き換えてみましょう [[ドキュメント](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-warm-start.html), [参考ブログ](https://aws.amazon.com/jp/blogs/news/amazon-sagemaker-automatic-model-tuning-becomes-more-efficient-with-warm-start-of-hyperparameter-tuning-jobs/)]。
    - ```python
      from sagemaker.tuner import WarmStartConfig, WarmStartTypes
      hpo_warm_start_config = WarmStartConfig(WarmStartTypes.IDENTICAL_DATA_AND_ALGORITHM,
                                    parents={'<parent_tuning_job_name>','<parent_tuning_job_name_2>'})
      ```
    
    を実行し、 `warm_start_config=hpo_warm_start_config` を `HyperparameterTuner()` 作成時に追加。

### 3. [転移学習](https://github.com/hariby/amazon-sagemaker-examples/blob/master/handson/pytorch/finetuning_torchvision_models_tutorial.ipynb "ADDITIONAL EXAMPLES > finetuning_torchvision_models_tutorial.ipynb")
- Torchvision で学習済みの Squeezenet を読み込んで、アリとハチのデータセットを用いて2値分類のモデルを学習させます。
    - `feature_extract` 変数により、 finetune / feature extract の2種類の方法を試すことができます。
- 未学習のモデルを学習させて、上記の手順との学習速度・精度を比較します。
- (optional) 他のモデルやサイズの違うモデル (Alexnet や Resnet34 など) を使って試します。
