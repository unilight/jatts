<p align="center">
  <img src="assets/jatts_logo.png" alt="Prometheus-Logo" style="width: 50%; display: block; margin: auto;">
</p>

# JATTS：日本語テキスト音声合成における手法比較に向けたオープンツールキット

### JATTS: A modern, research-oriented Japanese Text-to-speech Open-sourced Toolkit

![visitors](https://visitor-badge.laobi.icu/badge?page_id=unilight.jatts)

[[Tech report(技術報告)]](./assets/tech-report-JATTS.pdf)

本リポジトリは研究に特化したオープンソース日本語テキスト音声合成（text-to-speech, TTS）ツールキットです。最大の目的としては異なるモデルの性能比較なので，推論速度などの最適化は現時点まだ行っていません。

> レシピやモデルの拡張に興味ある方はぜひissueやPRなどで一緒に開発してください！

## インストール

ESPnetと同じく，本リポジトリは学習の時に必要な環境を事前に構築し，学習など指令を行うたびにその環境を自動的に起動する形になります。環境の構築は以下の指令に従ってください。

```
git clone https://github.com/unilight/jatts.git
cd jatts/tools
make
```

## 使用方法（2025年5月末時点）

TTSモデルの学習には，(1)どの学習データセットを使うか，(2)どのモデルを使うか，を決めないといけないです。  
基本的にな考え方として，

> **学習データセットはレシピ(recipe)で決める。モデルは設定ファイル(config)で決める。**

### レシピとは？

本リポジトリは，KaldiやESPnetなど音声処理ツールキットと同じく，レシピという形式でまとめられています。レシピとは，実験を再現するために必要な手順が全て含まれているスクリプト一式のことを指します。データのダウンロード，前処理，特徴量抽出，モデルの学習，そしてモデルの評価を含みます。つまり，レシピを実行するだけで，ユーザーは実験の結果を再現することが可能です。

現在サポートしているレシピは以下になります。

- [JSUT](egs/jsut)：単一話者
- [Hi-Fi-Captain](egs/hificaptain_jp_female/)：単一話者
- [JVS](egs/jvs)：多話者

さらに，各レシピの下に，`tts1`/`tts2`があります。

- `tts1`：forced alignerによる音素ごと継続長取得。FastSpeech2とMatcha-TTSサポート。現時点は[Julius](https://github.com/julius-speech/segmentation-kit)を使用します。
- `tts2`：[自動アライメント探索](https://arxiv.org/abs/2108.10447)による継続長モデリング。Matcha-TTSとVITSサポート。

### モデル

現時点の**研究向け**のTTSモデルは主にメルスペクトログラム(Mel-spectrogram；メルスペ)という音声特徴量を媒質として，テキストからメルスペを生成するText2Melモデルと、メルスペから波形を生成するボコーダ（Vocoder）モデルに別れます。

Vocoderに関して，現時点は[ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)でサポートされるボコーダを利用します。その中，主に[HiFi-GAN](https://arxiv.org/abs/2010.05646)を使用します。

現時点サポートしているText2Melモデルは主に以下になります。

- FastSpeech 2
- VITS（メルスペ版）
- Matcha-TTS

### レシピ構造

各レシピには`egs/XXX/ttsX/run.sh`というスクリプトがあります。スクリプトには以下のステージに分かれます。

#### データセットダウンロード

```
./run.sh --stage -1 --stop_stage -1
```

このステージはデータセットのダウンロードを行います。
- `local/data_download.sh`というスクリプトが自動的に実行されます。
- ダウンロードしたデータセットに位置は`run.sh`の`db_root`変数で変えられます。デフォルトは`downloads`に置きます。

#### データ準備

```
./run.sh --stage 0 --stop_stage 0
```

このステージはデータセットを`.csv`形式でデータをまとめます。

- 各csvファイルには以下の列があります。
  - `sample_id`：サンプルのID。
  - `spk`：話者ID。単一話者データセットの場合は使われません。
  - `wav_path`：音声サンプルのファイルパス。
  - `start`,`end`：発話の開始と終了時間。forced alignerかVADで抽出します。
  - `original_text`：テキストの原本。通常は漢字仮名交じり文になります。
  - `phonemes`：G2Pで抽出した音素系列。Juliusのforced alignerの変換結果をそのまま使うか，[pyopenjtalk](https://github.com/r9y9/pyopenjtalk)で変換したものを使用します。
  - `durations`：各音素の継続長系列。forced alignerで抽出します。`tts1`のみ使用します。
  - `feat_path`：抽出した特徴量ファイルのパス。以下の"特徴量抽出"に参照。
  - `ref_wav_path`：参照音声サンプルのファイルパス。多話者データセットのみ使用します。
- `tts1`だと，はじめに`train.pre_julius.csv`,`dev.pre_julius.csv`が生成されます。そして`tts1`では，juliusによるforced alignmentが実行されます。抽出した音素継続長を加えて，`train.csv`,`dev.csv`に保存します。
- `tts2`だと，`train.csv`,`dev.csv`がそのまま生成されます。


#### 特徴量抽出

```
./run.sh --stage 1 --stop_stage 1
```

このステージは学習用の特徴量をマルチプロセル方式で抽出し，保存します。
- デフォルトは`dump/train/feats`,`dump/dev/feats`に置きます。
- `.h5`の形で保存します。以下，例です。
```
>>> h5ls egs/jvs/tts1/dump/dev/feats/jvs087_nonparallel_BASIC5000_0068.h
energy                   Dataset {47}
mel                      Dataset {295, 80}
pitch                    Dataset {47}
spkemb                   Dataset {192}
wave                     Dataset {88320}
```
- `.h5`のファイルパスを更新し，`train_raw_feat.csv`,`dev_raw_feat.csv`に保存します。

#### トークンリスト生成

```
./run.sh --stage 2 --stop_stage 2
```

このステージはテキストトークンのリストを生成し，保存します。
- デフォルトは`dump/token_list/train_phn_XXX`に置きます。
  - `tts1`だと，音素はJuliusのforced alignerで変換するので，`dump/token_list/train_phn_julius`になります。
  - `tts2`だと，音素はpyopenjtalkで変換するので，`dump/token_list/train_phn_pyopenjtalk`になります。

#### モデル学習

```
# Single GPU学習
./run.sh --stage 3 --stop_stage 3 \
    --conf <conf/config.yml> --tag <tag> # 2行目の引数はオプショナル

# Multi GPU学習
./run.sh --stage 3 --stop_stage 3 --n_gpus 4 \
    --conf <conf/config.yml> --tag <tag> # オプション
```

このステージはモデル学習を行います。
- 中間産物（モデル重み，検証セットの生成音声）などは実験フォルダに保存されます。デフォルトのフォルダパスは`exp/train_phn_none_<設定ファイルの名前>`。`--tag`が設定される場合，フォルダのパスは`exp/train_phn_none_<tag>`になります。
- 検証セットの生成音声は`exp/train_phn_none_<設定ファイルの名前>/predictions`に保存されます。
- Multi-nodeの学習は現時点サポートされていません。

#### 推論

```
./run.sh --stage 4 --stop_stage 4 \
  --checkpoint <checkpoint_path>
```

このステージはモデル推論を行います。
- `--checkpoint`引数で指定したモデル重みファイルの評価を行います。引数を指定しないと，自動的に保存時間順で一番遅い重みファイルを使います。
- 生成結果は`results/checkpoint-XXXsteps`に置きます。
  - `wav`に生成した音声サンプルがあります。
  - `outs`に生成したメルスペの図があります。

#### 客観評価

```
./run.sh --stage 5 --stop_stage 5
```

このステージはモデル評価を行います。
- `run.sh`の`eval_metrics`で評価指標を指定することができます。
- 現時点サポートする評価指標は以下になります。
  - `mcd`：Mel cepstrum distortion。これは，正解音声のメルケプストラムとの距離。ついでに，以下の正解音声に基づく指標も計算されます。
    - `F0RMSE`,`F0CORR`：f0のRMSEとCORR(相関係数)
    - `DDUR`：継続長の絶対差
  - `sheet`：[SHEET](https://github.com/unilight/sheet)を用いた自動音声品質主観評価
  - `asr`：[nue-asr](https://huggingface.co/rinna/nue-asr)を用いた音声認識による明瞭度評価
  - `spkemb`：SpeechBrainで公開した[ECAPA-TDNNモデル](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)を用いた話者埋め込みのコサイン類似度。多話者データセットのみ計算できます。

## 謝辞

本リポジトリは以下のリポのもとに作成されました。

- [ESPNet](https://github.com/espnet/espnet)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN/)

## 開発者

Wen-Chin Huang  
名古屋大学　戸田研究室  
E-mail: wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp

Lester Phillip Violeta ([@lesterphillip](https://github.com/lesterphillip))  
名古屋大学　戸田研究室  
E-mail: violeta.lesterphillip@g.sp.m.is.nagoya-u.ac.jp