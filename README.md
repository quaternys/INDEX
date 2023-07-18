# 概要
CS学徒なら当然こいつら全部説明できるよね？？

･･････と，言われたときに迷いなくYESと答えるための知識まとめです．

- 線型代数学
  - 線型空間・線型変換
  - 行列分解：固有値・特異値，NMF
- 解析学
  - 関数論：複素平面，三角関数・指数関数・対数関数
  - 微分積分学：微分，積分，偏微分，微分方程式，線積分
  - 級数展開・変換：Taylor展開，Maclaurin展開，Laurent展開，Fourier展開，Laplace変換，z変換
- 離散数学
  - グラフ理論：最短経路問題，DAG，最大流問題，カット，最小全域木，彩色問題，同相問題
  - グラフ信号処理：中心性，PageRank，スペクトルグラフ理論
  - 推薦システム：協調フィルタリング
- 統計学・データサイエンス
  - 確率分布：分布関数，期待値・統計量，一様分布・正規分布・二項分布・Poisson分布・指数分布・t分布
  - 多変量解析：同時確率・i.i.d.，ベイズ統計，共分散・相関係数・KL情報量，多様体
  - 仮説検定：母平均の差の検定（t検定），正規性の検定，等分散性の検定（F検定），独立性・適合度の検定（カイ二乗検定），母比率の差の検定（z検定）
  - 推定：大数の法則・中心極限定理，信頼区間
- 機械学習
  - クラスター分析：階層型クラスター分析，k-means法・k-means++法，GMM
  - 次元削減：次元の呪い・多重共線性，変数選択（stepwise法），PCA，MDS，t-SNE，Autoencoder
  - 回帰：線形回帰，多項式回帰，Logistic回帰，Lasso回帰，Ridge回帰，Elastic net，RBF
  - 分類：k-NN/k*-NN，LDA，SVM，決定木
  - 評価：決定係数，二乗誤差，混同行列，ROC AUC，交差エントロピー，交差検証，CPCV
  - 最適化：LP/DP，グリッドサーチ，Lagrange未定乗数法，最小二乗法，正則化，勾配降下法(SGD/Momentum/Adagrad/RMSprop/Adam)，バッチ学習，進化計算(GA/CMA-ES/Bi-level GA)
    - 強化学習：Q学習, DQN, Rainbow, AlphaZero, NNUE
    - ゲーム理論：二人零和有限確定完全情報ゲーム，ナッシュ均衡・囚人のジレンマ
- アルゴリズム
  - 探索：線形探索・二分探索，幅優先探索・深さ優先探索，minimax探索・MCTS
  - ソート：選択ソート，バブルソート，クイックソート，マージソート，トポロジカルソート・Kahn's algorithm
  - 数値計算：乱数，連立1次方程式（Gaussの消去法，LU分解法，Gauss-Seidel法，Gauss-Jordan法），数値微分・数値積分
  - 並列計算
  - カオス・フラクタル・シミュレーション
- OS・通信インフラ
  - CPU/GPU/量子コンピューティング，メモリ，OS，機械語・アセンブリ言語・プログラミング
  - インターネット・分散システム：IP，P2P，ビザンチン問題，ブロックチェーン，暗号理論
  - ウェブ：HTML/CSS/JS，データベース(SQL, NoSQL)，スクレイピング
- 信号処理・時系列データ
  - 確率過程
  - 周波数解析：FT, DFT, FFT, STFT, zT, DCT, GFT, JPEG/MPEG, WT, EMD, HHT
  - フィルタリング：ノイズ・品質評価，平均値フィルタ，中央値フィルタ，IIR/FIRフィルタ・畳み込みフィルタ・加重平均値フィルタ，Gaussianフィルタ，バイラテラルフィルタ，非局所平均値フィルタ，周波数フィルタ（ノッチフィルタ・バンドパスフィルタ）
  - 信号源分離：ICA, CSP
  - パターン認識：加算平均，DTW
  - Fintech
- 非構造化データ
  - 画像処理：画像認識，画像生成，輪郭抽出（微分フィルタ，メキシカンハットフィルタ，Gaborフィルタ）
    - モデル：Neocognitron, CNN, AlexNet, ResNet
  - 自然言語処理：言語モデル，ネガポジ判定，文章生成（翻訳，対話）
    - モデル：RNN, LSTM, GRU, Transformer, BERT, GPT
  - 音声処理：音声認識，音声合成
    - モデル：formant, cepstrum, LPC
  - グラフ信号処理：ノード分類，エッジ分類，グラフ分類
    - モデル：Spectral network, ChebNet, GCN, MPNN
  - 生成モデル・マルチモーダルAI・AGI
    - AE, VAE, GAN, Diffusion model, CLIP, Stable Diffusion
    - Seq2seq, Transformer, GPT, ChatGPT
- 認知科学
  - 認知神経科学：Perceptron, BP, HFN/Bortzmann machine, SOM, Neocognitron, SNN, FEP, FC
  - 脳波：律動，ERP (P300, ERD)，BCI
