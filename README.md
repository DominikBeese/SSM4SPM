[![](https://img.shields.io/badge/Python-3.10.6-informational)](https://www.python.org/)
[![](https://img.shields.io/github/license/DominikBeese/FairGer?label=License)](/LICENSE)
# SSM4SPM
<div align="center">
  <h3>Multimodal Surgical Process Modeling With State Space Models<br>(Dominik Beese, 2025)</h3>
  <p>
    Computer-assisted intervention (CAI) uses AI to improve the accuracy, efficiency, and safety of surgical procedures, including pre-operative planning, intra-operative guidance, and post-operative analysis. A modern approach is to use a combination of convolutional neural networks (CNNs) and recurrent neural networks (RNNs) to analyze surgical videos, providing a detailed understanding of phases, instruments and anatomical structures. Recent advances in state-space models (SSMs), such as the Mamba architecture, improve the efficiency of sequence modeling compared to Transformers by reducing computational complexity while maintaining performance. This work applies AI to analyze videos of cataract surgeries and compares CNN-Mamba architectures with other models such as LSTMs, GRUs and Transformers. The results highlight the potential of SSMs for the advancement of multimodal surgical video analysis.
  </p>
  <img src="https://github.com/user-attachments/assets/1f2fc098-b69b-4f92-9167-52f78f913e02" alt="Real-Time Phase Recognition" width="800px">
</div>

## Content
The repository contains the following elements:
 * 📂 [Data](/Data)
   * 🗃 Phase, Instrument, Anatomy, and Irregularity annotations
   * 📂 [Preparation](/Data/Preparation) code for downloading and preprocessing the dataset
   * 📂 [Visualization](/Data/Visualization) code for generating plots
 * 📂 [Code](/Code)
   * 📂 Models
     * 📜 Mamba and Transformer implementation
	 * 📜 CNN-RNN architecture implementation
   * 📂 Training
     * 📜 Callback, Loss, and Metric implementations
	 * 📜 Grid search implementation
 * 📂 [Experiments](/Experiments)
     * 📜 Code for running the experiments
	 * 📂 All experiments
	   * 📂 Data splits
	   * 🖼 Evaluation plots
	   * 📜 Evaluation results
       * 🖼 Analysis plots
