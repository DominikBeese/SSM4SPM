# Preparation
This folder contains the following elements:
 1. 📜 [Download datasets](1-DownloadDatasets.py)
 2. 📜 [Convert videos and extract frames](2-ConvertVideosExtractFrames.py)
 3. 📜 [(Pre-)compute features](3-ComputeFeatures.py) for two-stage learning

## Download datasets

> [!IMPORTANT]
> You need to enter your personal access token for [Synapse](https://www.synapse.org/) in [line 17](1-DownloadDatasets.py#L17).

## Convert videos and extract frames

The result should look as follows:
 * 📂 Data
   * 📂 0709
     * 🖼️ 00001.png
     * 🖼️ 00002.png
     * ...
   * 📂 0712
   * ...
   * 📽️ 0709.mp4
   * ...
   * 🗃 [anatomies.json](/Data/anatomies.json)
   * 🗃 [cases.json](/Data/cases.json)
   * 🗃 [instruments.json](/Data/instruments.json)
   * 🗃 [irregularities.json](/Data/irregularities.json)
   * 🗃 manifest.json
   * 🗃 [phases.json](/Data/phases.json)
