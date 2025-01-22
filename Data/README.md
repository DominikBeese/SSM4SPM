# Data
This folder contains the following elements:
 * 🗃 [Metadata](cases.json) for all 106 relevant videos
 * 🗃 [Phase annotations](phases.json)
 * 🗃 [Instrument annotations](instruments.json)
 * 🗃 [Anatomy annotations](anatomies.json)
 * 🗃 [Irregularity annotations](irregularities.json)
 * 📂 [Preparation](Preparation) code for downloading and preprocessing the dataset
 * 📂 [Visualization](Visualization) code for generating plots

## Metadata

| Key              | Value   | Description                                     |
| -----------------|---------|-------------------------------------------------|
| `caseId`         | number  | the case id of the video in the [Cataract-1K dataset](https://doi.org/10.1038/s41597-024-03193-4) |
| `frames`         | number  | the number of frames in the 30 fps video        |
| `length`         | number  | the video length in seconds                     |
| `phases`         | boolean | `true` if phase annotations are provided        |
| `instruments`    | boolean | `true` if instrument annotations are provided   |
| `anatomies`      | boolean | `true` if anatomie annotations are provided     |
| `irregularities` | boolean | `true` if irregularity annotations are provided |

## Phase/Instrument/Anatomy annotations

| Key          | Value  | Description                                  |
| -------------|--------|----------------------------------------------|
| `caseId`     | number | the case id of the video in the [Cataract-1K dataset](https://doi.org/10.1038/s41597-024-03193-4) |
| `phaseId`    | number | consecutive number for phases per video      |
| `phase`<br>`instrument`<br>`anatomy` | string | the name of the phase/instrument/anatomy |
| `start`      | number | start frame in the 30 fps video              |
| `end`        | number | end frame in the 30 fps video                |
| `startSec`   | number | start time                                   |
| `endSec`     | number | end time                                     |

## Irregularity annotations

| Key            | Value  | Description                                  |
| ---------------|--------|----------------------------------------------|
| `caseId`       | number | the case id of the video in the [Cataract-1K dataset](https://doi.org/10.1038/s41597-024-03193-4) |
| `irregularity` | string | the name of the irregularity, or `null`      |
