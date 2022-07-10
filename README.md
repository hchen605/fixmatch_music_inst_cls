# fixmatch_music_inst_cls

## Problem formulation

Challenge of collecting large-scale datasets  -> Weakly labeled dataset
- Missing labels can mean either present or absent instances  -> problematic!

No conditioning
- Timing, pitch, ...etc

How to detect and classify with few knowledge?​
- Semi-supervised learning (SSL) approach to handle missing labels​
- Self-supervised learning?

## OpenMIC Dataset

The first open, large-scale, multi-instrument music dataset
- 20000 audio clips, each of 10s length -> diverse set of musical instruments and genres
- 20 musical instruments class
- No time stamp for instruments
- Weakly labeled -> 90% of the labels are missing

## Baseline system


<img src="https://github.com/hchen605/fixmatch_music_inst_cls/blob/master/fig/bs.png" width="6000" height="500" />

## FixMatch

Applying in audio classification task
- VGGish features as input
- Pseudo-label

<img src="https://github.com/hchen605/fixmatch_music_inst_cls/blob/master/fig/fix_2.png" width="2000" height="400" />

Data augmentation
- VGGish features masking
- Audio effects

<img src="https://github.com/hchen605/fixmatch_music_inst_cls/blob/master/fig/data_aug.png" width="500" height="200" />


## Evaluation

Macro F1 score = 81.3% 

<img src="https://github.com/hchen605/fixmatch_music_inst_cls/blob/master/fig/f1.png" width="500" height="400" />

F1 score vs Dataset label distribution
- Fewer pos labels would result in low F1 score
- More neg labels not helpful
- FixMatch improve few pos label instrument

<img src="https://github.com/hchen605/fixmatch_music_inst_cls/blob/master/fig/f1_data.png" width="1000" height="350" />
