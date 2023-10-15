# Mammographic_Density_Classification
Multi-View Mammographic Density Classification by Dilated and Attention-Guided Residual Learning

## Task
Classify mamograms according to four breast tissue density levels:
BI-RADS I: fat breast,
BI-RADS II: fat with some fibroglandular tissue,
BI-RADS III: heterogeneously dense breast,
and BI-RADS IV: extremely dense breast.

## Datasets
The INbreast dataset: 115 patients (410 images). Among the 410 images, 136 belong to BI-RADS I, 147 belong to BI-RADS II, 99 belong to BI-RADS III, and 28 belong to BI-RADS IV.

In-house dataset: 500 patients (1985 images). Among the 1985 images, 319 belong to BI-RADS I (86 patients), 423 belong to BI-RADS II (106 patients), 541 belong to BI-RADS III (133 patients), and 702 belong to BI-RADS IV (175 patients).

## Method
1. Introducing dilated convolutions and attention to the ResNet architecture;
2. For multi-view inputs, building multi-stream feature encoders.

## Results
On the INbreast dataset: accuracy - 70.0%, F1 score - 63.5%, and AUC - 84.7%.
On the in-house dataset: accuracy - 88.7%, F1 score - 87.1%, and AUC - 97.4%.
