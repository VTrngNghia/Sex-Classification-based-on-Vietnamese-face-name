Model A & B only had Dense layers, with very low accuracy, stuck at ~0.80.

Model C adds Conv2D, with better accuracy, but still stuck at 0.93. Accuracy on testing set is lower than on training
set. I suspect overfit but don't know how to measure degree of overfit.

Model D adds more Conv2D layers resulting in much better accuracy (0.9598) but also much slower learning speed. Also,
accuracy seems to flatten out by 7th epoch.

Model E adds MaxPool2D to optimize speed. Accuracy down to 0.9352.

Model F adds Dropout to reduce overfit. Accuracy is slightly better than D: 0.9655.

Model G is a simplified version of VGGNet which I found on a blog. This one seems to bring the best accuracy of 0.9891.
The full model could probably be even better, but training time is too costly (>30min/epoch) for this project.