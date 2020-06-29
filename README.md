# Mono-to-Stereo
This script can transform mono audio files into stereo surround audio files. The stereo spatial audio follows HRTF(Head-related transfer function) and it provides you with a scenario that the source of audio comes from one specific direction over your head and then the source slowly comes to another direction.
The HRIR database `hrir_final.mat` comes from experimental results. It is a two-dimention array holding parameters that used to do convolution with sound data to achieve stereo audios. This MATLAB file concludes data from 72 directions horizontally and 36 directions vertically over head. The angle between two sampling directions is 5 degree. Thanks to the support of this matrix, I'm able to transit mono data to stereo data.
The audio files `Mario.wav` and `Nokia.wav` are examples to demo this.

## Way to do that
There two ways to achieve this effect
1. Do convolution calculation between origin audio and hrir data straightly for both left channel and right channel.
2. Apply fast fourier transform function to hrir data and then do multiplication with origin audio.
