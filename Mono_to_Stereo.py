import numpy as np
import wave, array
import scipy.io as scio
import scipy.io.wavfile as wave1
from scipy import signal as si
import matplotlib.pyplot as plt


# a. Using experimentally measured HRIRs
# 1) Time domain
def load_HRIR1(database):
    """
    Load up Head Related Impulse Response database as in horizontal aspect for both ears.
    :param database: Provided Head Related Impulse Response database as .mat format.
    :return: Array of database.
    """
    # Load up the database
    HRIR = scio.loadmat(database)

    # Import two half circle models of the left ear
    left_1 = list(HRIR['hrir_l'][:, 8, :])
    left_2 = HRIR['hrir_l'][:, 40, :]

    # get left ear horizontal HRIR
    out_left = []
    for i in range(len(left_1) // 2 + 1, len(left_1)):
        out_left.append(left_1[i])
    left_a = []
    for i in range(len(left_2)):
        left_a.append(left_2[-i - 1])
    left_2 = left_a
    for i in range(len(left_2)):
        out_left.append(left_2[i])
    for i in range(len(left_1) // 2 + 1):
        out_left.append(left_1[i])

    # Import two half circle parts of the right ear
    right_1 = list(HRIR['hrir_r'][:, 8, :])
    right_2 = HRIR['hrir_r'][:, 40, :]

    # get right ear horizontal HRIR
    out_right = []
    for i in range(len(right_1) // 2 + 1, len(right_1)):
        out_right.append(right_1[i])
    right_a = []
    for i in range(len(right_2)):
        right_a.append(right_2[-i - 1])
    right_2 = right_a
    for i in range(len(right_2)):
        out_right.append(right_2[i])
    for i in range(len(right_1) // 2 + 1):
        out_right.append(right_1[i])

    # return the array of horizontal sound effect model for both two ears
    out_left = np.array(out_left)
    out_right = np.array(out_right)
    out = []
    for i in range(len(out_left)):
        out.append(out_left[i])
        out.append(out_right[i])
    out = np.array(out)
    return out


def load_HRIR2(database):
    """
    Load up Head Related Impulse Response database as in vertical aspect for both ears.
    :param database: Provided Head Related Impulse Response database as .mat format.
    :return: Array of database.
    """
    # Load up the database
    hrir = scio.loadmat(database)
    # import left vertical HRIR
    left_v = hrir['hrir_l'][:, 24, :]

    # the left HRIR function
    left_a = []
    for i in range(len(left_v)):
        left_a.append(left_v[-i - 1])
    out_left = left_a

    # import right vertical HRIR
    right_v = hrir['hrir_r'][:, 24, :]

    # the right HRIR function
    right_a = []
    for i in range(len(right_v)):
        right_a.append(right_v[-i - 1])
    out_right = right_a

    # the final HRIR function vertical model
    out_left = np.array(out_left)
    out_right = np.array(out_right)
    out = []
    for i in range(len(out_left)):
        out.append(out_left[i])
        out.append(out_right[i])
    out = np.array(out)
    return out


def load_wave(wav, hrir):
    """
    Load up the audio file and cut the wave into appropriate length snippets, the length of one snippets
    and the step length will be returned.
    :param wav: The audio file to be dealt with.
    :param hrir: Array of spatial sound model.
    :return signal: The list of audio data.
    :return points_number: The number of directions of the spatial sound effect.
    :return step: The length of each snippet to be dealt with one direction sound effect.
    """
    sfile = wave.open(wav, 'rb')
    (nchannels, sampwidth, fs, nf, comptype, compname) = sfile.getparams()

    # Compressed not supported yet
    assert comptype == 'NONE'
    array_type = {1: 'B', 2: 'h', 4: 'l'}[sampwidth]
    signal = list(array.array(array_type, sfile.readframes(nf))[::nchannels])

    # the 20 secs signal
    signal = np.array(signal)
    # length of signal
    length = len(signal)
    signal = signal.tolist()
    # number of directions
    points_number = len(hrir) // 2
    # m = ((length // points_number + 1) * points_number - length)
    # if length % points_number != 0:
    #     if m != 0:
    #         signal.append(0)  # zero padding
    #         m -= 1

    # length of signal in each direction
    step = length // points_number
    return signal, points_number, step


def Convolve(signal, hrir, step, points_number):
    """
    Use convolution function to deal with origin audio and spatial sound effect to achieve a stereo effect.
    :param signal: Original audio.
    :param hrir: Array of sound effect.
    :param step: Length of per snippet to be handled.
    :param points_number: The number of directions of the spatial sound effect.
    :return: Synthetic audio file of left channel and right channel.
    """
    # Convolution of two pieces
    segment_L = []
    segment_R = []
    signal = np.array(signal)
    for i in range(points_number):
        segment_L.append(np.convolve(hrir[2 * i], signal[step * i:step * (i + 1)], 'same'))
        segment_R.append(np.convolve(hrir[2 * i + 1], signal[step * i:step * (i + 1)], 'same'))
    return segment_L, segment_R


def Write(name, segment_L, segment_R):
    """
    Write the synthetic stereo effect audio data into the file with .wav format.
    :param name: The name of file.
    :param segment_L: Left channel data.
    :param segment_R: Right channel data.
    """
    # write 2D wave
    left = []
    right = []
    for item in segment_L:
        for sub in item:
            left.append(sub.real)
    for item in segment_R:
        for sub in item:
            right.append(sub.real)
    output = array.array('h', [])[::2]

    for i in range(len(left)):
        output.append(int(right[i] / 2))
        output.append(int(left[i] / 2))

    ofile = wave.open(name, 'w')
    ofile.setparams((2, 2, 44100, len(output), 'NONE', 'NONE'))
    ofile.writeframes(output.tostring())


print('1) Time domain')

hrir_h = load_HRIR1('hrir_final.mat')  # horizontal
signal_h, points_number_h, step_h = load_wave('Mario.wav', hrir_h)
segment_L_h, segment_R_h = Convolve(signal_h, hrir_h, step_h, points_number_h)
Write("timeDomain_horizontal.wav", segment_L_h, segment_R_h)
print('Horizontal sound of time domain have finished.')

hrir_v = load_HRIR2('hrir_final.mat')  # vertical
signal_v, points_number_v, step_v = load_wave('Mario.wav', hrir_v)
segment_L_v, segment_R_v = Convolve(signal_v, hrir_v, step_v, points_number_v)
Write("timeDomain_vertical.wav", segment_L_v, segment_R_v)
print('Vertical sound of time domain have finished.')


# 2) Frequency domain
def freq(signal, hrir):
    s = list(np.array(signal).T)  # signal
    for i in range(len(hrir)):
        s.append(0)
    s = np.array(s)

    number = 2 * (len(s) // len(hrir))  # the size of one piece
    s = s[:number // 2 * len(hrir)]
    x = []
    for i in range(len(hrir) // 2):
        x.append(s[i * number:(i + 1) * number])  # pieces of signal
    hrir = list(hrir)

    left_fft = []
    right_fft = []
    x_new = [[0] * 1 for i in range(len(x))]
    for i in range(len(hrir) // 2):
        x_new[i].pop(0)
        x_new[i] = np.fft.fft(x[i])
        hrir[2 * i] = np.fft.fft(hrir[2 * i], len(x_new[i]))
        hrir[2 * i + 1] = np.fft.fft(hrir[2 * i + 1], len(x_new[i]))
        left_fft.append(np.fft.ifft(np.multiply(x_new[i], hrir[2 * i])).real)  # total of multiply of left channel
        right_fft.append(np.fft.ifft(np.multiply(x_new[i], hrir[2 * i + 1])).real)  # total of multiply of right channel

    left = np.zeros(len(s) + 200 - 1)
    right = np.zeros(len(s) + 200 - 1)
    t = np.zeros(199)
    for i in range(len(left_fft)):
        left_fft[i][0:199] += t[0:199]
        t = left_fft[i][-199:]
        left[i * number:(i + 1) * number] = left_fft[i][0:number].real  # left channel fft
    t = np.zeros(199)
    for i in range(len(right_fft)):
        right_fft[i][0:199] += t[0:199]
        t = right_fft[i][-199:]
        right[i * number:(i + 1) * number] = right_fft[i][0:number].real  # right channel fft

    return left, right


def Write_f(name, left, right):
    output = array.array('h', [])[::2]
    for i in range(len(left)):
        output.append(int(right[i] / 2))
        output.append(int(left[i] / 2))
    # output = np.array(output).T
    ofile = wave.open(name, 'w')
    ofile.setparams((2, 2, 44100, len(output), 'NONE', 'NONE'))
    ofile.writeframes(output.tostring())


print('\n2) Frequency domain')

left_h, right_h = freq(signal_h, hrir_h)
Write_f("freqDomain_horizontal.wav", left_h, right_h)
print('Horizontal sound of frequency domain have finished.')

left_v, right_v = freq(signal_v, hrir_v)
Write_f("freqDomain_vertical.wav", left_v, right_v)
print('Vertical sound of frequency domain have finished.')


# 3) Design part
def design_hrir():
    t = 0.0001
    hrir_l = [[0] * 1 for i in range(36)]
    hrir_r = [[0] * 1 for i in range(36)]
    for theta in range(-90, 90, 5):
        hrir_r[(theta + 90) // 5].pop(0)
        hrir_l[(theta + 90) // 5].pop(0)
        for i in range(-50, 51, 1):
            w = 2 * i * np.pi * 441
            alpha = 0.5 * (1 + np.sin(theta / 180 * np.pi))
            Tr = (1 - alpha) * t
            Tl = alpha * t
            HR = (1 + 2j * alpha * w * t) / (1 + 1j * w * t) * np.exp(-1j * w * Tr)
            HL = (1 + 2j * (1 - alpha) * w * t) / (1 + 1j * w * t) * np.exp(-1j * w * Tl)
            hrir_r[(theta + 90) // 5].append(HR)
            hrir_l[(theta + 90) // 5].append(HL)

        hrir_r[(theta + 90) // 5] = hrir_r[(theta + 90) // 5][51:] + hrir_r[(theta + 90) // 5][:51]
        hrir_l[(theta + 90) // 5] = hrir_l[(theta + 90) // 5][51:] + hrir_l[(theta + 90) // 5][:51]
        # plt.plot(range(len(hrir_l[(theta + 90) // 5])), hrir_l[(theta + 90) // 5])
        # plt.plot(range(len(hrir_r[(theta + 90) // 5])), hrir_r[(theta + 90) // 5])
        # plt.show()
        hrir_r[(theta + 90) // 5] = list(np.fft.ifft(hrir_r[(theta + 90) // 5]).real)
        hrir_l[(theta + 90) // 5] = list(np.fft.ifft(hrir_l[(theta + 90) // 5]).real)

        hrir_r[(theta + 90) // 5] = hrir_r[(theta + 90) // 5][51:] + hrir_r[(theta + 90) // 5][:51]
        hrir_l[(theta + 90) // 5] = hrir_l[(theta + 90) // 5][51:] + hrir_l[(theta + 90) // 5][:51]
        # plt.plot(range(len(hrir_r[(theta + 90) // 5])), hrir_r[(theta + 90) // 5])
        # plt.plot(range(len(hrir_l[(theta+90)//5])), hrir_l[(theta+90)//5])
        # plt.show()
    hrir = []
    for i in range(len(hrir_r)):
        hrir.append(hrir_l[i])
        hrir.append(hrir_r[i])
    return hrir


print('\n3) Design part')
d_hrir = design_hrir()
d_signal, d_points_number, d_step = load_wave("Mario.wav", d_hrir)
design_L, design_R = Convolve(d_signal, d_hrir, d_step, d_points_number)

Write("Design.wav", design_L, design_R)
print('Horizontal sound of design finished.\n')
print('All done.')
