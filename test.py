
#%%
import numpy as np
import soundfile
import torch


#%%
try :
    import importlib
    importlib.reload(wavmark)
except:
    import wavmark



#%%
# 1.load model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = wavmark.load_model().to(device)


# %%
# 2.create 16-bit payload
payload = np.random.choice([0, 1], size=16)
print("Payload:", payload)


# %%
# 3.read host audio
# the audio should be a single-channel 16kHz wav, you can read it using soundfile:

signal, sample_rate = soundfile.read("whitenoisegaussian.wav")
# Otherwise, you can use the following function to convert the host audio to single-channel 16kHz format:
# from wavmark.utils import file_reader
# signal = file_reader.read_as_single_channel("example.wav", aim_sr=16000)


# %%
# 4.encode watermark
watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=True)

#%% 
# you can save it as a new wav:
soundfile.write("output-whitenoisegaussian.wav", watermarked_signal, 16000)


# %%
# 5.decode watermark
payload_decoded, info, restored_signal  = wavmark.decode_watermark(model, watermarked_signal, show_progress=True)


# %%
BER = (payload != payload_decoded).mean() * 100

print("Decode BER:%.1f" % BER)

print(payload_decoded)








#%% 
# soundfile.write("restored.wav", np.ravel(sample_rate), 16000)
soundfile.write("restored.wav", sample_rate, 16000)

# %%
print(wavmark.__file__)

# %%
import numpy as np
import scipy.io.wavfile as wav

def calculate_snr(signal, noise):
    # Ensure the signal and noise are the same length
    min_len = min(len(signal), len(noise))
    signal = signal[:min_len]
    noise = noise[:min_len]

    # Calculate signal power
    signal_power = np.mean(signal ** 2)

    # Calculate noise power
    noise_power = np.mean(noise ** 2)

    # Calculate SNR in decibels (dB)
    snr = 10 * np.log10(signal_power / noise_power)

    return snr

# Load the audio file
#rate, data = wav.read('')

# Assuming noise is present in the audio, calculate SNR
# You may need to separate the signal and noise yourself
# Here we assume noise is the difference between the signal and original data
# Example: noise = data - clean_signal (clean_signal is your processed or reference signal)
noise = watermarked_signal - np.mean(watermarked_signal)  # Simplified, assuming noise is just the deviation from mean

# Calculate the SNR
snr = calculate_snr(watermarked_signal, noise)
print(f"SNR: {snr:.2f} dB")

# %%
snr = wavmark.metric_util.signal_noise_ratio(signal, watermarked_signal)
print(snr)

# %%
# soundfile.write("new-output.wav", watermarked_signal, 16000)

