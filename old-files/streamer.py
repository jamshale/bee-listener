import pyaudio
import numpy as np

CHUNK = 4410
RATE = 44100

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))

stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True,
                frames_per_buffer=CHUNK)  # uses default input device

for i in range(10):
    data = np.fromstring(stream.read(CHUNK), dtype=np.int16)
    print(data)

stream.stop_stream()
stream.close()
p.terminate()
