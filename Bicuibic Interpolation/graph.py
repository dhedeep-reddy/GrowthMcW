import matplotlib.pyplot as plt


scales = [2,3,4,5,6,7,8,9,10]

forward_v1 = [1.26157, 2.79962, 4.9152, 7.71482, 11.4985, 15.9591, 21.5675, 26.9026, 35.1396]
backward_v1 = [6.11328, 22.0027, 69.7702, 114.34, 194.153, 302.806, 448.082, 641.351, 865.04]


forward_v6 = [1.53736, 3.37101, 5.89333, 9.15169, 13.23862, 18.55490, 23.84286, 32.10538, 40.82612]
backward_v6 = [4.79195, 13.42440, 31.89066, 63.88084, 131.20317, 172.79780, 234.15960, 324.73633, 485.00980]


forward_v7 = [1.54215, 3.37955, 5.91442, 9.19091, 13.27718, 18.40551, 24.30980, 31.52370, 40.44258]
backward_v7 = [3.52826, 5.55669, 12.27196, 16.22655, 29.66015, 40.25507, 60.60650, 69.96120, 95.82546]


plt.figure()
plt.plot(scales, forward_v1, marker='o', label='V1')
plt.plot(scales, forward_v6, marker='s', label='V6')
plt.plot(scales, forward_v7, marker='^', label='V7')

plt.xlabel("Scale")
plt.ylabel("Time (ms)")
plt.title("Forward Pass Time Comparison")
plt.legend()
plt.grid()

plt.show()


plt.figure()
plt.plot(scales, backward_v1, marker='o', label='V1')
plt.plot(scales, backward_v6, marker='s', label='V6')
plt.plot(scales, backward_v7, marker='^', label='V7')

plt.xlabel("Scale")
plt.ylabel("Time (ms)")
plt.title("Backward Pass Time Comparison")
plt.legend()
plt.grid()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(len(scales))
width = 0.25

plt.figure()

plt.bar(x - width, forward_v1, width, label='V1')
plt.bar(x, forward_v6, width, label='V6')
plt.bar(x + width, forward_v7, width, label='V7')

plt.xticks(x, scales)
plt.xlabel("Scale")
plt.ylabel("Time (ms)")
plt.title("Forward Pass Comparison (Bar)")
plt.legend()
plt.grid(axis='y')

plt.show()

plt.figure()

plt.plot(scales, backward_v1, marker='o', label='V1')
plt.plot(scales, backward_v6, marker='s', label='V6')
plt.plot(scales, backward_v7, marker='^', label='V7')

plt.yscale('log')  # 🔥 key change

plt.xlabel("Scale")
plt.ylabel("Time (ms) - Log Scale")
plt.title("Backward Pass (Log Scale)")
plt.legend()
plt.grid()

plt.show()