import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys

def plot_graphic(filename, y_1, y_2, y_3, lr):
    with PdfPages(filename + '.pdf') as pdf:
        fig = plt.figure()

        x = [33, 50, 100]

        plt.style.use('seaborn-whitegrid')
        plt.xlabel("# neurons")
        plt.ylabel("Accuracy")

        plt.plot(x, y_1, marker="o", linestyle='--', color='b', label='learn rate='+lr+'; #iterations=100')

        plt.plot(x, y_2, marker="o", linestyle='--', color='r', label='learn rate='+lr+'; #iterations=1000')

        plt.plot(x, y_3, marker="o", linestyle='--', color='g', label='learn rate='+lr+'; #iterations=10000')

        plt.legend()
        pdf.savefig(fig)  # or you can pass a Figure object to pdf.savefig
        plt.close()


y_1 = [0.2086956522, 0.1434782609, 0.2355072464]

y_2 = [0.2086956522, 0.1434782609, 0.2355072464]

y_3 = [0.2086956522, 0.1434782609, 0.2355072464]

plot_graphic("g1", y_1, y_2, y_3, "0.9")

y_1 = [0.302173913, 0.2688405797, 0.2239130435]

y_2 = [0.302173913, 0.2688405797, 0.2239130435]

y_3 = [0.302173913, 0.2688405797, 0.2239130435]

plot_graphic("g2", y_1, y_2, y_3, "0.05")

y_1 = [0.3173913043, 0.3173913043, 0.2739130435]
y_2 = [0.3123188406, 0.3173913043, 0.2739130435]
y_3 = [0.3123188406, 0.3173913043, 0.2739130435]

plot_graphic("g3", y_1, y_2, y_3, "0.008")
