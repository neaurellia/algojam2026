# Library Imports
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

##############################
# Define constants
##############################
DATA_FOLDER = "data/"
OUTPUT_DIR = "./simulation_results"
OUTPUT_FILE = "price_history_plot.png"

BG          = "#FFFFFF"
GRID_COLOR  = "#E0E0E0"
SPINE_COLOR = "#CCCCCC"
TEXT_COLOR  = "#1A1A2E"
LINE_COLOR  = "#1565C0"
##############################


# Load every instrument's price history CSV from the data folder
def load_price_histories(dataFolder):
    priceHistories = {}
    for file in sorted(os.listdir(dataFolder)):
        if file.endswith("_price_history.csv"):
            instrumentName = file.replace("_price_history.csv", "")
            filePath = os.path.join(dataFolder, file)
            priceHistories[instrumentName] = pd.read_csv(filePath)
    return priceHistories


# Plot each instrument's price history as its own subplot, combined into one PNG
def plot_price_histories(priceHistories):
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        9,
        "axes.titlesize":   10,
        "axes.titleweight": "bold",
        "axes.titlepad":    8,
        "axes.labelsize":   8,
        "axes.labelcolor":  TEXT_COLOR,
        "xtick.labelsize":  7,
        "ytick.labelsize":  7,
        "xtick.color":      TEXT_COLOR,
        "ytick.color":      TEXT_COLOR,
    })

    instruments = list(priceHistories.keys())
    numInstruments = len(instruments)
    numCols = 3
    numRows = math.ceil(numInstruments / numCols)

    fig, axes = plt.subplots(
        numRows, numCols, figsize=(6 * numCols, 4 * numRows), facecolor=BG
    )
    axes = axes.flatten()

    for ax, instrument in zip(axes, instruments):
        data = priceHistories[instrument]
        ax.plot(data["Day"], data["Price"], color=LINE_COLOR, linewidth=1.4, zorder=3)
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8, linestyle="-", zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(SPINE_COLOR)
        ax.margins(x=0)
        ax.set_title(instrument)
        ax.set_xlabel("Day")
        ax.set_ylabel("Price")

    # Hide any unused subplot axes (grid may be larger than instrument count)
    for ax in axes[numInstruments:]:
        ax.set_visible(False)

    fig.suptitle("Round 1 Price Histories", fontsize=14, fontweight="bold", color=TEXT_COLOR)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outputPath = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    fig.savefig(outputPath, dpi=300, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined price history plot to {outputPath}")


if __name__ == "__main__":
    priceHistories = load_price_histories(DATA_FOLDER)
    plot_price_histories(priceHistories)
