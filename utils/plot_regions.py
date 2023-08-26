#! /usr/bin/env python3

import json
import matplotlib.pyplot as plt

# English to Slovene City Name Dictionary
city_translations = {
    "Ljubljana": "Ljubljana",
    "Venice": "Benetke",
    "MB": "Maribor",
    "Trieste": "Trst",
    "Zagreb": "Zagreb",
    "Graz": "Gradec",
    "Klagenfurt": "Celovec",
    "Udine": "Videm",
    "Pula": "Pulj",
    "Pordenone": "Pordenone",
    "Szombathely": "Sombotel",
}


def extract_and_translate_city(full_name, translations):
    city_name = full_name.split("_")[1]

    return translations.get(city_name, city_name)


def add_counts_on_bars(bars, ax):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            "{}".format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot(data):
    regions = [
        extract_and_translate_city(region, city_translations) for region in data.keys()
    ]
    green_counts = [data[region]["green-field"] for region in data.keys()]
    building_counts = [data[region]["building"] for region in data.keys()]

    bar_width = 0.35
    index = range(len(regions))

    fig, ax = plt.subplots(figsize=(12, 8))
    bars1 = ax.bar(index, green_counts, bar_width, label="Zelena površina", color="g")
    bars2 = ax.bar(
        [i + bar_width for i in index],
        building_counts,
        bar_width,
        label="Zgradba",
        color="b",
    )

    # Add counts on top of the bars
    add_counts_on_bars(bars1, ax)
    add_counts_on_bars(bars2, ax)

    ax.set_xlabel("Regija")
    ax.set_ylabel("Število")
    ax.set_title("Število zelenih površin proti zgradbam po regijah")
    ax.set_xticks([i + bar_width / 2 for i in index])
    ax.set_xticklabels(regions, rotation=45, ha="right")
    ax.legend()

    plt.savefig("./res/region_structures.png", bbox_inches="tight", dpi=200)


if __name__ == "__main__":
    with open("./res/region_structures.json", "r") as f:
        data = json.load(f)
    plot(data)
