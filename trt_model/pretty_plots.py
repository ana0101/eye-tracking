import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

def plot_sentence_heatmap(words_dict, sentence_id, cmap='coolwarm'):
    """
    Plots a heatmap over a sentence where each word's background is shaded
    according to its average TRT (Total Reading Time).
    """
    # Extract words and their TRT values
    words = []
    trts = []
    for stimulus in words_dict:
        for word_idx in words_dict[stimulus]:
            word_info = words_dict[stimulus][word_idx]
            if word_info['sentence_id'] == sentence_id:
                words.append(word_info['word'])
                trts.append(word_info['average_TRT'])

    if not words:
        print(f"No data found for sentence_id {sentence_id}")
        return

    # Normalize the TRT values for color mapping
    trts = np.array(trts)
    norm = mcolors.Normalize(vmin=min(trts), vmax=max(trts))

    # Create figure
    fig, ax = plt.subplots(figsize=(min(12, len(words)), 2))
    ax.axis('off')

    # Starting x position
    x = 0.01
    y = 0.5

    for i, word in enumerate(words):
        trt = trts[i]
        color = plt.cm.get_cmap(cmap)(norm(trt))

        # Draw background rectangle
        ax.text(x, y, word, fontsize=12, bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.2'))

        # Estimate width of the word for spacing (roughly)
        word_width = 0.015 * len(word)
        x += word_width + 0.01  # spacing between words

    # Add a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, orientation='horizontal', ax=ax, pad=0.2, aspect=40)
    cbar.set_label('Average TRT')

    plt.title(f"TRT Heatmap for Sentence ID {sentence_id}")
    plt.tight_layout()
    plt.show()


def plot_pred_and_true(words, true_trt, pred_trt, sentence_id=None):
    x = list(range(len(words)))
    plt.figure(figsize=(12, 5))
    plt.plot(x, true_trt, marker='o', label='True TRT', color='blue')
    plt.plot(x, pred_trt, marker='x', label='Predicted TRT', color='orange')
    plt.xticks(ticks=x, labels=words, rotation=45)
    plt.xlabel('Words in Sentence')
    plt.ylabel('TRT')
    title = f'Sentence ID: {sentence_id}' if sentence_id is not None else 'TRT Prediction'
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def barplot_accuracy(models, accuracies_non_embedding, accuracies_embedding, accuracies_all):
    df = pd.DataFrame({
        "Model": models,
        "Non-embedding": accuracies_non_embedding,
        "Embedding": accuracies_embedding,
        "All": accuracies_all
    }).set_index("Model")

    ax = df.plot(kind="bar", figsize=(14, 6), colormap='Set2')
    plt.title("Model Accuracy Comparison Across Feature Sets")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="Feature Set", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
