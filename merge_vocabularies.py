# -*- coding: utf-8 -*-
# ממזג את המילון העברי והארמי לקובץ אחד

import pickle

HEBREW_PATH  = r"D:\פרויקט-לפיני\models\hebrew_words.pkl"
ARAMAIC_PATH = r"D:\פרויקט-לפיני\models\aramaic_words.pkl"
OUTPUT_PATH  = r"D:\פרויקט-לפיני\models\merged_words.pkl"
DEFAULT_BOOST = 500


def merge(boost=DEFAULT_BOOST):
    """ממזג את המילון העברי והארמי לקובץ merged_words.pkl"""
    with open(HEBREW_PATH, "rb") as f:
        hebrew = pickle.load(f)
    with open(ARAMAIC_PATH, "rb") as f:
        aramaic = pickle.load(f)

    # העתקת counts מהעברי
    counts = dict(hebrew["counts"])

    # מיזוג ארמית עם BOOST
    for word, freq in aramaic["counts"].items():
        counts[word] = counts.get(word, 0) + freq * boost

    merged = {
        "counts":          counts,
        "total":           hebrew["total"] + sum(aramaic["counts"].values()) * boost,
        "oov_logprob":     hebrew["oov_logprob"],
        "prefix_discount": hebrew["prefix_discount"],
    }

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(merged, f)

    return len(counts)


# הרצה ישירה מהטרמינל
if __name__ == "__main__":
    total = merge()
    print(f"נשמר: {OUTPUT_PATH}")
    print(f"סה\"כ {total} מילים")
