#calculates the average dvoice percentage in a text

def dvoice_percentage(path):
    patterns = ["al", "am", "an", "ar", "el", "em", "en", "er",
                "il", "im", "in", "ir", "ul", "um", "un", "ur",
                "ai", "au", "ei", "ui", "ie", "uo"]
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    total_pairs = len(text) // 2
    count = sum(text.count(p) for p in patterns)
    return count / total_pairs if total_pairs else 0.0

if __name__ == "__main__":
    path = "text.txt"
    print(dvoice_percentage(path))
