"""
generate_sample_data.py
------------------------
Creates a small demo dataset (data/news.csv) so the whole project
can be run immediately without downloading anything from the internet.

For a real / graded project, replace this with the Kaggle
"Fake and Real News Dataset" (see README.md, Step 5 - Option B).
"""

import pandas as pd
import random

random.seed(42)

# ---- Building blocks used to auto-generate many realistic-looking rows ----

real_subjects = [
    "the central bank", "the Ministry of Finance", "the state government",
    "the World Health Organization", "the United Nations", "the Election Commission",
    "the Reserve Bank", "the Supreme Court", "the Parliament", "local municipal authorities",
    "the Department of Agriculture", "the education ministry", "the transport authority",
    "the health ministry", "the science and technology department"
]

real_actions = [
    "announced a new policy to regulate", "released its quarterly report on",
    "held a press conference regarding", "signed an agreement concerning",
    "published official guidelines for", "approved a budget allocation for",
    "conducted a review of", "issued a statement clarifying",
    "launched an initiative to improve", "confirmed updated statistics on"
]

real_topics = [
    "inflation and interest rates", "public healthcare infrastructure",
    "renewable energy projects", "road safety regulations", "vaccination coverage",
    "agricultural subsidies", "digital literacy programs", "urban water supply",
    "national exam schedules", "climate change mitigation", "employment growth data",
    "foreign trade agreements", "banking sector reforms", "public transport fares",
    "disaster relief funding"
]

fake_hooks = [
    "SHOCKING: Scientists CONFIRM that", "You won't BELIEVE what happens when",
    "BREAKING: Secret government plan reveals", "Doctors HATE this one trick because",
    "LEAKED document PROVES that", "Insider CONFESSES that",
    "This ONE weird fact means", "Experts are TERRIFIED to admit that",
    "URGENT warning: authorities are HIDING that", "Anonymous source claims"
]

fake_claims = [
    "drinking hot water cures every disease overnight",
    "the moon landing footage was filmed in a secret basement",
    "a hidden chip in vaccines controls your thoughts",
    "eating this fruit makes you immortal",
    "the government is replacing money with mind control coins",
    "aliens have already taken over three major cities",
    "a magic pill can make you lose 20kg in two days",
    "5G towers are secretly reading your dreams",
    "a popular celebrity is actually a robot built in a lab",
    "one banned vegetable can instantly cure blindness",
    "banks are planning to delete everyone's savings next week",
    "a miracle spray can make cars run on tap water forever"
]

fake_endings = [
    "Share this before it gets deleted!!!",
    "The mainstream media refuses to report this.",
    "Forward to everyone you know right now.",
    "This is being hidden from the public on purpose.",
    "Wake up and see the truth nobody is telling you.",
    "This changes EVERYTHING you thought you knew."
]


def make_real_row():
    s = random.choice(real_subjects)
    a = random.choice(real_actions)
    t = random.choice(real_topics)
    text = (f"{s.capitalize()} {a} {t} during an official briefing held earlier today. "
            f"Officials said the move is part of an ongoing effort to strengthen "
            f"transparency and improve outcomes over the coming fiscal year. "
            f"A detailed report has been made available on the official website "
            f"for public review and feedback.")
    return text


def make_fake_row():
    hook = random.choice(fake_hooks)
    claim = random.choice(fake_claims)
    end = random.choice(fake_endings)
    text = f"{hook} {claim}. {end}"
    return text


def main(n_per_class: int = 150):
    rows = []
    for _ in range(n_per_class):
        rows.append({"text": make_real_row(), "label": "REAL"})
        rows.append({"text": make_fake_row(), "label": "FAKE"})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.to_csv("data/news.csv", index=False)
    print(f"Saved {len(df)} rows to data/news.csv")
    print(df['label'].value_counts())


if __name__ == "__main__":
    main()
