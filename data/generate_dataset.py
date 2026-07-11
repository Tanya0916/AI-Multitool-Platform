"""
Generates data/sentiment_dataset.csv
A synthetic-but-varied dataset of short text samples labeled
positive / negative / neutral, for demo training of the sentiment model.
Combines subject/aspect templates with sentiment-bearing phrase banks
to create ~600 diverse rows.
"""
import csv
import random

random.seed(42)

subjects = [
    "the product", "this movie", "the service", "the food", "this app",
    "the hotel room", "the customer support", "this book", "the flight",
    "the software update", "the delivery", "this course", "the concert",
    "the phone", "this laptop", "the restaurant", "the team", "the manager",
    "the new policy", "this game", "the website", "the tutorial", "the class",
    "the event", "the presentation", "the meeting", "this recipe", "the car",
    "the neighborhood", "this album",
]

positive_phrases = [
    "exceeded all my expectations",
    "was absolutely fantastic",
    "made me really happy",
    "worked perfectly and saved me a lot of time",
    "was a wonderful experience overall",
    "is honestly one of the best I've come across",
    "left me feeling very satisfied",
    "was smooth, fast, and reliable",
    "impressed everyone in the room",
    "deserves five stars, no question",
    "was thoughtfully designed and easy to use",
    "brought a smile to my face",
    "was well worth the money",
    "turned out better than I imagined",
    "was handled with great care and professionalism",
]

negative_phrases = [
    "was a complete disappointment",
    "left me frustrated and annoyed",
    "did not work as advertised at all",
    "was a total waste of time and money",
    "was poorly organized and confusing",
    "broke down after just one use",
    "was rude and unhelpful",
    "was way below my expectations",
    "felt cheap and poorly made",
    "was delayed with zero communication",
    "made the whole experience miserable",
    "was buggy and kept crashing",
    "was overpriced for what you actually get",
    "was honestly one of the worst I've experienced",
    "left a really bad taste, literally and figuratively",
]

neutral_phrases = [
    "was okay, nothing special",
    "met the basic requirements but nothing more",
    "was about what I expected, no surprises",
    "is fine for occasional use",
    "was average compared to similar options",
    "did the job, though it could be improved",
    "was neither great nor terrible",
    "had some good points and some drawbacks",
    "was acceptable given the price",
    "is a standard option without anything unique",
    "was released on schedule as planned",
    "covers the basics adequately",
    "is similar to what competitors offer",
    "was a routine, unremarkable experience",
    "will probably be updated again next quarter",
]

intensifiers = ["", "Honestly, ", "To be fair, ", "Overall, ", "In my opinion, ", "Frankly, "]

rows = []
for subj in subjects:
    for phrase in positive_phrases:
        text = f"{random.choice(intensifiers)}{subj} {phrase}."
        rows.append((text.strip().capitalize(), "positive"))
    for phrase in negative_phrases:
        text = f"{random.choice(intensifiers)}{subj} {phrase}."
        rows.append((text.strip().capitalize(), "negative"))
    for phrase in neutral_phrases:
        text = f"{random.choice(intensifiers)}{subj} {phrase}."
        rows.append((text.strip().capitalize(), "neutral"))

random.shuffle(rows)

# keep dataset to a reasonable size for a demo (~900 rows)
rows = rows[:900]

with open("data/sentiment_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to data/sentiment_dataset.csv")
