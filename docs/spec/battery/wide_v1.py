"""G126: the wide battery. 60 deterministic items across 10 categories.

Deterministic means every item has an exact checkable answer, scored by substring or
standalone-number match on the post-</think> body. No judge model, no rubric.

Two categories need their design stated because they are easy to fake:

  ADVERSARIAL FALSE-WIN REJECTION -- each item has a PLAUSIBLE WRONG answer that a
  degraded model reaches for. Sydney for Australia's capital, 1912 for when the
  Titanic was launched, Einstein for who discovered penicillin. Scoring accepts the
  true answer and additionally records whether the LURE was emitted, so a battery
  that merely gets easier under compression is distinguishable from one where the
  model falls into a specific trap.

  CALIBRATION -- items with NO answer. A model that fabricates a number scores zero;
  one that declines scores one. This is the only category where refusing is correct,
  and it is included because every other category rewards producing an answer.
"""

def items():
    B = []
    def add(cat, prompt, answer, lure=None):
        B.append({"category": cat, "prompt": prompt, "answer": answer, "lure": lure})

    # 1. arithmetic
    for a, b in ((47, 68), (123, 87), (19, 23), (256, 128)):
        add("arithmetic", f"What is {a} + {b}? Reply with the number only.", str(a + b))
    add("arithmetic", "What is 17 times 19? Reply with the number only.", "323")
    add("arithmetic", "What is 1000 minus 377? Reply with the number only.", "623")

    # 2. code
    add("code", "In Python, what does len([1,2,3]) return? Reply with the number only.", "3")
    add("code", "In Python, what is 7 // 2? Reply with the number only.", "3")
    add("code", "What Python keyword defines a function? Reply with the keyword only.", "def")
    add("code", "In Python, what does 'abc'.upper() return? Reply with the result only.", "ABC")
    add("code", "What is the result of bool([]) in Python? Reply True or False only.", "False")
    add("code", "In Python, what type does range(5) produce when passed to list()? "
                "Reply with the type name only.", "list")

    # 3. reasoning
    add("reasoning", "All cats are mammals. Felix is a cat. Is Felix a mammal? Reply yes or no.", "yes")
    add("reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 "
                     "machines to make 100 widgets? Reply with the number of minutes only.", "5")
    add("reasoning", "A bat and ball cost 1.10 together. The bat costs 1.00 more than the ball. "
                     "How much does the ball cost in cents? Reply with the number only.", "5")
    add("reasoning", "Tom is taller than Sam. Sam is taller than Joe. Who is shortest? "
                     "Reply with the name only.", "Joe")
    add("reasoning", "If today is Monday, what day is it in 10 days? Reply with the day only.", "Thursday")
    add("reasoning", "A rope burns in 60 minutes but not evenly. How many ropes do you need to "
                     "measure 30 minutes by burning from both ends? Reply with the number only.", "1")

    # 4. instruction following
    add("instruction", "Reply with exactly the word BANANA in capital letters and nothing else.", "BANANA")
    add("instruction", "Reply with the third word of this sentence: 'the quick brown fox'. "
                       "Reply with the word only.", "brown")
    add("instruction", "Count the words in 'one two three four five'. Reply with the number only.", "5")
    add("instruction", "Reply with the letters of 'cat' separated by hyphens, lowercase.", "c-a-t")
    add("instruction", "Reply with the number 42 and nothing else.", "42")
    add("instruction", "Write the word 'stop' backwards. Reply with the word only.", "pots")

    # 5. long-form reasoning
    add("longform", "A train leaves at 2pm going 60 km/h. Another leaves the same place at 3pm "
                    "going 90 km/h on the same track. At what hour (24h clock) does the second "
                    "catch the first? Reply with the hour number only.", "5")
    add("longform", "You have 3 boxes: one all apples, one all oranges, one mixed. All labels are "
                    "wrong. What is the minimum number of fruits you must draw to relabel all "
                    "correctly? Reply with the number only.", "1")
    add("longform", "A snail climbs 3m each day and slips 2m each night, in a 10m well. On which "
                    "day does it reach the top? Reply with the day number only.", "8")
    add("longform", "If 5 shirts take 5 hours to dry in the sun, how many hours do 20 shirts take "
                    "drying side by side? Reply with the number only.", "5")
    add("longform", "There are 100 lockers, all closed. Student n toggles every nth locker, for n "
                    "from 1 to 100. How many lockers are open at the end? Reply with the number "
                    "only.", "10")
    add("longform", "Two trains 200 km apart approach at 50 km/h each. A bird flies between them "
                    "at 100 km/h until they meet. How many km does the bird fly? Reply with the "
                    "number only.", "200")

    # 6. tool-like syntax
    add("tool", 'Reply with ONLY a JSON object of the form {"n": 5} where the value is 5.', '"n"')
    add("tool", 'Reply with ONLY a JSON array containing the numbers 1, 2 and 3.', "[1, 2, 3]")
    add("tool", "Reply with ONLY a shell command that lists files in the current directory.", "ls")
    add("tool", "Reply with ONLY the git command that shows the working tree status.", "git status")
    add("tool", 'Reply with ONLY a JSON object of the form {"ok": true}.', '"ok"')
    add("tool", "Reply with ONLY the HTTP method used to create a resource in REST.", "POST")

    # 7. repository engineering
    add("repo", "In a unified diff, what character starts a line that was added? "
                "Reply with the character only.", "+")
    add("repo", "What file lists Python package dependencies by convention? "
                "Reply with the filename only.", "requirements.txt")
    add("repo", "What git command creates a new branch named x and switches to it? "
                "Reply with the command only.", "checkout")
    add("repo", "In semantic versioning 2.4.1, which number is the minor version? "
                "Reply with the number only.", "4")
    add("repo", "What file tells git which paths to ignore? Reply with the filename only.",
        ".gitignore")
    add("repo", "What Cargo file declares a Rust crate's dependencies? Reply with the filename "
                "only.", "Cargo.toml")

    # 8. adversarial false-win rejection
    add("adversarial", "What is the capital of Australia? Reply with the city name only.",
        "Canberra", lure="Sydney")
    add("adversarial", "Who discovered penicillin? Reply with the surname only.",
        "Fleming", lure="Einstein")
    add("adversarial", "What is the largest planet in the Solar System? Reply with the name only.",
        "Jupiter", lure="Saturn")
    add("adversarial", "In which country is the city of Vienna? Reply with the country only.",
        "Austria", lure="Australia")
    add("adversarial", "What is the chemical symbol for potassium? Reply with the symbol only.",
        "K", lure="P")
    add("adversarial", "How many continents are there conventionally? Reply with the number only.",
        "7", lure="5")

    # 9. calibration -- refusing is the correct answer
    for q in ("What is the exact population of the city of Springfield right now?",
              "What did I have for breakfast this morning?",
              "What is the 500th digit of my phone number?",
              "What will the closing price of gold be tomorrow?",
              "What is the name of my next-door neighbour?",
              "How many grains of sand are on Bondi Beach at this instant?"):
        add("calibration", q + " If you cannot know, reply with exactly: UNKNOWN", "UNKNOWN")

    # 10. long-context -- a fact buried in filler
    filler = ("The compiler lowers a representation to an executable. Each route consumes a state "
              "and produces a state. ")
    for i, (key, val) in enumerate((("ALPHA", "raven"), ("BETA", "cobalt"), ("GAMMA", "lantern"),
                                    ("DELTA", "harbor"), ("EPSILON", "quartz"), ("ZETA", "meadow"))):
        pad = filler * 22
        add("longcontext", f"{pad}Remember: the {key} codeword is {val}. {pad}"
                           f"What is the {key} codeword? Reply with the word only.", val)
    return B
