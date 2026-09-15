"""
src/make_synthetic_negatives.py

Generates synthetic MINOR (non-SIF) incident narratives in OSHA abstract style.

=====================  READ THIS  =====================
These rows are SYNTHETIC. They are permitted in TRAINING folds only and are
excluded from every TEST fold by src/train_model_a.py. No reported metric is
ever computed on synthetic data. This is disclosed in reports/limitations.md.
=======================================================

Design goals:
  1. Match OSHA sentence structure ("At <time> on <date>, an employee ...")
     so the model cannot separate classes on writing style alone.
  2. Describe genuinely low-energy events: minor cuts, bruises, sprains,
     eye irritation, slips at same level with no injury, near-misses with
     no energy transfer, housekeeping and ergonomic issues.
  3. Never mention: falls from height, amputation, electrocution, crushing,
     confined space, hospitalization, fracture, unconsciousness.
"""

from pathlib import Path
import random
import pandas as pd

OUT_PATH = Path("data/processed/synthetic_negatives.csv")
N_ROWS = 900
SEED = 42

TIMES = ["7:15 a.m.", "8:40 a.m.", "9:05 a.m.", "10:30 a.m.", "11:50 a.m.",
         "1:20 p.m.", "2:45 p.m.", "3:10 p.m.", "4:35 p.m.", "5:05 p.m."]

DATES = [f"{m} {d}, {y}"
         for m in ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
         for d in [3, 8, 12, 17, 21, 26]
         for y in [2016, 2017]]

WORKERS = ["an employee", "Employee #1", "a maintenance technician",
           "a warehouse associate", "a storekeeper", "an operator",
           "a housekeeping attendant", "an office assistant",
           "a workshop helper", "a contractor's helper"]

# (activity, minor outcome) pairs - all low energy, all non-severe
# HARD NEGATIVES: deliberately reuse the SAME hazard vocabulary as the
# positive class (slip, fall, struck, caught, ladder, machine, cut) but with
# explicitly low-energy, no-harm outcomes. This forces the classifier to learn
# severity rather than topic. Easy negatives (paperwork, canteen, printers)
# are kept only as a minority so the model still sees benign context.
EVENTS = [
    # --- slips / trips / same-level falls, minor outcome ---
    ("slipped on a small patch of spilled water in the corridor and briefly lost balance before steadying against the wall",
     "The employee was uninjured and continued working. The spill was mopped immediately."),
    ("tripped over a trailing extension cable at floor level and stumbled but did not fall",
     "No injury occurred. The cable was rerouted and secured with a floor cover."),
    ("slipped while stepping off a low kerb and twisted the left ankle slightly",
     "An ice pack was applied at the first aid post. The employee walked unaided and returned to duty the same shift."),
    ("lost footing on a wet floor near the wash bay and landed on one knee at the same level",
     "A minor bruise was noted. No treatment beyond first aid was required and no time was lost."),
    ("stumbled on an uneven floor tile in the walkway but recovered balance without falling",
     "No injury resulted. The tile was reported to facilities and repaired the following day."),

    # --- ladders / steps, LOW height, no fall ---
    ("was descending the last two steps of a three-step stepladder when a foot slipped off the tread",
     "The employee landed upright on the floor from a height of under half a metre and was uninjured."),
    ("noticed that a stepladder in the store room had a worn non-slip foot while carrying out a pre-use check",
     "The ladder was tagged out of service and replaced. No one was injured and no work at height took place."),
    ("stepped down from a low platform approximately 40 centimetres high and felt a slight jolt in the knee",
     "No swelling or restriction was observed. The employee declined medical attention."),

    # --- struck by / struck against, low energy ---
    ("was struck on the forearm by a lightweight empty cardboard box that shifted from a shelf at waist height",
     "No injury was sustained. The shelf stacking arrangement was corrected."),
    ("bumped their head lightly against a low-hanging cable tray while walking through the plant room",
     "The employee was wearing a hard hat and was uninjured. The tray was marked with hazard tape."),
    ("was struck against the edge of an open drawer while turning around at the workstation",
     "A small bruise formed on the thigh. First aid was applied and no further treatment was needed."),
    ("had a plastic hand tool fall from a bench onto their boot from a height of about 30 centimetres",
     "Safety footwear prevented any injury. The bench edge guard was refitted."),

    # --- caught / pinched, low energy, no machinery running ---
    ("pinched a finger between two stacked plastic pallets while manually separating them",
     "A minor pinch mark was noted with no break in the skin. The employee continued working."),
    ("caught a glove on a cabinet latch while closing a de-energised control panel door",
     "The glove was snagged but the hand was uninjured. The latch was filed smooth."),
    ("had a shirt cuff briefly catch on the handle of a stationary hand trolley",
     "The trolley was not in motion and no injury occurred. The handle was taped."),

    # --- cuts / abrasions, minor ---
    ("received a shallow cut on the index finger while opening a carton with a safety knife",
     "The cut did not require stitches. It was cleaned and dressed at the first aid post."),
    ("grazed a knuckle against a rough edge while removing a hand-tightened cover plate",
     "A minor abrasion was dressed on site and the edge was deburred."),
    ("sustained a small paper cut while sorting shift handover documents",
     "First aid was applied and the employee resumed duties immediately."),

    # --- machine-adjacent but ISOLATED / not running ---
    ("carried out a visual inspection of a conveyor guard while the conveyor was isolated and locked out",
     "A loose fixing bolt was found and tightened. No injury occurred and no energy was present."),
    ("observed that a machine guard interlock label had faded and was difficult to read",
     "The label was replaced during the next planned shutdown. The machine was not in operation at the time."),
    ("noticed a small oil seep beneath a stationary pump during a routine walk-round",
     "A drip tray was placed and maintenance scheduled the repair. No spill reached the drain and no one was exposed."),

    # --- ergonomic / manual handling, minor ---
    ("felt mild discomfort in the lower back after repeatedly bending to pick up small fittings from a low shelf",
     "The employee rested briefly, reported the ergonomic concern, and resumed light duties."),
    ("felt a mild strain in the shoulder after manually shifting a light toolbox onto a bench",
     "The employee was advised on lifting technique and reported no lasting discomfort."),

    # --- eye / skin irritation, minor ---
    ("got a small amount of dust in the right eye while sweeping the workshop floor",
     "The eye was rinsed at the eyewash station and mild irritation resolved within the hour."),
    ("experienced mild skin irritation on the wrist after wearing a damp glove for an extended period",
     "The glove was replaced with a dry pair and the irritation subsided."),

    # --- pure observations / housekeeping (easy negatives, kept as minority) ---
    ("found an empty cable reel left in a walkway during a routine housekeeping round",
     "The reel was removed to the storage yard. No one was injured."),
    ("noticed that a fluorescent light fitting in the corridor was flickering intermittently",
     "The observation was logged and maintenance replaced the tube the same day. No injury occurred."),
    ("reported that a fire extinguisher inspection tag was overdue by three days",
     "The extinguisher was inspected and re-tagged. It remained fully charged throughout."),
]

FOLLOWUPS = [
    "The supervisor was notified and the observation was entered into the site log.",
    "A toolbox talk reminder was issued at the next shift briefing.",
    "No lost time resulted from this event.",
    "The area was inspected and returned to normal condition.",
    "The employee continued their assigned duties without restriction.",
    "No medical treatment beyond first aid was required.",
    "",
]


def main():
    random.seed(SEED)
    rows, seen = [], set()

    attempts = 0
    while len(rows) < N_ROWS and attempts < N_ROWS * 30:
        attempts += 1
        activity, outcome = random.choice(EVENTS)
        text = (f"At {random.choice(TIMES)} on {random.choice(DATES)}, "
                f"{random.choice(WORKERS)} {activity}. {outcome} "
                f"{random.choice(FOLLOWUPS)}").strip()
        text = " ".join(text.split())
        if text in seen:
            continue
        seen.add(text)
        rows.append(text)

    df = pd.DataFrame({
        "Description": rows,
        "is_sif": 0,
        "source": "SYNTHETIC_NEG",          # <-- used to exclude from test folds
        "Industry Sector": "Others",
        "Employee or Third Party": "Employee",
        "iogp_rule": "Uncategorized",
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Generated {len(df):,} unique synthetic negative narratives.")
    print(f"Saved to {OUT_PATH}\n")
    print("Three samples:")
    for i, t in enumerate(df["Description"].head(3), 1):
        print(f"\n [{i}] {t}")


if __name__ == "__main__":
    main()