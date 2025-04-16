# 🧬 Character Growth Chart Generator

Generate realistic height growth charts for fictional characters (infant to adult),
based on percentile curves, nationality, growth style, and key life events like illness, malnourishment, or puberty timing.

---

## 📥 Input File (`character_growth_template.csv`)

Fill in the CSV to describe your characters. Leave fields blank to randomise.

| Column              | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `name`              | Character’s name                                                            |
| `sex`               | `male` or `female` (lowercase)                                              |
| `size_class`        | One of: `XXS`, `XS`, `S`, `SM`, `M`, `ML`, `LM`, `L`, `XL`, `XXL`           |
| `birth`             | One of: `premature`, `normal`, `late` (or leave blank for random)           |
| `growth_style`      | One of: `early`, `standard`, `late` (or leave blank for random)             |
| `nation`            | Country name, nationality (`Dutch`, `Japanese`, etc.), or code (`NL`, `GB`) |
| `dip`               | Optional. Use Python-style list of tuples:<br>`[(4.0, 5.0, "illness")]`     |
| `add_random_dips`   | `True` to generate extra random dips (illness, malnutrition, etc.)          |
| `add_random_spurts` | `True` to generate extra random spurts (recovery, growth bursts, etc.)      |

---

## 📤 Output Format

The output CSV includes:
- Character name
- Height at each milestone (birth, 3mo, 6mo, every 6 months up to 20 years)
- A `comment` column with:
  - Nationality + shift
  - Dips (manual + random)
  - Spurts (manual + random)

---

## 🔁 Example Workflow

1. Download and fill in `character_growth_template.csv`
2. Upload it to the generator
3. Get back a fully populated CSV growth chart
4. Use in your story, art, or developmental timeline!

---

## 🧠 Tips

- Random values are **biologically plausible** and reflect population norms.
- Dips and spurts simulate life events (illness, puberty, recovery).
- Nationality and size class shift the percentile to reflect regional averages.

---

For questions or advanced usage, open an issue or start a discussion.
