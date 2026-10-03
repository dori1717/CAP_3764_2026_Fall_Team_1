from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def create_visualizations():
    df = pd.read_csv("data/student-por-clean.csv")

    output_dir = Path("outputs/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chart 1: Final grade distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["G3"], bins=range(0, 21), edgecolor="black")
    plt.title("Distribution of Final Grades")
    plt.xlabel("Final Grade (G3)")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(output_dir / "final_grade_distribution.png", dpi=150)
    plt.close()

    # Chart 2: Grades by study time
    study_levels = sorted(df["studytime"].unique())
    groups = [
        df.loc[df["studytime"] == level, "G3"]
        for level in study_levels
    ]

    plt.figure(figsize=(8, 5))
    plt.boxplot(groups, tick_labels=[str(x) for x in study_levels])
    plt.title("Final Grades by Study Time")
    plt.xlabel("Study Time Level")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()
    plt.savefig(output_dir / "grades_by_studytime.png", dpi=150)
    plt.close()

    # Chart 3: Second-period grade vs final grade
    plt.figure(figsize=(8, 5))
    plt.scatter(df["G2"], df["G3"], alpha=0.6)
    plt.title("Second-Period Grades vs Final Grades")
    plt.xlabel("Second-Period Grade (G2)")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()
    plt.savefig(output_dir / "g2_vs_g3.png", dpi=150)
    plt.close()

    # Chart 4: Absences vs final grade
    plt.figure(figsize=(8, 5))
    plt.scatter(df["absences"], df["G3"], alpha=0.6)
    plt.title("School Absences vs Final Grades")
    plt.xlabel("Number of Absences")
    plt.ylabel("Final Grade (G3)")
    plt.tight_layout()
    plt.savefig(output_dir / "absences_vs_g3.png", dpi=150)
    plt.close()

    print("Successfully created 4 visualizations!")
    print(f"Charts saved in: {output_dir}")


if __name__ == "__main__":
    create_visualizations()
