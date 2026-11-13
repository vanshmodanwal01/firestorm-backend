import glob, os
import data, demo

# Find the Giovanni CSV automatically (the newest .csv in this folder)
csvs = sorted(glob.glob("*.csv"), key=os.path.getmtime, reverse=True)
if not csvs:
    raise SystemExit("No .csv file found in this folder. Put your Giovanni CSV next to run_real.py.")
CSV = csvs[0]
print("Using file:", CSV)

s = data.load_giovanni_csv(CSV)
print(s.head())
print("Number of monthly values:", len(s), "(expect 288 for 2000-2023)")

s = s - 273.15          # Kelvin -> Celsius. Delete this line if you chose Celsius in Giovanni.

os.makedirs("outputs", exist_ok=True)
res = demo.analyze(s, "2-meter air temperature", "Lucknow", "deg C",
                   plot_path="outputs/lucknow_result.png")
print()
print(res["summary"])
print(f"Sen slope per decade: {res['slope_per_decade']:+.3f} deg C | "
      f"Mann-Kendall p = {res['mk_p']:.4f} | significant: {res['significant']}")
print("Plot saved to outputs/lucknow_result.png")