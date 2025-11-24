import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from src.data_loader import load_flights
from src.features import add_time_features
from src.analysis import busiest_slots, best_time_windows

def plot_and_save(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    figs = []

    # 1) Hourly average departure delay
    if "dep_delay_min" in df and "dep_hour" in df:
        plt.figure()
        df.groupby("dep_hour")["dep_delay_min"].mean().plot(kind="bar")
        p = os.path.join(out_dir, "avg_dep_delay_by_hour.png")
        plt.title("Average Departure Delay by Hour")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        figs.append(p)

    # 2) Busiest departure slots (top 20)
    if "dep_slot_15m" in df:
        busy = busiest_slots(df, by="departure").head(20)
        plt.figure()
        busy.plot(x="dep_slot_15m", y="flights", kind="bar")
        p = os.path.join(out_dir, "busiest_dep_slots.png")
        plt.title("Busiest Departure 15-min Slots (Top 20)")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        figs.append(p)

    # 3) Best time windows
    best = best_time_windows(df, by="departure", window_minutes=60).head(20)
    if not best.empty:
        plt.figure()
        best.sort_values("avg_delay").head(10).plot(x="scheduled_departure_dt", y="avg_delay", kind="bar")
        p = os.path.join(out_dir, "best_windows.png")
        plt.title("Best 60-min Windows (Lowest Avg Departure Delay)")
        plt.tight_layout()
        plt.savefig(p, dpi=150)
        plt.close()
        figs.append(p)

    return figs

def build_pdf(df: pd.DataFrame, figs: list, out_pdf: str):
    c = canvas.Canvas(out_pdf, pagesize=A4)
    W, H = A4

    # Cover
    c.setFont("Helvetica-Bold", 20)
    c.drawString(2*cm, H-3*cm, "Flight Scheduling & Delay Insights")
    c.setFont("Helvetica", 12)
    c.drawString(2*cm, H-4*cm, "Airports: e.g., Mumbai (BOM)")
    c.drawString(2*cm, H-4.7*cm, f"Rows: {len(df)}")
    c.showPage()

    # Figures pages
    for fig in figs:
        c.drawImage(ImageReader(fig), 2*cm, 5*cm, width=W-4*cm, height=H-10*cm, preserveAspectRatio=True, anchor='c')
        c.showPage()

    # Insights page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, H-3*cm, "Key Insights")
    c.setFont("Helvetica", 11)
    y = H - 4*cm
    bullets = [
        "Best time windows are derived from 60-min buckets minimizing average delay.",
        "Busiest slots (15-min) reveal congestion peaks to avoid when possible.",
        "Use the simulator to quantify impact of shifting specific flights.",
        "Cascade scores highlight rotations that could propagate delays."
    ]
    for b in bullets:
        c.drawString(2.5*cm, y, f"- {b}")
        y -= 1*cm
    c.showPage()
    c.save()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/Flight_Data.xlsx")
    p.add_argument("--out", default="reports/hackathon_report.pdf")
    args = p.parse_args()

    df = load_flights(args.path)
    df = add_time_features(df)
    figs = plot_and_save(df, out_dir="reports/figs")
    build_pdf(df, figs, out_pdf=args.out)
    print(f"PDF written to {args.out}")

if __name__ == "__main__":
    main()
