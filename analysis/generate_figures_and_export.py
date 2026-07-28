"""
Generate all 5 figures, fix section titles, insert figures into docx, save to Desktop
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import nsdecls
from docx.oxml import parse_xml

FIG_DIR = r"d:\MARKETING MIX MODELING\temp_figs"
os.makedirs(FIG_DIR, exist_ok=True)

# ===== COLOR PALETTE =====
C_MER_GEO = "#1B5E9E"      # Deep blue
C_MER_NAT = "#4DA8DA"       # Medium blue
C_ROBYN   = "#E8563A"       # Coral red
C_BG      = "#F7F9FC"       # Light background

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.facecolor': C_BG,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# ============================================================
# FIGURE 1: Experimental Pipeline Diagram
# ============================================================
print("Generating Figure 1: Pipeline diagram...")
fig1, ax1 = plt.subplots(figsize=(10, 5.5))
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 6)
ax1.axis('off')
fig1.patch.set_facecolor('white')

def draw_box(ax, x, y, w, h, text, color='#E3F2FD', ec='#1565C0', fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor=ec, linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            weight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#455A64'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))

# Row 1: Data source
draw_box(ax1, 3.5, 5.0, 3.0, 0.7, "Kaggle FMCG Dataset\n(1,248 obs, 8 regions, 156 weeks)", 
         color='#FFF3E0', ec='#E65100', fontsize=9, bold=True)

# Row 2: Data split
draw_arrow(ax1, 5.0, 5.0, 3.0, 4.5)
draw_arrow(ax1, 5.0, 5.0, 7.0, 4.5)
draw_box(ax1, 1.5, 3.8, 3.0, 0.7, "Geo-Level Format\n(8 regions × 156 weeks)", 
         color='#E8EAF6', ec='#283593', fontsize=8.5)
draw_box(ax1, 5.5, 3.8, 3.0, 0.7, "National-Level Format\n(156 weeks aggregated)", 
         color='#E8EAF6', ec='#283593', fontsize=8.5)

# Row 3: Three configurations
draw_arrow(ax1, 3.0, 3.8, 1.5, 3.2)
draw_arrow(ax1, 7.0, 3.8, 5.0, 3.2)
draw_arrow(ax1, 7.0, 3.8, 8.5, 3.2)

draw_box(ax1, 0.2, 2.5, 2.5, 0.7, "Meridian-Geo\n(10 seeds)", 
         color='#BBDEFB', ec=C_MER_GEO, fontsize=9, bold=True)
draw_box(ax1, 3.7, 2.5, 2.5, 0.7, "Meridian-National\n(10 seeds)", 
         color='#B3E5FC', ec=C_MER_NAT, fontsize=9, bold=True)
draw_box(ax1, 7.2, 2.5, 2.5, 0.7, "Robyn-National\n(10 seeds)", 
         color='#FFCDD2', ec=C_ROBYN, fontsize=9, bold=True)

# Row 4: Evaluation
draw_arrow(ax1, 1.45, 2.5, 3.5, 1.7)
draw_arrow(ax1, 4.95, 2.5, 5.0, 1.7)
draw_arrow(ax1, 8.45, 2.5, 6.5, 1.7)

draw_box(ax1, 2.5, 1.0, 5.0, 0.7, "Unified Evaluation\nMAPE | NRMSE | R² | CV | Wilcoxon | Cohen's d", 
         color='#E8F5E9', ec='#2E7D32', fontsize=9, bold=True)

# Row 5: Outputs
draw_arrow(ax1, 5.0, 1.0, 5.0, 0.5)
draw_box(ax1, 2.0, -0.2, 6.0, 0.65, "Results: Accuracy ↔ Stability ↔ Attribution ↔ Statistical Significance ↔ Sensitivity", 
         color='#FFF9C4', ec='#F57F17', fontsize=8.5, bold=True)

ax1.set_title("Figure 1. Experimental Pipeline: Three-Configuration Comparison", 
              fontsize=11, fontweight='bold', pad=10)
fig1.tight_layout()
fig1.savefig(os.path.join(FIG_DIR, "fig1_pipeline.png"), dpi=200, bbox_inches='tight')
plt.close(fig1)

# ============================================================
# FIGURE 2: ROI Comparison Bar Chart with Error Bars
# ============================================================
print("Generating Figure 2: ROI comparison...")
channels = ['TV', 'Digital', 'Print', 'Radio']
roi_mg = [8.34, 4.12, 1.98, 2.67]
roi_mn = [5.72, 2.48, 1.82, 2.11]
roi_rb = [3.87, 0.27, 5.16, 1.42]
err_mg = [0.08, 0.03, 0.03, 0.04]
err_mn = [0.27, 0.14, 0.11, 0.16]
err_rb = [0.94, 0.25, 1.04, 0.48]

x = np.arange(len(channels))
w = 0.25

fig2, ax2 = plt.subplots(figsize=(9, 5.5))
bars1 = ax2.bar(x - w, roi_mg, w, yerr=err_mg, label='Meridian-Geo', 
                color=C_MER_GEO, edgecolor='white', linewidth=0.8, capsize=4, error_kw={'linewidth':1.2})
bars2 = ax2.bar(x, roi_mn, w, yerr=err_mn, label='Meridian-National', 
                color=C_MER_NAT, edgecolor='white', linewidth=0.8, capsize=4, error_kw={'linewidth':1.2})
bars3 = ax2.bar(x + w, roi_rb, w, yerr=err_rb, label='Robyn-National', 
                color=C_ROBYN, edgecolor='white', linewidth=0.8, capsize=4, error_kw={'linewidth':1.2})

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.3:
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.15, f'{h:.2f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

ax2.set_xlabel('Media Channel', fontsize=12, fontweight='bold')
ax2.set_ylabel('ROI', fontsize=12, fontweight='bold')
ax2.set_title('Figure 2. Estimated ROI by Media Channel Across Three Configurations\n(Error bars = std. dev. across 10 seeds)', 
              fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(channels, fontsize=11)
ax2.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax2.set_ylim(0, 10.5)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig2_roi_comparison.png"), dpi=200, bbox_inches='tight')
plt.close(fig2)

# ============================================================
# FIGURE 3: CV Comparison Bar Chart
# ============================================================
print("Generating Figure 3: CV comparison...")
cv_mg = [1.0, 0.7, 1.5, 1.5]
cv_mn = [4.7, 5.6, 6.0, 7.6]
cv_rb = [24.3, 92.6, 20.2, 33.8]

fig3, ax3 = plt.subplots(figsize=(9, 5.5))
bars1 = ax3.bar(x - w, cv_mg, w, label='Meridian-Geo', color=C_MER_GEO, edgecolor='white', linewidth=0.8)
bars2 = ax3.bar(x, cv_mn, w, label='Meridian-National', color=C_MER_NAT, edgecolor='white', linewidth=0.8)
bars3 = ax3.bar(x + w, cv_rb, w, label='Robyn-National', color=C_ROBYN, edgecolor='white', linewidth=0.8)

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., h + 0.8, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

ax3.set_xlabel('Media Channel', fontsize=12, fontweight='bold')
ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12, fontweight='bold')
ax3.set_title('Figure 3. Coefficient of Variation (CV) of ROI Estimates Across 10 Seeds\n(Lower CV = Higher Stability)', 
              fontsize=11, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(channels, fontsize=11)
ax3.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax3.set_ylim(0, 110)

# Add stability annotation
ax3.axhline(y=5, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax3.text(3.5, 6.5, 'Excellent stability threshold (CV < 5%)', fontsize=8, color='green', alpha=0.7)

fig3.tight_layout()
fig3.savefig(os.path.join(FIG_DIR, "fig3_cv_comparison.png"), dpi=200, bbox_inches='tight')
plt.close(fig3)

# ============================================================
# FIGURE 4: Stacked Bar Chart - Contribution Decomposition
# ============================================================
print("Generating Figure 4: Contribution decomposition...")
configs = ['Meridian-Geo', 'Meridian-National', 'Robyn-National']
baseline = [78.24, 84.63, 91.17]
tv_c =     [7.91, 5.38, 3.42]
dig_c =    [6.67, 3.94, 0.41]
print_c =  [3.56, 3.22, 3.14]
radio_c =  [3.62, 2.83, 1.86]

fig4, ax4 = plt.subplots(figsize=(8, 6))
x4 = np.arange(len(configs))
bw = 0.5

b1 = ax4.bar(x4, baseline, bw, label='Baseline', color='#90A4AE', edgecolor='white')
b2 = ax4.bar(x4, tv_c, bw, bottom=baseline, label='TV', color='#1976D2', edgecolor='white')
b3 = ax4.bar(x4, dig_c, bw, bottom=[baseline[i]+tv_c[i] for i in range(3)], 
             label='Digital', color='#7B1FA2', edgecolor='white')
b4 = ax4.bar(x4, print_c, bw, bottom=[baseline[i]+tv_c[i]+dig_c[i] for i in range(3)], 
             label='Print', color='#388E3C', edgecolor='white')
b5 = ax4.bar(x4, radio_c, bw, bottom=[baseline[i]+tv_c[i]+dig_c[i]+print_c[i] for i in range(3)], 
             label='Radio', color='#F57C00', edgecolor='white')

# Add percentage labels on each segment
for i, config in enumerate(configs):
    vals = [baseline[i], tv_c[i], dig_c[i], print_c[i], radio_c[i]]
    names = ['Baseline', 'TV', 'Digital', 'Print', 'Radio']
    cum = 0
    for j, v in enumerate(vals):
        if v > 2:
            ax4.text(i, cum + v/2, f'{v:.1f}%', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        cum += v

ax4.set_ylabel('Revenue Contribution (%)', fontsize=12, fontweight='bold')
ax4.set_title('Figure 4. Revenue Attribution Decomposition\nAcross Three Configurations', 
              fontsize=11, fontweight='bold')
ax4.set_xticks(x4)
ax4.set_xticklabels(configs, fontsize=10, fontweight='bold')
ax4.legend(loc='upper right', fontsize=9, framealpha=0.9, bbox_to_anchor=(1.18, 1))
ax4.set_ylim(0, 105)
fig4.tight_layout()
fig4.savefig(os.path.join(FIG_DIR, "fig4_contribution.png"), dpi=200, bbox_inches='tight')
plt.close(fig4)

# ============================================================
# FIGURE 5: Radar Chart
# ============================================================
print("Generating Figure 5: Radar chart...")
categories = ['Predictive\nAccuracy', 'Estimation\nStability', 'Media Attribution\nCompleteness', 
              'Computational\nEfficiency', 'Prior\nTransparency']
N = len(categories)

# Scores (0-10 scale)
scores_mg = [9.7, 9.8, 8.5, 7.0, 6.5]  # Meridian-Geo
scores_mn = [6.9, 8.0, 7.0, 8.5, 6.5]  # Meridian-National
scores_rb = [5.8, 3.0, 4.5, 4.0, 9.0]  # Robyn-National

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for s in [scores_mg, scores_mn, scores_rb]:
    s += s[:1]

fig5, ax5 = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
ax5.set_facecolor(C_BG)

ax5.plot(angles, scores_mg, 'o-', linewidth=2.5, color=C_MER_GEO, label='Meridian-Geo', markersize=7)
ax5.fill(angles, scores_mg, alpha=0.15, color=C_MER_GEO)

ax5.plot(angles, scores_mn, 's-', linewidth=2.5, color=C_MER_NAT, label='Meridian-National', markersize=7)
ax5.fill(angles, scores_mn, alpha=0.12, color=C_MER_NAT)

ax5.plot(angles, scores_rb, 'D-', linewidth=2.5, color=C_ROBYN, label='Robyn-National', markersize=7)
ax5.fill(angles, scores_rb, alpha=0.12, color=C_ROBYN)

ax5.set_xticks(angles[:-1])
ax5.set_xticklabels(categories, fontsize=10, fontweight='bold')
ax5.set_ylim(0, 10)
ax5.set_yticks([2, 4, 6, 8, 10])
ax5.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=8, color='gray')
ax5.set_title('Figure 5. Multi-Dimensional Comparison\nAcross Three Configurations', 
              fontsize=12, fontweight='bold', pad=25)
ax5.legend(loc='lower right', bbox_to_anchor=(1.25, -0.05), fontsize=10, framealpha=0.9)

fig5.tight_layout()
fig5.savefig(os.path.join(FIG_DIR, "fig5_radar.png"), dpi=200, bbox_inches='tight')
plt.close(fig5)

print("All 5 figures generated successfully!\n")

# ============================================================
# INSERT FIGURES INTO DOCX
# ============================================================
print("Loading docx...")
doc = Document(r"d:\MARKETING MIX MODELING\ICATSD2026_Final_Format_V3.docx")

# ---- Fix section titles ----
print("Fixing section titles...")
for p in doc.paragraphs:
    if p.text.strip() == "RELATED WORK":
        for run in p.runs:
            run.clear()
        p.runs[0].text = "RELATED STUDIES"
        p.runs[0].font.size = Pt(12)
        p.runs[0].font.bold = True
        p.runs[0].font.name = "Times New Roman"
        print("  Reverted to RELATED STUDIES")
    
    if p.text.strip() == "THEORETICAL FOUNDATIONS":
        for run in p.runs:
            run.clear()
        p.runs[0].text = "THEORETICAL BASIS"
        p.runs[0].font.size = Pt(12)
        p.runs[0].font.bold = True
        p.runs[0].font.name = "Times New Roman"
        print("  Reverted to THEORETICAL BASIS")

# ---- Insert figures ----
figure_map = {
    "[Figure 1.": ("fig1_pipeline.png", "Figure 1. Experimental pipeline diagram for the three-configuration comparison between Meridian-Geo, Meridian-National, and Robyn-National."),
    "[Figure 2.": ("fig2_roi_comparison.png", "Figure 2. Comparison of estimated ROI for 4 media channels across three configurations (error bars represent standard deviation across 10 seeds)."),
    "[Figure 3.": ("fig3_cv_comparison.png", "Figure 3. Coefficient of Variation (CV) of estimated ROI across 10 seeds for three configurations."),
    "[Figure 4.": ("fig4_contribution.png", "Figure 4. Decomposition of revenue attribution across baseline and media channels for the three configurations."),
    "[Figure 5.": ("fig5_radar.png", "Figure 5. Radar chart comparing three configurations across five dimensions."),
}

print("\nInserting figures into document...")
for i, p in enumerate(doc.paragraphs):
    text = p.text.strip()
    for key, (filename, caption) in figure_map.items():
        if text.startswith(key) or key in text:
            img_path = os.path.join(FIG_DIR, filename)
            print(f"  Inserting {filename} at para {i}")
            
            # Clear the placeholder text
            for run in p.runs:
                run.clear()
            
            # Set alignment to center
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add the image
            run = p.add_run()
            run.add_picture(img_path, width=Inches(5.5))
            
            # Add caption below as new run with line break
            run2 = p.add_run("\n" + caption)
            run2.font.size = Pt(10)
            run2.font.italic = True
            run2.font.name = "Times New Roman"
            
            break

# ---- Save to Desktop ----
desktop_path = os.path.expanduser(r"~\Desktop\ICATSD2026_Final_Format_V3.docx")
doc.save(desktop_path)
print(f"\nSaved to: {desktop_path}")

# Also save back to original location
doc.save(r"d:\MARKETING MIX MODELING\ICATSD2026_Final_Format_V3.docx")
print(f"Also saved to original location.")
print("\nDONE!")
