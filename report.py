import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = os.path.join("data", "outputs")
REPORT_PATH = os.path.join(OUT_DIR, "customer_review_report.pdf")


def image(name, width=15 * cm):
    path = os.path.join(OUT_DIR, name)
    img = Image(path)
    img.drawHeight = img.drawHeight * width / img.drawWidth
    img.drawWidth = width
    return img


def table(rows):
    t = Table(rows, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#264653')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
    ]))
    return t


def build_report():
    with open(os.path.join(OUT_DIR, "dashboard_data.json"), encoding="utf-8") as f:
        d = json.load(f)
    styles = getSampleStyleSheet()
    ov = d['overview']
    story = [
        Paragraph("Customer Review Intelligence Report", styles['Title']),
        Paragraph(f"Analysis of {ov['total_reviews']:,} Amazon fine food reviews.", styles['Normal']),
        Spacer(1, 12),
        Paragraph("1. Overview", styles['Heading2']),
        table([["Metric", "Value"],
               ["Total reviews", f"{ov['total_reviews']:,}"],
               ["Positive (4-5 stars)", f"{ov['positive']:,}"],
               ["Negative (1-2 stars)", f"{ov['negative']:,}"],
               ["Neutral (3 stars)", f"{ov['neutral']:,}"],
               ["Average rating", ov['average_score']]]),
        Spacer(1, 12),
        image("rating_sentiment_distribution.png"),
        Paragraph("2. What customers talk about", styles['Heading2']),
        Paragraph("Words most specific to negative reviews: " + ", ".join(d['distinctive_words']['negative']), styles['Normal']),
        Spacer(1, 6),
        Paragraph("Words most specific to positive reviews: " + ", ".join(d['distinctive_words']['positive']), styles['Normal']),
        Spacer(1, 8),
        image("wordcloud_negative.png", 12 * cm),
        image("wordcloud_positive.png", 12 * cm),
        PageBreak(),
        Paragraph("3. Topics", styles['Heading2']),
        Paragraph("Complaint themes (LDA)", styles['Heading3']),
    ]
    story += [Paragraph(f"Topic {i + 1}: {', '.join(t)}", styles['Normal']) for i, t in enumerate(d['topics']['negative'])]
    story.append(Paragraph("Praise themes (LDA)", styles['Heading3']))
    story += [Paragraph(f"Topic {i + 1}: {', '.join(t)}", styles['Normal']) for i, t in enumerate(d['topics']['positive'])]
    story += [
        Spacer(1, 10),
        image("top_adjectives.png"),
        PageBreak(),
        Paragraph("4. Sentiment Classification", styles['Heading2']),
        Paragraph(f"Best model: {d['best_model']} (selected on macro F1, because the data is about 84% positive "
                  "and plain accuracy hides weak negative-class performance).", styles['Normal']),
        Spacer(1, 8),
        table([["Model", "Accuracy", "Macro F1", "Negative F1"]] +
              [[m['model'], m['accuracy'], m['macro_f1'], m['negative_f1']] for m in d['model_comparison']]),
        Spacer(1, 8),
        image("classification_confusion_matrix.png", 9 * cm),
        image("top_predictive_words.png", 12 * cm),
        Paragraph("Limitations", styles['Heading3']),
        Paragraph("Labels come from star ratings, not human annotation. Neutral (3 star) reviews were excluded from "
                  "training, so the model tends to treat neutral text as negative.", styles['Normal']),
    ]
    SimpleDocTemplate(REPORT_PATH, pagesize=A4, title="Customer Review Intelligence Report").build(story)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
