from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def save_pdf(patient, diagnosis, confidence, path):
    c = canvas.Canvas(path, pagesize=A4)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "🧠 NeuroScan AI - Alzheimer MRI Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"Patient: {patient}")
    c.drawString(50, 750, f"Diagnosis: {diagnosis}")
    c.drawString(50, 730, f"Confidence: {confidence:.2f}%")
    c.drawString(50, 710, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.save()
    return path
