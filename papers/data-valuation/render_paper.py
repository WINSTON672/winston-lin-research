"""Render paper.md to PDF using ReportLab. Run from this file's directory."""
from pathlib import Path
from xml.sax.saxutils import escape
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
root=Path(__file__).resolve().parent
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body',fontName='Times-Roman',fontSize=10.5,leading=14,spaceAfter=8,alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='Equation',fontName='Courier',fontSize=8,leading=11,spaceBefore=6,spaceAfter=10,leftIndent=8))
styles['Title'].fontName='Times-Bold';styles['Title'].fontSize=21;styles['Title'].leading=25
for name in ('Heading1','Heading2'):
 styles[name].keepWithNext=True;styles[name].fontName='Times-Bold';styles[name].textColor=colors.HexColor('#17354a')
styles['Heading1'].fontSize=14;styles['Heading1'].leading=18
styles['Heading2'].fontSize=11.5;styles['Heading2'].leading=15
styles.add(ParagraphStyle(name='Cell',fontName='Times-Roman',fontSize=9,leading=12))

def fmt(s):
 s=escape(s)
 return re.sub(r'(https://[^\s]+)',r'<link href="\1" color="#245b7a">\1</link>',s)

story=[]
for block in (root/'paper.md').read_text().split('\n\n'):
 if not block.strip():continue
 if block.startswith('# '):story.append(Paragraph(fmt(block[2:]),styles['Title']))
 elif block.startswith('## '):story.append(Paragraph(fmt(block[3:]),styles['Heading1']))
 elif block.startswith('### '):story.append(Paragraph(fmt(block[4:]),styles['Heading2']))
 elif block.startswith('    '):story.append(Preformatted('\n'.join(l[4:] for l in block.splitlines()),styles['Equation']))
 elif block.startswith('|'):
  rows=[[Paragraph(fmt(c.strip()),styles['Cell']) for c in l.strip('|').split('|')] for l in block.splitlines() if not re.match(r'^\|[-:| ]+\|$',l)]
  t=Table(rows,repeatRows=1,hAlign='LEFT',colWidths=[468/len(rows[0])]*len(rows[0]))
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef2')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#17354a')),('LINEBELOW',(0,-1),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
  story.extend([t,Spacer(1,10)])
 else:story.append(Paragraph(fmt(block.replace('`','')),styles['Body']))

def page(c,d):
 c.saveState();c.setStrokeColor(colors.HexColor('#c9d3da'));c.line(72,752,540,752)
 c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#51616d'));c.drawString(72,762,'CONTEXTUAL DATASET VALUATION | RESEARCH NOTE')
 c.drawString(72,35,'Winston Zai Lin | October 2026 | Not peer reviewed');c.drawRightString(540,35,str(d.page));c.restoreState()
SimpleDocTemplate(str(root/'paper.pdf'),pagesize=(612,792),rightMargin=72,leftMargin=72,topMargin=57,bottomMargin=55,title='Contextual Dataset Valuation and the Limits of a Universal Price',author='Winston Zai Lin').build(story,onFirstPage=page,onLaterPages=page)
print(root/'paper.pdf')
