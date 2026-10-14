from pathlib import Path
import sys,json,xml.etree.ElementTree as ET,zipfile
import pymupdf
from PIL import Image,ImageDraw

out=Path(sys.argv[1]);figures=[]
for pdf in sorted(out.glob('fig*.pdf')):
    doc=pymupdf.open(pdf);assert len(doc)==1
    page=doc[0];mm=page.rect.width/72*25.4;assert abs(mm-175)<.1
    spans=[s for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['text'].strip()]
    smallest=min(s['size'] for s in spans);assert smallest>=7.9,(pdf,smallest)
    clipping=[s['text'] for s in spans if s['bbox'][0]<-.5 or s['bbox'][1]<-.5 or s['bbox'][2]>page.rect.width+.5 or s['bbox'][3]>page.rect.height+.5]
    assert not clipping,(pdf,clipping)
    png=out/(pdf.stem+'.png');im=Image.open(png);assert im.width>=4130
    assert im.info.get('dpi',(0,0))[0]>=599
    rendered=out/('qa_pdf_'+pdf.stem+'.png');page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(rendered)
    figures.append({'path':pdf.name,'width_mm':mm,'minimum_font_pt':smallest,'raster_size':list(im.size),'dpi':list(im.info.get('dpi')),'text_clipping':clipping})
    doc.close()
assert len(figures)==8
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(out/'STEH实验核验.xlsx') as z:
    workbook=ET.fromstring(z.read('xl/workbook.xml'));sheets=workbook.findall('s:sheets/s:sheet',ns)
    assert len(sheets)==10,len(sheets)
    sheet_records=[]
    for i,s in enumerate(sheets,1):
        x=ET.fromstring(z.read(f'xl/worksheets/sheet{i}.xml'))
        errors=[c.attrib['r'] for c in x.findall('.//s:c',ns) if c.get('t')=='e'];assert not errors,(s.attrib['name'],errors)
        rows=len(x.findall('s:sheetData/s:row',ns))
        formulas=[f.text for f in x.findall('.//s:f',ns)]
        if s.attrib['name']=='Paired primary':assert rows==524 and sum(str(f).startswith('="') or str(f).startswith('"') for f in formulas)>=521
        if s.attrib['name'] in ['Circuits','Convergence']:assert rows==1093
        sheet_records.append({'name':s.attrib['name'],'rows':rows,'cell_errors':errors})
previews=sorted(out.glob('qa_workbook_*.png'));assert len(previews)==12
for category,paths in [('figures',sorted(out.glob('qa_pdf_fig*.png'))),('workbook',previews)]:
    columns=2;w=1050;h=500;rows=(len(paths)+columns-1)//columns
    sheet=Image.new('RGB',(w*columns,h*rows),'#D0D0D0');draw=ImageDraw.Draw(sheet)
    for i,p in enumerate(paths):
        im=Image.open(p).convert('RGB');im.thumbnail((w-10,h-35));x=(i%columns)*w;y=(i//columns)*h
        draw.text((x+6,y+7),p.stem,fill='black');sheet.paste(im,(x+5,y+30))
    sheet.save(out/f'qa_contact_{category}.png')
record={'status':'PASS','figures':figures,'workbook':sheet_records,'workbook_previews':len(previews),
        'scope':'PDF font/bounds/width, raster dpi, XLSX rows/errors/ID formulas verified. Human-style visual inspection of previews remains required.'}
(out/'artifact-qa.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
