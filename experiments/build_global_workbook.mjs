import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';
const out=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8'));
const wb=Workbook.create();
const results=wb.worksheets.add('Results');
const runs=wb.worksheets.add('Station models');
const trials=wb.worksheets.add('Tuning station fits');
const color='#243A52';
function note(sheet,cell,text,bold=false){sheet.getRange(cell).values=[[text]];sheet.getRange(cell).format.font={name:'Arial',size:10,color:bold?color:'#52616C',bold};}
function table(sheet,row,headers,matrix){
 const r=sheet.getRangeByIndexes(row-1,0,matrix.length+1,headers.length);r.values=[headers,...matrix];
 r.format.font={name:'Arial',size:10,color:'#202B33'};r.format.columnWidth=16;r.format.rowHeight=22;r.format.verticalAlignment='center';
 const h=sheet.getRangeByIndexes(row-1,0,1,headers.length);h.format.fill=color;h.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};h.format.rowHeight=34;h.format.wrapText=true;h.format.horizontalAlignment='center';
 if(matrix.length){
  sheet.getRangeByIndexes(row,0,matrix.length,headers.length).setNumberFormat('0.000');
  if(/^\d{8}$/.test(String(matrix[0][0]))){
   sheet.getRangeByIndexes(row,0,matrix.length,1).formulas=matrix.map(v=>[`="${v[0]}"`]);
   sheet.getRangeByIndexes(row,0,matrix.length,1).setNumberFormat('00000000');
  }
  sheet.getRangeByIndexes(row+matrix.length-1,0,1,headers.length).format.borders={bottom:{style:'thin',color:'#AAB5BF'}};
 }
 sheet.getRangeByIndexes(row-1,0,matrix.length+1,1).format.columnWidth=19;sheet.showGridLines=false;
}
note(results,'A2',`LSTM baseline | ${data.runs.length} stations; shared parameters; seed ${data.seed}`,true);
results.getRange('A2').format.font={name:'Arial',size:14,bold:true,color};
note(results,'A3','One fresh model per station. SD describes between-station variation, not seed uncertainty.');
table(results,5,['Model','Stations','NSE mean','NSE SD','NSE median','KGE mean','KGE SD','KGE median'],data.summary.map(r=>[r.model,r.basins,r.nse_mean,r.nse_sd,r.nse_median,r.kge_mean,r.kge_sd,r.kge_median]));
results.getRange('B6:B8').setNumberFormat('0');
note(results,'A9','All 521 NSE/KGE scores defined. Extreme station 08190500 retained; 13 LSTM NSE scores < 0.');
note(results,'A10','Selected shared hyperparameters | validation-only Bayesian optimization',true);
const selected=data.candidates.find(r=>r.selected);
table(results,12,['Hidden size','Layers','Learning rate','Dropout','Weight decay','Trial index'],[[data.hyperparameters.hidden_size,data.hyperparameters.num_layers,data.hyperparameters.learning_rate,data.hyperparameters.dropout,data.hyperparameters.weight_decay,selected.trial]]);
for(const c of ['A','B','F'])results.getRange(`${c}13`).setNumberFormat('0');
results.getRange('C13:E13').setNumberFormat('0.000000');
note(results,'A15','30 epochs; 15-day history; next-day target; batch 128; seed 42; FP32.');
note(results,'A17',`Current STEH comparison subset | ${data.comparison.length} prespecified stations`,true);
table(results,19,['Basin ID','Test days','LSTM NSE','LSTM KGE','Persistence NSE','AR NSE','RMSE (cfs)','MAE (cfs)'],data.comparison.map(r=>[r.station,r.n,r.nse,r.kge,r.persistence_nse,r.ar_nse,r.rmse,r.mae]));
const comparisonEnd=19+data.comparison.length;
results.getRange(`B20:B${comparisonEnd}`).setNumberFormat('0');results.getRange(`G20:H${comparisonEnd}`).setNumberFormat('#,##0.0');
note(results,'A33','Global candidates | mean validation NSE on the same six random stations',true);
table(results,35,['Trial index','Mean val NSE','Hidden size','Layers','Learning rate','Dropout','Weight decay','Selected'],data.candidates.map(r=>[r.trial,r.mean_validation_nse,r.hidden_size,r.num_layers,r.learning_rate,r.dropout,r.weight_decay,r.selected?'Yes':'No']));
const ce=35+data.candidates.length;
for(const c of ['A','C','D'])results.getRange(`${c}36:${c}${ce}`).setNumberFormat('0');
results.getRange(`E36:G${ce}`).setNumberFormat('0.000000');results.getRange(`H36:H${ce}`).format.horizontalAlignment='center';

const runKeys=['station','seed','n','nse','kge','rmse','mae','bias','hidden_size','num_layers','learning_rate','dropout','weight_decay','best_epoch','best_validation_mse','total_seconds','test_start','test_end'];
note(runs,'A1','Source: seed_metrics.csv; all final station records, independently checked against daily predictions.');
table(runs,3,['Basin ID','Seed','Test days','NSE','KGE','RMSE (cfs)','MAE (cfs)','Bias (cfs)','Hidden size','Layers','Learning rate','Dropout','Weight decay','Best epoch','Best val MSE','Elapsed (s)','Test start','Test end'],data.runs.map(r=>runKeys.map(k=>r[k]??null)));
const re=3+data.runs.length;
for(const c of ['B','C','I','J','N'])runs.getRange(`${c}4:${c}${re}`).setNumberFormat('0');
runs.getRange(`K4:M${re}`).setNumberFormat('0.000000');runs.getRange(`F4:H${re}`).setNumberFormat('#,##0.0');runs.getRange(`P4:P${re}`).setNumberFormat('0.0');
runs.getRange(`Q4:R${re}`).setNumberFormat('yyyy-mm-dd');runs.getRange(`Q4:R${re}`).format.horizontalAlignment='center';
runs.freezePanes.freezeRows(3);runs.freezePanes.freezeColumns(1);
const trialKeys=['station','trial','seed','hidden_size','num_layers','learning_rate','dropout','weight_decay','validation_nse','validation_mse','best_epoch','seconds','selected'];
note(trials,'A1','Source: candidate_trials.csv; each row is one candidate on one random tuning station.');
table(trials,3,['Basin ID','Trial index','Seed','Hidden size','Layers','Learning rate','Dropout','Weight decay','Validation NSE','Validation MSE','Best epoch','Elapsed (s)','Selected'],data.trials.map(r=>trialKeys.map(k=>k==='selected'?(r[k]?'Yes':'No'):(r[k]??null))));
const te=3+data.trials.length;
trials.getRange(`B4:E${te}`).setNumberFormat('0');trials.getRange(`F4:H${te}`).setNumberFormat('0.000000');trials.getRange(`K4:K${te}`).setNumberFormat('0');trials.getRange(`L4:L${te}`).setNumberFormat('0.0');trials.getRange(`M4:M${te}`).format.horizontalAlignment='center';
trials.freezePanes.freezeRows(3);trials.freezePanes.freezeColumns(1);results.tabColor=color;
await wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Results!A5:H8',include:'values,formulas',tableMaxRows:4,tableMaxCols:8})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20}})).ndjson);
for(const [sheet,range,id] of [['Results','A1:H31','results'],['Results',`A33:H${ce}`,'search'],['Station models','A1:H13','models'],['Station models','I3:R13','parameters'],['Tuning station fits','A1:G13','trials'],['Tuning station fits','H3:M13','trial_scores']]){
 const blob=await wb.render({sheetName:sheet,range,scale:1.5});await fs.writeFile(path.join(out,'qa_workbook_'+id+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(path.join(out,'LSTM共享参数基线.xlsx'));
