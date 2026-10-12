import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const out=path.resolve(process.argv[2]);
const data=JSON.parse(await fs.readFile(path.join(out,'workbook_data.json'),'utf8'));
const wb=Workbook.create();
const navy='#243A52';

function sheetTable(name, rows, columns, note){
 if(data.layout_qa)note='INTERNAL LAYOUT QA: partial development records only. Population and confirmation results pending.';
 const sheet=wb.worksheets.add(name);sheet.showGridLines=false;
 sheet.getRange('A1').values=[[note]];
 sheet.getRange('A1').format.font={name:'Arial',size:11,bold:true,color:navy};
 const noteRange=sheet.getRangeByIndexes(0,0,2,columns.length);
 noteRange.merge();noteRange.format.wrapText=true;
 noteRange.format.verticalAlignment='center';noteRange.format.rowHeight=22;
 const range=sheet.getRangeByIndexes(2,0,rows.length+1,columns.length);
 range.values=[columns.map(c=>c[1]),...rows.map(r=>columns.map(c=>r[c[0]]??null))];
 range.format.font={name:'Arial',size:10,color:'#202B33'};
 range.format.columnWidth=18;range.format.rowHeight=22;range.format.verticalAlignment='center';
 const header=sheet.getRangeByIndexes(2,0,1,columns.length);
 header.format.fill=navy;header.format.font={name:'Arial',size:10,bold:true,color:'#FFFFFF'};
 header.format.wrapText=true;header.format.rowHeight=38;
 if(rows.length){
  sheet.getRangeByIndexes(3,0,rows.length,columns.length).setNumberFormat('0.000');
  for(let i=0;i<columns.length;i++){
   const [key,,format]=columns[i];
   if(format)sheet.getRangeByIndexes(3,i,rows.length,1).setNumberFormat(format);
   if(key==='station'){
    sheet.getRangeByIndexes(3,i,rows.length,1).formulas=rows.map(r=>[`="${r.station}"`]);
    sheet.getRangeByIndexes(3,i,rows.length,1).setNumberFormat('00000000');
   }
  }
  sheet.getRangeByIndexes(rows.length+2,0,1,columns.length).format.borders={bottom:{style:'thin',color:'#AAB5BF'}};
 }
 sheet.freezePanes.freezeRows(3);sheet.freezePanes.freezeColumns(1);
 return sheet;
}

const populationSummarySheet=sheetTable('Population summary',data.model_summary,[
 ['measure','Measure','@'],['defined','Defined basins','0'],['mean','Mean'],['median','Median'],
 ['sd_between_basins','Between-basin SD'],['q25','25th percentile'],['q75','75th percentile'],
 ['min','Minimum'],['max','Maximum']],
 'All 521 basins retained. Census descriptions; between-basin SD is not seed uncertainty.');
populationSummarySheet.getRange('A1:A18').format.columnWidth=30;
sheetTable('Paired primary',data.paired_primary,[
 ['station','Basin ID'],['steh_with_q_nse','STEH with Q NSE'],['lstm_nse','LSTM NSE'],
 ['steh_no_q_nse','STEH no Q NSE'],['with_q_minus_lstm','STEH Q - LSTM'],
 ['with_q_minus_no_q','STEH Q - no Q'],['persistence_nse','Persistence NSE'],['ar_nse','AR NSE']],
 'One primary seed42 model per basin and setting; identical test target dates.');
sheetTable('Circuits',data.all_runs,[
 ['station','Basin ID'],['seed','Seed','0'],['with_q','Q history'],['binary_full_nse','Binary full NSE'],
 ['core_nse','Core NSE'],['removed_nse','Removed NSE'],['core_delta_nse','Full - core'],
 ['remove_delta_nse','Full - removed'],['random_win_fraction','Nonidentical-draw win'],
 ['distinct_control_count','Nonidentical draws','0'],['criterion_pass','Fixed criterion pass'],
 ['active_units','Active units','0'],['core_units','Retained units','0']],
 'Core loss <= .05; removal loss >= .02; win >= .8. Draws exclude the core itself and may repeat.');
sheetTable('Convergence',data.convergence_diagnostics,[
 ['station','Basin ID'],['seed','Seed','0'],['with_q','Q history'],
 ['mask_tail_gate_churn','Last 3 gate churn'],['mask_tail_weight_churn','Last 3 weight churn'],
 ['mask_final_near_threshold_fraction','Final gates near .5'],['mask_selected_epoch','Mask best epoch','0'],
 ['mask_best_validation_mse','Best val MSE'],['mask_final_validation_mse','Final val MSE'],
 ['training_seconds','Training seconds','0.0'],['job_seconds','Job seconds','0.0'],
 ['mask_selected_near_threshold_fraction','Selected gates near .5'],
 ['mask_selected_weight_nonzero_fraction','Selected matrix density']],
 'Finite 10-epoch mask budget. Gate churn and threshold proximity do not prove convergence.');
sheetTable('Seed stability',data.stability_basin,[
 ['station','Basin ID'],['with_q','Q history'],['input_rank_mean','Mean rank correlation'],
 ['input_rank_min','Min rank correlation'],['hidden_cka_mean','Mean linear CKA'],['hidden_cka_min','Min linear CKA'],
 ['full_prediction_correlation_mean','Full forecast correlation'],['core_prediction_correlation_mean','Core forecast correlation']],
 '12 new basins; three fixed seeds; three pairs averaged per basin. CKA centered in FP64; constants undefined.');
sheetTable('Confirmation',data.confirmation_basin,[
 ['station','Basin ID'],['with_q','Q history'],['all_three_seeds_pass','All 3 pass'],
 ['seeds_passing','Seeds passing','0'],['nse_mean','3-seed mean NSE'],
 ['nse_sd_fixed_seeds','Fixed-seed NSE SD'],['core_loss_max','Maximum core loss']],
 'Primary LSTM remains single-seed. Confirmation success requires all 3 STEH circuits to pass.');
sheetTable('Proxy probes',data.probe_summary??[],[
 ['with_q','Q history'],['target','Proxy','@'],['representation','Representation','@'],
 ['basins','Basins','0'],['defined','Defined','0'],['mean','Mean R2'],['median','Median R2'],
 ['negative_r2','Negative R2','0'],['failed_or_undefined','Failed or undefined','0']],
 'Input-derived rainfall proxies; decodability only. Unfiltered scores; R2 not independently recomputed.');
const intervalRows=data.quality.confirmation.map(r=>({
 with_q:r.with_q,sample:r.basins,population:r.population,success:r.all_three_seeds_pass,
 lower:r.finite_population_interval.success_fraction_bounds[0],
 upper:r.finite_population_interval.success_fraction_bounds[1]}));
sheetTable('Confirmation interval',intervalRows,[
 ['with_q','Q history'],['sample','Sample basins','0'],['population','Eligible population','0'],
 ['success','All 3 pass basins','0'],['lower','95% lower fraction'],['upper','95% upper fraction']],
 'Exact hypergeometric SRS interval: 503 eligible basins, three fixed seeds; not all initializations.');
let caliperData=null;
try{caliperData=JSON.parse(await fs.readFile(path.join(out,'caliper-audit.json'),'utf8'));}
catch(error){if(error.code!=='ENOENT')throw error;}
if(caliperData){
 sheetTable('Control sensitivity',caliperData.population_summary,[
  ['with_q','Q history'],['caliper','Fixed caliper','0'],['basins','Basins','0'],
  ['nearest_degree_passes','Primary passes','0'],['caliper_passes','Caliper passes','0'],
  ['undefined_control_win','No valid control win','0'],['median_unique_control_masks','Median unique controls','0'],
  ['median_core_overlap','Median core overlap']],
  'Original nearest-degree criterion retained. Both calipers reported; neither chosen using test results.');
 const intervals=caliperData.confirmation.map(r=>({...r,
  lower:r.success_fraction_bounds[0],upper:r.success_fraction_bounds[1]}));
 sheetTable('Caliper confirmation',intervals,[
  ['with_q','Q history'],['caliper','Fixed caliper','0'],['sample_successes','All 3 pass basins','0'],
  ['sample_basins','Sample basins','0'],['population','Eligible population','0'],
  ['lower','98.75% lower fraction'],['upper','98.75% upper fraction']],
  'Conditional SRS: 492 basins. Four Bonferroni intervals have >=95% joint coverage; three fixed seeds.');
}
await wb.recalculate();
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20}})).ndjson);
const previews=[
 ['Population summary','A1:I18','summary'],['Paired primary','A1:H12','paired'],
 ['Circuits','A1:G12','circuits'],['Circuits','H3:N12','criteria'],
 ['Convergence','A1:F12','convergence'],['Convergence','G3:M12','timing'],
 ['Seed stability','A1:H12','stability'],['Confirmation','A1:G12','confirmation'],
 ['Confirmation interval','A1:F6','interval'],['Proxy probes','A1:I15','probes']];
if(caliperData)previews.push(['Control sensitivity','A1:I8','caliper'],
 ['Caliper confirmation','A1:G8','caliper_interval']);
for(const [sheet,range,id] of previews){
 const blob=await wb.render({sheetName:sheet,range,scale:1.5});
 await fs.writeFile(path.join(out,'qa_workbook_'+id+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(out,'STEH实验核验.xlsx'));
