#!/usr/bin/env python3
"""CPU-only acceptance and summaries for the completed three-arm collapse probe."""
import argparse
import csv
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import struct
import sys

REPO=Path('/Users/ethan/Documents/comllm-edge-scheduler')
ARMS=('baseline','full_support','balanced_prefix')
ARM_UPDATES={'baseline':7,'full_support':6,'balanced_prefix':7}
PROBE_SHA='66615b07692a83221ab33ed25192af38fe1595e3bd3781c7d3a0f90abe8797ad'
FIXED_SHA='31a7a05b849a623931b315ed75d41c822da7638d79c948d41c7603024be57225'
ACCEPT_SHA='71f4ee5603d63ad6a377f5dd568c1f6f355c8f20510d7d2b735b9dea5b4ced94'
ARM_FILES=('config.json','initial_fingerprint.json','initial_probe.json','steps.jsonl','probes.jsonl','summary.json')
EXPECTED_FILES={'preflight.json','summary.json'}|{a+'/'+f for a in ARMS for f in ARM_FILES}|{a+'/panel_greedy_after'+str(u)+'.json' for a in ARMS for u in ((6,) if a=='full_support' else (6,7))}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def load(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def finite(value):return type(value) in (int,float) and math.isfinite(value)
def validate_probe_rows(record,panel,local_token=None,edge_token=None,top_p=.95):
 require(record.get('rng_restored') is True,'probability probe RNG not restored')
 local=record['canonical_local_tokens'];edge=record['canonical_edge_tokens']
 require(local and edge and local[0]!=edge[0],'canonical token IDs invalid')
 if local_token is not None:require(local[0]==local_token and edge[0]==edge_token,'canonical tokens changed')
 require([r['idx'] for r in record['rows']]==[r['_idx'] for r in panel],'probability panel order changed')
 for got,want in zip(record['rows'],panel):
  require(got['best']==want['best'] and got['local_token_id']==local[0] and got['edge_token_id']==edge[0],'panel identity changed')
  require(type(got['local_cut']) is bool,'invalid cut flag')
  for k in ('raw_local_mass','nucleus_local_mass','raw_edge_mass'):
   require(finite(got[k]) and 0<=got[k]<=1,'invalid probability '+k)
  require(finite(got['entropy']) and got['entropy']>=0,'invalid entropy')
  require(len(got['top5_ids'])==len(got['top5_mass'])==5 and len(set(got['top5_ids']))==5,'invalid top5')
  require(all(type(i) is int for i in got['top5_ids']) and all(finite(p) and 0<=p<=1 for p in got['top5_mass']),'invalid top5 values')
  require(got['top5_mass']==sorted(got['top5_mass'],reverse=True) and sum(got['top5_mass'])<=1.000001,'top5 not sorted/normalized')
  if got['local_cut']:require(got['nucleus_local_mass']==0,'cut token has nonzero nucleus mass')
  else:require(got['nucleus_local_mass']+1e-7>=got['raw_local_mass'],'nucleus renormalization inconsistent')
  if top_p==1:require(not got['local_cut'] and abs(got['raw_local_mass']-got['nucleus_local_mass'])<=1e-7,'full support probe is truncated')
 return local[0],edge[0]

def collect(repo,root,probe_source,provenance):
 require(sha(probe_source)==PROBE_SHA,'frozen v5 probe SHA differs')
 old=repo/'runs/20261108_s7_fixed_nolacs';accept_path=old/'fixed_acceptance_20261108T134519.138956Z.json'
 require(sha(accept_path)==ACCEPT_SHA,'completed fixed acceptance SHA differs')
 accepted=read(accept_path);require(accepted['status']=='passed','completed fixed acceptance failed')
 require(sha(repo/'train/s7_train_fixed.py')==FIXED_SHA,'fixed engine SHA differs')
 for name,digest in accepted['identity']['implementation_sha256'].items():require(sha(repo/'train'/name)==digest,'helper SHA differs '+name)
 require(sha(old/'fixed_preflight.json')==accepted['source_files']['fixed_preflight.json']['sha256'],'original environment receipt changed')
 require(sha(old/'nolacs/train_log.jsonl')==accepted['source_files']['nolacs/train_log.jsonl']['sha256'],'original training log changed')
 for version,pin in [('v3','69deede9916a9d7b0be4dc13ced23200398f8b72a981f4c4af1c723b90f4f793'),('v4','35d53f75c7e539479d178225f02811d7655e7f318eea41c3718fc8d21e08a707')]:
  failed=repo/('runs/20261108_s7_collapse_probe_failed_'+version);failed_steps=lines(failed/'replay/steps.jsonl')
  require(len(failed_steps)==2 and [r['update'] for r in failed_steps]==[1,2] and sha(failed/'replay/steps.jsonl')==read(failed/'manifest.json')['files']['replay/steps.jsonl']['sha256'],'prior failed2+2 evidence differs')
  require(read(failed/'preflight.json')['script_sha256']==pin,'prior failed protocol differs')
 manifest=read(provenance);entries={f['relative_path']:f for f in manifest['files']}
 require(EXPECTED_FILES<=set(entries),'download provenance does not cover all25 finished files')
 for rel in EXPECTED_FILES:
  info=entries[rel];p=root/rel
  require(info.get('drive_file_id') and sha(p)==info['local_sha256'] and p.stat().st_size==info['local_bytes'],'source provenance mismatch '+rel)
 sys.path.insert(0,str(repo));S=load(repo/'train/s7_train_fixed.py','_completed_collapse_fixed_source');P=load(probe_source,'_completed_collapse_probe_source')
 from train.grpo_reward import group_advantages
 from train.s7_reward import reward_details
 cfg=S.TrainConfig(base='/content/s7_sft7b',s3_data=str(repo/'data/s3'),s4_data=str(repo/'data/s4'),out='/content/drive/MyDrive/COMLLM/s7_collapse_pair_v5_20261108',group='nolacs',mode='full')
 data=S.load_data(cfg);require(data['audit']==accepted['identity']['data'],'current data differs')
 original,balanced,panel=P.build_prefixes(data['train'],S.training_batches(data['train'],cfg))
 pre=read(root/'preflight.json');summary=read(root/'summary.json')
 require(pre['script_sha256']==PROBE_SHA and pre['source_sha256']==FIXED_SHA and pre['accepted_report_sha256']==ACCEPT_SHA,'probe source lineage differs')
 require(pre['actual_implementation_sha256']==accepted['identity']['implementation_sha256'] and pre['data_audit']==data['audit'],'probe data/helpers differ')
 require(pre['original_prefix']==[r['_idx'] for r in original] and pre['balanced_prefix']==[r['_idx'] for r in balanced] and pre['probe_ids']==[r['_idx'] for r in panel],'prefix/panel construction differs')
 require(pre['arm_updates']==ARM_UPDATES and pre['fair_common_comparison_update']==6 and 'new matched-initialization' in pre['design'],'matched design metadata differs')
 require(all(sum(r['best']==0 for r in balanced[k:k+4])==2 for k in range(0,28,4)),'balanced batches not exactly50% local')
 require(pre['probe_training_overlap']==0 and pre['original_prefix_local_best']==2 and pre['balanced_prefix_local_best']==14 and pre['max_total_updates']==20 and pre['max_seconds']==480 and pre['previous_failed_updates']==4 and pre['whole_investigation_update_ceiling']==24,'bounded protocol differs')
 require(summary['status']=='complete' and summary['total_updates']==20 and finite(summary['invocation_seconds']),'root run not complete20')
 environment=read(old/'fixed_preflight.json')['environment'];expected_env={k:environment[k] for k in ('versions','gpu')}
 old_logs=lines(old/'nolacs/train_log.jsonl');step_rows=[];candidate_rows=[];panel_rows=[];mass_comparison=[];arm_summaries=[];sft_reference_raw=None;initial_reference=None;initial_fingerprint=None;initial_rng=None;local_token=None;edge_token=None;archived_descriptive=[];greedy_rows=[]
 for arm in ARMS:
  folder=root/arm;acfg=read(folder/'config.json');initial=read(folder/'initial_probe.json');steps=lines(folder/'steps.jsonl');probes=lines(folder/'probes.jsonl');arm_summary=read(folder/'summary.json')
  expected_cfg=dataclasses.asdict(cfg);expected_cfg.update(s3_data='/content/comllm_s7_fixed/data/s3',s4_data='/content/comllm_s7_fixed/data/s4')
  if arm=='full_support':expected_cfg['top_p']=1.
  require(acfg==expected_cfg,'arm config differs beyond declared intervention '+arm)
  require(arm_summary['arm']==arm and arm_summary['status']=='complete' and arm_summary['updates']==ARM_UPDATES[arm] and arm_summary['runtime']==expected_env and arm_summary['first_token_probe_scope_only'] is True and arm_summary['new_full_test_run'] is False,'arm incomplete/runtime differs '+arm)
  require(arm_summary==next(s for s in summary['arms'] if s['arm']==arm),'root/arm summaries differ')
  fingerprint=read(folder/'initial_fingerprint.json');require(fingerprint==arm_summary['initial_fingerprint'],'fingerprint/summary differs')
  require(len(fingerprint['lora_A_sha256'])==64 and all(type(fingerprint[k]) is int and fingerprint[k]>0 for k in ('torch_num_threads','torch_num_interop_threads')),'invalid initial fingerprint')
  if initial_fingerprint is None:initial_fingerprint=fingerprint
  require(fingerprint==initial_fingerprint,'fresh adapter initialization/threads differ between arms')
  rng=arm_summary['initial_rng_sha256'];require(len(rng)==64 and all(c in '0123456789abcdef' for c in rng),'invalid initial RNG fingerprint')
  if initial_rng is None:initial_rng=rng
  require(rng==initial_rng and arm_summary['archive_exactness_required'] is False,'initial RNG differs or archive-exact requirement mislabeled')
  require(all(initial.get(k) is True for k in ('passed','initial_lora_B_all_zero','finite','RNG_restored','dropout_zero','base_parameters_frozen','optimizer_lora_only')) and initial['lora_B_nonzero_count']==0,'arm not fresh SFT '+arm)
  if initial_reference is None:initial_reference=initial['reference_logprobs_sha256']
  require(initial['reference_logprobs_sha256']==initial_reference==accepted['policy_probe_before']['reference_logprobs_sha256'],'initial SFT reference differs')
  require([r['update'] for r in steps]==list(range(1,ARM_UPDATES[arm]+1)) and [r['after_update'] for r in probes]==([0,6] if arm=='full_support' else [0,6,7]),'arm update/probability checkpoints differ')
  prefix=balanced if arm=='balanced_prefix' else original
  for u,step in enumerate(steps):
   batch=prefix[u*4:u*4+4];ids=[r['_idx'] for r in batch];require(step['prompt_indices']==ids and step['GT_local_count']==sum(r['best']==0 for r in batch),'observed prefix mismatch')
   for k in ('loss','kl','lr','reward_mean','zero_std_frac','grad_norm_before_clip'):require(finite(step[k]),'nonfinite training metric')
   expected_lr=cfg.lr*.5*(1+math.cos(math.pi*u/208));require(step['lr']==expected_lr,'LR was not scheduled with total_steps208')
   keys=('sampled_texts','sampled_rewards','sampled_reward_actions','sampled_actions','generated_token_ids','completion_lengths_with_eos')
   require(all(len(step[k])==16 for k in keys),'expected16 completions per update')
   adv,zero=group_advantages(step['sampled_rewards'],4);require(zero/4==step['zero_std_frac'],'zero-std mismatch')
   if arm=='baseline':
    old_step=old_logs[u];same=all(step[k]==old_step[k] for k in ('prompt_indices','sampled_texts','sampled_rewards'))
    require(step.get('original_samples_rewards_exact_match') is same,'descriptive archive-exact flag differs')
    archived_descriptive.append({'update':u+1,'same_archived_samples_rewards':same})
   for k in range(16):
    row=batch[k//4];text=step['sampled_texts'][k];details=reward_details(text,row,0.,seed=0,sample_key=(u,row['_idx']))
    require(details['reward']==step['sampled_rewards'][k] and details['action']==step['sampled_reward_actions'][k] and S.parse_prediction(text,row['servers'])==step['sampled_actions'][k],'actual reward/parser mismatch')
    tokens=step['generated_token_ids'][k];require(len(tokens)==step['completion_lengths_with_eos'][k] and 1<=len(tokens)<=15 and all(type(t) is int and t>=0 for t in tokens),'invalid completion token record')
    require(len(tokens)==15 or tokens[-1]==151645,'short completion has no expected EOS')
    candidate_rows.append({'arm':arm,'update':u+1,'idx':row['_idx'],'best':row['best'],'GT':'local' if row['best']==0 else 'offload','completion':k%4,'text':text,'action':details['action'],'reward':details['reward'],'format_reward':details['format_reward'],'current_cost':details['current_cost'],'advantage_float32':f32(adv[k]),'policy_score_coefficient':f32(adv[k])/16,'token_length_with_eos':len(tokens),'generated_token_ids':json.dumps(tokens,separators=(',',':'))})
   this=[r for r in candidate_rows if r['arm']==arm and r['update']==u+1];loc=[r for r in this if r['action']==0]
   step_rows.append({'arm':arm,'update':u+1,'GT_local_prompts':sum(r['best']==0 for r in batch),'local_candidates':len(loc),'local_correct':sum(r['best']==0 for r in loc),'local_positive_adv':sum(r['advantage_float32']>0 for r in loc),'local_negative_adv':sum(r['advantage_float32']<0 for r in loc),'local_zero_adv':sum(r['advantage_float32']==0 for r in loc),'local_policy_score_coefficient_sum':sum(r['policy_score_coefficient'] for r in loc),'mean_completion_tokens_with_eos':statistics.mean(r['token_length_with_eos'] for r in this),'zero_std_frac':step['zero_std_frac'],'kl':step['kl'],'grad_norm_before_clip':step['grad_norm_before_clip']})
  require(arm!='baseline' or arm_summary['archived_candidate_reward_exact']==all(r['same_archived_samples_rewards'] for r in archived_descriptive),'descriptive archived comparison summary differs')
  for record in probes:
   require(record['measurement']==('frozen_SFT_adapter_disabled_after6' if record['after_update']==0 else 'trained_policy'),'measurement timing mislabeled')
   local_token,edge_token=validate_probe_rows(record,panel,local_token,edge_token,acfg['top_p'])
   if record['after_update']==0:
    raw=[(r['raw_local_mass'],r['raw_edge_mass'],r['entropy']) for r in record['rows']]
    if sft_reference_raw is None:sft_reference_raw=raw
    require(all(abs(x-y)<=1e-7 for aa,bb in zip(raw,sft_reference_raw) for x,y in zip(aa,bb)),'arms differ in frozen SFT reference measured after6')
   for r in record['rows']:panel_rows.append({'arm':arm,'source_after_update_tag':record['after_update'],'measured_after_optimizer_updates':6 if record['after_update']==0 else record['after_update'],'measurement':record['measurement'],'GT':'local' if r['best']==0 else 'offload',**r,'top5_ids':json.dumps(r['top5_ids']),'top5_mass':json.dumps(r['top5_mass'])})
   for gt in ('all','local','offload'):
    subset=[r for r in record['rows'] if gt=='all' or (r['best']==0)==(gt=='local')]
    mass_comparison.append({'arm':arm,'source_after_update_tag':record['after_update'],'measured_after_optimizer_updates':6 if record['after_update']==0 else record['after_update'],'measurement':record['measurement'],'GT':gt,'n':len(subset),'raw_local_mean':statistics.mean(r['raw_local_mass'] for r in subset),'raw_local_median':statistics.median(r['raw_local_mass'] for r in subset),'raw_local_min':min(r['raw_local_mass'] for r in subset),'raw_local_max':max(r['raw_local_mass'] for r in subset),'nucleus_local_mean':statistics.mean(r['nucleus_local_mass'] for r in subset),'local_cut_count':sum(r['local_cut'] for r in subset),'entropy_mean':statistics.mean(r['entropy'] for r in subset)})
  for checkpoint in ((6,) if arm=='full_support' else (6,7)):
   greedy=read(folder/('panel_greedy_after'+str(checkpoint)+'.json'));require([r['idx'] for r in greedy]==[r['_idx'] for r in panel],'greedy panel order changed')
   for r,row in zip(greedy,panel):
    require(r['best']==row['best'] and r['pred']==S.parse_prediction(r['text'],row['servers']),'greedy raw/parser mismatch')
    greedy_rows.append({'arm':arm,'after_update':checkpoint,**r})
   arm_steps=[r for r in step_rows if r['arm']==arm and r['update']<=checkpoint]
   first=next(r for r in mass_comparison if r['arm']==arm and r['source_after_update_tag']==0 and r['GT']=='local');last=next(r for r in mass_comparison if r['arm']==arm and r['source_after_update_tag']==checkpoint and r['GT']=='local')
   arm_summaries.append({'arm':arm,'comparison_update':checkpoint,'comparison_set':'fair_after6' if checkpoint==6 else 'extra_after7_baseline_balanced_only','training_GT_local_prompts':sum(r['best']==0 for r in prefix[:checkpoint*4]),'training_local_candidates':sum(r['local_candidates'] for r in arm_steps),'training_correct_local_candidates':sum(r['local_correct'] for r in arm_steps),'training_local_policy_score_coefficient_sum':sum(r['local_policy_score_coefficient_sum'] for r in arm_steps),'panel_GT_local':8,'SFT_reference_local_first_token_mass_mean':first['raw_local_mean'],'trained_local_first_token_mass_mean':last['raw_local_mean'],'trained_over_SFT_reference_mean_ratio':last['raw_local_mean']/first['raw_local_mean'] if first['raw_local_mean'] else None,'final_local_nucleus_cut_count':last['local_cut_count'],'greedy_local_on_GT_local':sum(r['best']==0 and r['pred']==0 for r in greedy),'greedy_local_total':sum(r['pred']==0 for r in greedy),'greedy_correct_all16':sum(r['best']==r['pred'] for r in greedy)})
 return {'status':'complete','checked_utc':datetime.now(timezone.utc).isoformat(),'summary':arm_summaries,'mass_comparison':mass_comparison,'panel_rows':panel_rows,'steps':step_rows,'candidates':candidate_rows,'input_sha256':{rel:sha(root/rel) for rel in sorted(EXPECTED_FILES)},'source_sha256':{'probe':PROBE_SHA,'engine':FIXED_SHA,'fixed_acceptance':ACCEPT_SHA,'postprocessor':sha(__file__),'download_provenance':sha(provenance)},'integrity':{'completed_arms':3,'updates':20,'candidates':320,'probe_rows':128,'fixed_panel_rows':16,'panel_GT_local':8,'panel_GT_offload':8,'panel_prefix_overlap':0,'new_matched_initialization_design':True,'archive_exactness_required':False,'fresh_SFT_before_all_arms':True,'initial_fingerprint_equal_across_arms':True,'previous_failed_updates':4,'whole_investigation_updates':24,'fair_comparison_update':6,'reference_probability_measured_after_update6':True,'all_probability_probe_RNG_flags_true':True},'limitations':['One seed; fair three-arm comparison after6 updates; after7 compares only baseline and balanced;12 training states changed; no full-test/generalization claim.','Raw mass means temperature-scaled pre-nucleus probability of the canonical first token, not total parseable-local answer probability.','Tag0 was measured with adapters disabled after6 updates; it is a frozen SFT reference, not a time-zero observation; trained policies were measured after6 and, for baseline/balanced only, after7.','Panel belongs to training dataset but is excluded from every intervention prefix; it is not a new test set.','Aggregate advantages are sampled policy score coefficients, not complete gradients or parameter/probability changes.'],'role':'new_matched_initialization_diagnostic','is_archived_replay':False,'archived_exact_comparison_descriptive_only':archived_descriptive,'greedy':greedy_rows,'actions':{'cpu_only_postprocessing':True,'model_inference':False,'training':False,'source_artifacts_modified':False}}

def write_outputs(out,result):
 require(not out.exists(),'output exists; refusing overwrite');out.mkdir(parents=True)
 for key in ('summary','mass_comparison','panel_rows','steps','candidates','greedy'):
  with (out/(key+'.csv')).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(result[key][0]));w.writeheader();w.writerows(result[key])
 with (out/'audit.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 report=['# 新的matched-initialization短诊断','', '本轮不是历史归档回放。三组初始LoRA A、CPU/CUDA RNG指纹及环境一致；baseline7/full_support6/balanced7，共20次更新。先前两个exact失败各2步仅计预算，合计24，不用作干预证据。','', '|对照范围|arm|训练local候选|正确local候选|SFT参考首token质量|训练后质量|相对参考比例|被top-p截断/8|GT-local greedy选本地/8|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in sorted(result['summary'],key=lambda x:(x['comparison_update'],ARMS.index(x['arm']))):
  fmt=lambda x:'null' if x is None else f'{x:.8g}'
  report.append('|'+ '|'.join(map(str,[r['comparison_set'],r['arm'],r['training_local_candidates'],r['training_correct_local_candidates'],fmt(r['SFT_reference_local_first_token_mass_mean']),fmt(r['trained_local_first_token_mass_mean']),fmt(r['trained_over_SFT_reference_mean_ratio']),r['final_local_nucleus_cut_count'],r['greedy_local_on_GT_local']]))+'|')
 report+=['','after6才是三组公平比较；after7仅baseline与balanced补充。tag0是在第6次更新后禁用adapter测得的冻结SFT参考，不是训练前的测量，也不表示时间轨迹。这里只测canonical“本地执行”的首token，raw是temperature1.3后的截断前质量，不能等同完整本地回答概率或新测试集准确率。','',*['- '+v for v in result['limitations']],'','历史exact一致性仅作描述，真假均不决定本轮matched对照验收；见audit.json。来源SHA和测量标签随CSV/audit保存。']
 with (out/'report.md').open('x') as f:f.write('\n'.join(report)+'\n')
 with (out/'manifest.json').open('x') as f:json.dump({'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'},'manifest_self_excluded':True},f,indent=2);f.write('\n')
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,default=REPO);p.add_argument('--root',type=Path,default=REPO/'runs/20261108_s7_collapse_pair_v5');p.add_argument('--probe-source',type=Path,default=REPO/'artifacts/s7_collapse_audit_20261108/gpu_probe/collapse_probe.py');p.add_argument('--provenance',type=Path);p.add_argument('--out',type=Path,default=REPO/'artifacts/s7_collapse_audit_20261108/gpu_results_v5');a=p.parse_args(argv)
 result=collect(a.repo,a.root,a.probe_source,a.provenance or a.root/'final_download_provenance.json');write_outputs(a.out,result)
 print(json.dumps({'status':'complete','out':str(a.out),'integrity':result['integrity'],'summary':result['summary']},ensure_ascii=False));return 0
if __name__=='__main__':raise SystemExit(main())
