from pathlib import Path
import ast,json,re,urllib.parse,importlib.util,tempfile,shutil,copy
root=Path('/mnt/data/autoresearch-agent-v0.6-readable')
spec=importlib.util.spec_from_file_location('verify_readability',root/'scripts/verify_readability.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=Path('/mnt/data/autoresearch-agent-v0.6.0.zip')
# A parser-only negative check must reject a real default change, even though
# indentation/comment changes in the delivered tree pass.
original_map=module.load_baseline(base)
with tempfile.TemporaryDirectory(prefix='arlab-readability-') as folder:
 target=Path(folder)
 for rel,data in original_map.items():
  if rel.endswith('.py') or module.protected(rel) or rel.startswith('src/arlab/'):
   p=target/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 positive=module.verify(base,target)
 assert positive['passed']
 p=target/'src/arlab/core/presets.py';text=p.read_text();assert 'max_decisions=20000' in text
 p.write_text(text.replace('max_decisions=20000','max_decisions=20001',1))
 negative=module.verify(base,target)
 assert not negative['passed'] and 'src/arlab/core/presets.py' in negative['ast_failures']
 assert 'src/arlab/core/presets.py' in negative['protected_byte_mismatches']
(root/'evidence/verifier-self-check.json').write_text(json.dumps({'original_tree_accepted':positive['passed'],'changed_budget_rejected':not negative['passed'],'changed_budget_reported_by_ast_and_byte_checks':True,'scope':'temporary copy only; delivery source was not modified'},indent=2)+'\n')
files=[root/'README.md',*sorted((root/'docs').glob('*.md')),root/'configs/README.md',root/'evidence/README.md']
bad=[];total=0
for p in files:
 for m in re.finditer(r'(?<!!)\[[^\]\n]+\]\(([^)\n]+)\)',p.read_text()):
  link=m.group(1).split(' "')[0]
  if re.match(r'^\w+:',link) or link.startswith('#'):continue
  total+=1
  if not (p.parent/urllib.parse.unquote(link.split('#')[0])).exists():bad.append([str(p.relative_to(root)),link])
current=[p for p in files if not p.name.startswith(('README-v','review-v','migration-0.')) and p.name!='deployment-review.md']
forbidden=['单个 D4RL 任务用于探索','不自动把短程权重续训成长程','每个 rung 从相同 seed 重新开始','每个 rung 从相同种子重新训练','最多 4 个算法候选','含 18 个研究动作','研究合同的完整预算沿用 v0.2']
stale=[]
for p in current:
 for token in forbidden:
  if token in p.read_text():stale.append([str(p.relative_to(root)),token])
functions={}
for rel in ['src/arlab/core/harness.py','src/arlab/core/experiments.py','src/arlab/core/context.py','src/arlab/classroom/session.py','src/arlab/audit.py']:
 tree=ast.parse((root/rel).read_text());functions[rel]=[n.name for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
report={'files_checked':len(files),'local_links_checked':total,'missing_local_targets':bad,'current_docs_checked':len(current),'known_stale_assertions':stale,'named_source_entries':functions,'external_links_checked_online':False,'scope':'local file targets and explicit known contradictions; not a complete semantic proof'}
assert not bad and not stale
(root/'evidence/doc-consistency.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
v=json.loads((root/'evidence/validation.json').read_text());v['checks']['document_links']=f'passed_{total}_local_links';v['checks']['known_old_document_claims']='absent_from_current_docs';v['checks']['verifier_negative_test']='detects_default_budget_change';(root/'evidence/validation.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'local_links':total,'missing':bad,'stale':stale,'negative_verifier':True}))
