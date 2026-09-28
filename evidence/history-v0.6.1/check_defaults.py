from pathlib import Path
import subprocess,json,os
baseline=Path('/mnt/data/readability-work/original/autoresearch-agent-v0.6')
target=Path('/mnt/data/autoresearch-agent-v0.6-readable')
code='''import json
from arlab.core.config import AgentConfig
from arlab.core.presets import research_budget,adaptive_exploration
from arlab.core.schema import FinalEvaluation,Decision,ResearchContract
from arlab.core.tool_catalog import TOOLS
from arlab.classroom.constants import EXPLORATION_TASKS,D4RL_SUITE9
print(json.dumps({"agent":AgentConfig().model_dump(mode="json"),"budget":research_budget().model_dump(mode="json"),"exploration":adaptive_exploration().model_dump(mode="json"),"final":FinalEvaluation(cases=list(D4RL_SUITE9)).model_dump(mode="json"),"exploration_tasks":EXPLORATION_TASKS,"final_tasks":D4RL_SUITE9,"tool_descriptions":TOOLS,"decision_schema":Decision.model_json_schema(),"contract_schema":ResearchContract.model_json_schema()},sort_keys=True))
'''
values=[]
for root in [baseline,target]:
 env={**os.environ,'PYTHONPATH':str(root/'src'),'PYTHONDONTWRITEBYTECODE':'1'}
 result=subprocess.run(['python','-B','-c',code],env=env,cwd=root,check=True,capture_output=True,text=True,timeout=30)
 values.append(json.loads(result.stdout))
keys=list(values[0]); report={"identical":values[0]==values[1],"fields":{key:values[0][key]==values[1][key] for key in keys},"scope":"Imports existing configuration types; no network, data download, training, or model calls"}
out=Path('/mnt/data/readability-work/evidence');(out/'defaults-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(out/'current-defaults.json').write_text(json.dumps(values[1],ensure_ascii=False,indent=2)+'\n')
assert report['identical'];print(report)
