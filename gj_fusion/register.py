"""Register only sealed, independently verified G/J evaluations."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from b1_structure_fusion.register import _atomic_write, _json_bytes, _lock, _relative, _complete
from results_catalog import build as catalog
from wbm_retrieval.common import digest, file_hash, read_json
from f_region_structure.state import TZ
from . import scoring, selection
from .run import VERSION, code_identity, read_frozen, check_binding
from .pipeline import KS, variant_signature
from .verify import REPORT_FILES
from .state import read_signed, verified


def proposal(output,*,root=ROOT):
    output,root=Path(output).resolve(),Path(root).resolve()
    _relative(output,root/'artificial')
    manifest=read_signed(output/'run_manifest.json')
    context=manifest['context']
    split=manifest.get('split')
    signature=digest(context)
    status=read_json(output/'status.json')
    if (manifest.get('version')!=VERSION or split not in ('valid','test')
            or context.get('split')!=split or context.get('code')!=code_identity()
            or status.get('status')!='completed' or status.get('context_signature')!=signature
            or status.get('test_evaluated')!=(split=='test') or not verified(output,signature)):
        raise ValueError('Only completed matching G/J stages may be registered')
    check_binding(context['source_binding'])
    valid_root=output if split=='valid' else Path(manifest['valid_run'])
    frozen,valid_manifest=read_frozen(valid_root)
    choice=output/('selection.json' if split=='valid' else 'frozen_selection.json')
    if read_signed(choice)!=frozen:
        raise ValueError('Frozen selection copy differs from the sealed valid decision')
    if split=='test':
        if (context['frozen_selection_signature']!=frozen['signature']
                or manifest['source_run']!=valid_manifest['source_run']
                or datetime.fromisoformat(manifest['started_at'])<datetime.fromisoformat(frozen['frozen_at'])):
            raise ValueError('Test source, selection or actual start precedes valid freeze')
    specs=scoring.frozen_grid() if split=='valid' else frozen['selected_specs']
    if context.get('grid')!=specs or context.get('full_grid')!=scoring.frozen_grid():
        raise ValueError('Stage contains an unexpected candidate set')
    index=read_json(output/f'evaluation_index_{split}.json')
    if (index.get('context_signature')!=signature or index.get('variants')!=specs
            or set(index.get('results',{}))!=set(specs)):
        raise ValueError('Incomplete or changed stage evaluation index')
    if read_json(output/'source_protection_after.json').get('status')!='unchanged':
        raise ValueError('Historical source protection failed')
    replay=read_json(output/'baseline_replay.json')
    if replay.get('status')!='verified' or set(replay.get('controls',{}))!={'J_only','G_only','product'}:
        raise ValueError('Original controls were not replayed')
    nquery,ngallery=(500,3599) if split=='valid' else (7199,7199)
    for name,control in replay['controls'].items():
        source=control.get('source')
        if (control.get('top500_identical') is not True or control.get('num_queries')!=nquery
                or not isinstance(source,str) or source not in context['source_binding']
                or control.get('source_sha256')!=context['source_binding'][source]
                or control.get('new_sha256')!=file_hash(output/split/name/'rankings.csv')):
            raise ValueError('Historical control replay is missing or stale')
        error=control.get('max_score_error')
        if name!='J_only' and (not isinstance(error,(int,float)) or not 0<=error<=2e-7):
            raise ValueError('Historical control scores differ beyond recorded backend tolerance')
    audit=read_json(output/f'component_audit_{split}.json')
    if (audit.get('status')!='verified' or audit.get('split')!=split
            or audit.get('context_signature')!=signature or audit.get('num_queries')!=37
            or len(set(audit.get('query_ids',[])))!=37 or audit.get('candidate_pairs')!=740
            or audit.get('max_candidates_per_query')!=20
            or not 0<=audit.get('max_global_absolute_error',float('inf'))<=2e-7
            or not 0<=audit.get('max_joint_distance_absolute_error',float('inf'))<=1e-12):
        raise ValueError('Independent source-component audit is incomplete or invalid')
    runs=[]
    for name,spec in specs.items():
        folder=output/split/name
        sig=variant_signature(signature,split,name,spec)
        marker=_complete(folder,sig)
        checked=read_json(folder/'verification.json')
        row=index['results'][name]
        ev=read_json(folder/'evaluation_manifest.json')
        ident=read_json(folder/'run_identity.json')
        if (row.get('signature')!=sig or row.get('completed_sha256')!=file_hash(marker)
                or ev.get('signature')!=sig or ev.get('context_signature')!=signature
                or ev.get('spec')!=spec or ev.get('split')!=split or ev.get('variant')!=name
                or ev.get('ks')!=KS or ev.get('tie_seed')!=2026
                or ev.get('num_queries')!=nquery or ev.get('gallery_count')!=ngallery
                or ev.get('candidate_rule')!='all_gallery_except_query'
                or ev.get('retrieval_protocol_signature')!=catalog.BASE_PROTOCOL_SIGNATURE
                or ev.get('structure_strength')!=2. or ev.get('angular_component') is not False
                or ev.get('local_component') is not False or ev.get('training_required') is not False
                or ev.get('test_parameter_search') is not False or ev.get('posthoc_research') is not True
                or any(ev.get(k)!=v for k,v in ident.items())):
            raise ValueError('Evaluation identity or frozen formula/protocol changed')
        if (checked.get('status')!='verified' or checked.get('signature')!=sig
                or checked.get('split')!=split or checked.get('variant')!=name
                or checked.get('ks')!=KS or checked.get('tie_seed')!=2026
                or checked.get('num_queries')!=nquery or checked.get('ranking_rows')!=nquery*500
                or checked.get('full_candidates_checked')!=nquery*(ngallery-1)
                or checked.get('all_queries_full_candidate_topk_exact') is not True
                or checked.get('independent_fusion_all_candidates') is not True
                or set(checked.get('files',{}))!=set(REPORT_FILES)|{'run_identity.json'}
                or checked.get('ranking_sha256')!=file_hash(folder/'rankings.csv')
                or checked.get('top10_hits')!=row.get('top10_hits')):
            raise ValueError('Full-candidate independent verification is incomplete')
        for key in ('max_metric_absolute_error','max_saved_score_absolute_error','max_component_absolute_error'):
            value=checked.get(key)
            if not isinstance(value,(float,int)) or not 0<=value<=1e-12:
                raise ValueError('Independent verification exceeds tolerance: '+key)
        for filename,expected in checked['files'].items():
            if file_hash(folder/filename)!=expected:
                raise ValueError('Verified report changed: '+filename)
        for family,filename,key in [('exact','metrics.json','metrics'),('coverage','coverage_metrics.json','coverage_metrics')]:
            report=read_json(folder/filename)
            if (report['split']!=split or report['num_queries']!=nquery or report['ks']!=KS
                    or report['metric_version']!=family+'_v1' or report['metrics']!=row[key]
                    or report['metadata']['signature']!=sig):
                raise ValueError('Formal metric report differs from indexed evidence')
        if abs(row['metrics']['Precision@10']-row['top10_hits']/(nquery*10))>1e-12:
            raise ValueError('Top-10 integer evidence does not match precision')
        evidence=['run_manifest.json','completed.json','inputs.json',f'evaluation_index_{split}.json',
                  'baseline_replay.json','source_protection_after.json',f'{split}/{name}/verification.json']
        if (output/f'component_audit_{split}.json').exists():evidence.append(f'component_audit_{split}.json')
        item=dict(key=ident['key'],repository='wbm-region-retrieval',path=_relative(folder,root),
            category='fusion_ablation',method='GJ',variant='GJ/'+name,split=split,
            manifest=_relative(folder/'evaluation_manifest.json',root),marker=_relative(marker,root),
            identity_schema_version=1,identity=_relative(folder/'run_identity.json',root),
            selection=_relative(choice,root),coverage=None,
            supervision='untrained' if name=='J_only' else 'self_supervised',
            display='整图G/J融合 '+name,required_families=['exact','coverage'],protocol='retrieval_v1',posthoc=True,
            supporting_manifests=[_relative(output/p,root) for p in evidence],
            design_note='Eight unique whole-map G/J valid recipes: shared G/J endpoints, three additive weights and three geometric weights, with G*J representing the equivalent alpha=.5 geometric ranking. Fixed J=exp(-2D_H), original G scale, no A/F, no calibration or new training. Integer valid P@10 freezes the overall and each-family winner; test evaluates their deduplicated roles plus fixed G/J/product references. Re-evaluated controls are numerical replay roles, not independent repetitions. Historical test-informed research; test never selects a new winner.')
        catalog.execution_identity(root,item,marker,ev)
        runs.append(item)
    return dict(schema_version=1,runs=runs)


def write_proposal(output,*,root=ROOT):
    value=proposal(output,root=root)
    path=Path(output)/'catalog_sources.json'
    if path.exists():
        if read_json(path)!=value:raise ValueError('Registration proposal changed')
    else:_atomic_write(path,_json_bytes(value))
    return path


def apply_proposal(output, *, workspace=None, root=ROOT):
    """Verify all sources and render in memory, then back up and replace catalogue files."""
    output, root = Path(output).resolve(), Path(root).resolve()
    workspace = root.parent if workspace is None else Path(workspace).resolve()
    if (workspace / "wbm-region-retrieval").resolve() != root:
        raise ValueError("Workspace must contain this wbm-region-retrieval repository")
    proposed = catalog.read(write_proposal(output, root=root))
    folder = root / "results_catalog"
    with _lock(folder):
        original = {name: (folder / name).read_bytes() for name in ("sources.json", "snapshot.json")}
        sources = json.loads(original["sources.json"])
        previous = json.loads(original["snapshot.json"])
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        if previous.get("source_spec_sha256") != fingerprint:
            raise ValueError("Existing catalogue sources and snapshot differ; collect/check before registration")
        old_generated = catalog.render(previous)
        for name, content in old_generated.items():
            path = folder / name
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                raise ValueError(f"Existing generated catalogue is stale: {name}")
            original[name] = path.read_bytes()
        entries = {item["key"]: item for item in sources["runs"]}
        if len(entries) != len(sources["runs"]):
            raise ValueError("Existing catalogue contains duplicate keys")
        additions = []
        for item in proposed["runs"]:
            if item["key"] in entries:
                if entries[item["key"]] != item:
                    raise ValueError(f"Conflicting catalogue key: {item['key']}")
            else:
                additions.append(item)
        if not additions:
            return {"status": "already_registered", "new_results": 0}
        merged = {**sources, "runs": [*sources["runs"], *additions]}
        # Collection verifies every report/identity before any shared data file is changed.
        updated = catalog.collect(workspace, merged, previous)
        old_runs={item['key']:item for item in previous['runs']}
        new_runs={item['key']:item for item in updated['runs']}
        if (len(new_runs)!=len(updated['runs']) or any(new_runs.get(key)!=value for key,value in old_runs.items())):
            raise ValueError('Existing registered result changed during collection')
        generated = catalog.render(updated)
        if any(not (folder / name).is_file() or (folder / name).read_bytes() != content
               for name, content in original.items()):
            raise ValueError("Catalogue changed concurrently during collection")
        contents = {"snapshot.json": _json_bytes(updated),
                    **{name: text.encode("utf-8") for name, text in generated.items()},
                    "sources.json": _json_bytes(merged)}
        if any(name not in original and (folder / name).exists() for name in contents):
            raise ValueError("New generated output would replace an untracked catalogue file")
        before = {name: original.get(name) for name in contents}
        backup_parent = output / "catalog_backup"
        backup_parent.mkdir(exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=datetime.now(TZ).strftime("%Y%m%dT%H%M%S%z") + "_", dir=backup_parent))
        for name, data in before.items():
            if data is not None:
                path = backup / "files" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
        audit = {"schema_version": 1, "status": "prepared", "catalog": str(folder),
                 "new_keys": [item["key"] for item in additions],
                 "before_sha256": {name: hashlib.sha256(data).hexdigest() if data is not None else None
                                   for name, data in before.items()},
                 "after_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}}
        (backup / "manifest.json").write_bytes(_json_bytes(audit))
        written = []
        try:
            for name, content in contents.items():
                if before[name] != ((folder / name).read_bytes() if (folder / name).exists() else None):
                    raise ValueError(f"Catalogue changed concurrently: {name}")
                written.append(name)
                _atomic_write(folder / name, content)
            for name, content in contents.items():
                if (folder / name).read_bytes() != content:
                    raise ValueError(f"Catalogue write verification failed: {name}")
        except BaseException as error:
            failures = []
            for name in reversed(written):
                try:
                    if before[name] is None:
                        (folder / name).unlink(missing_ok=True)
                    else:
                        _atomic_write(folder / name, before[name])
                except BaseException as restore_error:
                    failures.append(f"{name}: {restore_error}")
            audit.update(status="rollback_failed" if failures else "rolled_back", error=str(error), rollback_errors=failures)
            (backup / "manifest.json").write_bytes(_json_bytes(audit))
            if failures:
                raise RuntimeError(f"Catalogue rollback incomplete; backups at {backup}: {failures}") from error
            raise
        audit["status"] = "committed"
        (backup / "manifest.json").write_bytes(_json_bytes(audit))
        return {"status": "registered", "new_results": len(additions), "backup": str(backup)}



def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args(argv)
    output=(ROOT/args.run_root).resolve()
    print(json.dumps(apply_proposal(output) if args.apply else {'proposal':str(write_proposal(output))},ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
