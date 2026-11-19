"""State integrity gates using sealed metadata fixtures, never real retrieval.

The synthetic formal counts exercise provenance/report binding only. Real ranking
and metric computation are independently covered by test_selection_and_evaluation.
"""
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from fgw_grid_search import pipeline, scoring, selection
from fgw_topk_test import state
from results_catalog.build import BASE_PROTOCOL_SIGNATURE
from wbm_retrieval.common import digest, file_hash, read_json, write_json


def snapshot(folder):
    return {str(p.relative_to(folder)): (file_hash(p), p.stat().st_mtime_ns)
            for p in Path(folder).rglob('*') if p.is_file()}


def seal_root(output):
    manifest = read_json(output / 'run_manifest.json')
    write_json(output / 'completed.json', {'status': 'completed', 'version': state.VERSION,
        'stage': 'test', 'signature': manifest['signature'], 'new_evaluations': 10,
        'files': {str(p.relative_to(output)): file_hash(p) for p in output.rglob('*')
                  if p.is_file() and p != output / 'completed.json'}})


def sealed_fixture(root):
    """Ten signed metadata-only reports; a separately mocked verified valid parent."""
    parent = root / 'artificial/fgw_grid_parent'
    output = root / 'artificial/fgw_topk_test_synthetic'
    settings = scoring.validate_settings({})
    all_specs = scoring.specs(settings)
    rows = [{'variant': name, 'family': spec['family'], 'split': 'valid', 'num_queries': 500,
             'precision10_hits': 2500, 'Precision@10': .5, 'mAP@100': .2,
             'Coverage-nDCG@10': .8} for name, spec in all_specs.items()]
    decision = selection.choose_winner(rows, all_specs, settings)
    chosen = state.signed({'version': 'fgw_grid_search_v1', 'parent_signature': 'valid-parent',
        'settings_sha256': digest(settings), 'evaluation_index_sha256': 'valid-index',
        'selected_spec': all_specs[decision['winner']], **decision})
    pcontext = {'source_run': str(root / 'source'), 'settings': settings,
                'code': {'old_scoring.py': 'frozen-code'}, 'environment': {'numpy': 'frozen-numpy'}}
    parent_manifest = state.signed({'version': 'fgw_grid_search_v1', 'stage': 'search', 'context': pcontext})
    write_json(parent / 'run_manifest.json', parent_manifest)
    write_json(parent / 'selection.json', chosen)
    write_json(parent / 'settings.json', settings)
    write_json(parent / 'completed.json', {'status': 'completed'})
    write_json(parent / 'evaluation_index_valid.json', {'status': 'metadata-only'})
    frozen = state.freeze_top10(chosen)
    names, specs = frozen['ranked_variants'], frozen['specs']
    paths = state.evaluation_paths(output, names)
    context = {'stage': 'test', 'split': 'test', 'top_k': 10, 'settings': settings,
        'source_run': pcontext['source_run'], 'specs': specs, 'ranked_variants': names,
        'evaluation_paths': paths, 'frozen_selection_signature': frozen['signature'],
        'primary_variant': frozen['primary_variant'],
        'selection_source': {'root': str(parent), 'files': {p: file_hash(parent / p) for p in state.PARENT_FILES}},
        'grid_code': pcontext['code'], 'environment': pcontext['environment'], 'code': {}, 'provenance': {}}
    manifest = state.signed({'version': state.VERSION, 'stage': 'test', 'context': context,
        'output': str(output), 'batch_id': '20261119T200000+0800__fgw_topk_test',
        'started_at': '2026-11-19T20:00:00+08:00'})
    write_json(output / 'run_manifest.json', manifest)
    write_json(output / 'settings.json', settings)
    write_json(output / 'frozen_selection.json', frozen)
    inp = state.signed({'parent_signature': manifest['signature'], 'split': 'test',
                        'context_digest': digest(context), 'provenance': context['provenance']})
    write_json(output / 'inputs.json', inp)
    write_json(output / 'source_audit.json', state.signed({'status': 'verified', 'context_digest': digest(context)}))
    write_json(output / 'source_protection_before.json', {})
    write_json(output / 'source_protection_after.json', {'status': 'unchanged', 'files': 0, 'inventory_digest': digest({})})
    cache_marker = output / 'component_cache/test/completed/completed.json'
    cache_signature = digest({'inputs': inp['signature'], 'operation': 'full_F_G_W_scores'})
    cache_plan = {'signature': cache_signature, 'split': 'test', 'protocol_signature': BASE_PROTOCOL_SIGNATURE,
                  'query_ids': list(range(7199)), 'gallery_ids': list(range(7199))}
    write_json(output / 'component_cache/test/plan.json', cache_plan)
    write_json(cache_marker, {'signature': cache_signature, 'metadata_only': True,
        'plan_sha256': file_hash(output / 'component_cache/test/plan.json'), 'plan_digest': digest(cache_plan)})
    binding = {'signature': cache_signature, 'completed_sha256': file_hash(cache_marker)}
    identities = {'query_ids_digest': digest(list(range(7199))), 'gallery_ids_digest': digest(list(range(7199)))}
    request = {'signature': inp['signature'], 'stage': 'test', 'split': 'test', 'variants': specs,
        'ranked_variants': names, 'evaluation_paths': paths, 'settings': settings, 'provider_source': binding,
        'retrieval_protocol_signature': BASE_PROTOCOL_SIGNATURE}
    index = {**request, 'children': {}, 'variant_signatures': {}, 'completed_sha256': {}, 'summary_rows': []}
    for name in names:
        spec = specs[name]
        child, folder = state.variant_output(output, name), state.evaluation_folder(output, name)
        signature = pipeline.variant_signature(inp['signature'], name, spec, 'test', settings,
                                                'test', binding, BASE_PROTOCOL_SIGNATURE)
        batch = re.sub(r'[^a-zA-Z0-9_]+', '_', child.name)
        key = 'fgw_grid_' + batch + '_test_' + name
        identity = {'schema_version': 1, 'key': key, 'evaluation_signature': signature, 'variant': name, 'split': 'test',
            'started_at': '2026-11-19T20:00:01+08:00', 'run_id': '20261119T200001+0800__' + key}
        write_json(folder / 'run_identity.json', identity)
        write_json(child / 'identities' / ('test__' + name + '.json'), identity)
        evaluation = {**identity, 'signature': signature, 'parent_signature': inp['signature'], 'stage': 'test',
            'spec': spec, 'settings': settings, 'ks': settings['ks'], 'tie_seed': settings['tie_seed'],
            'provider_source': binding, 'num_queries': 7199, 'gallery_count': 7199,
            'candidate_rule': 'all_gallery_except_query', 'retrieval_protocol_signature': BASE_PROTOCOL_SIGNATURE,
            'scoring_device': 'cpu', 'training_required': False, 'posthoc_research': True, 'test_parameter_search': False}
        write_json(folder / 'evaluation_manifest.json', evaluation)
        metadata = {'signature': signature, 'retrieval_protocol_signature': BASE_PROTOCOL_SIGNATURE}
        report = {'split': 'test', 'num_queries': 7199, 'ks': settings['ks'], 'metadata': metadata}
        write_json(folder / 'metrics.json', {**report, 'metric_version': 'exact_v1',
                                            'metrics': {'Precision@10': .5, 'mAP@100': .2}})
        write_json(folder / 'coverage_metrics.json', {**report, 'metric_version': 'coverage_v1',
                                                     'metrics': {'Coverage-nDCG@10': .8}})
        (folder / 'per_query.csv').write_text('query_id,Precision@10\n' + ''.join(f'{q},0.5\n' for q in range(7199)))
        for name_ in ('rankings.csv','per_class.csv','coverage_per_query.csv','coverage_per_class.csv'):
            (folder / name_).write_text('synthetic metadata fixture only\n')
        verification = {'status': 'verified', 'signature': signature, 'provider_source': binding,
            'num_queries': 7199, 'ks': settings['ks'], 'ranking_rows': 7199*500,
            'full_candidates_checked': 7199*7198, 'all_queries_full_candidate_topk_exact': True,
            'ranking_sha256': file_hash(folder / 'rankings.csv'),
            'files': {p: file_hash(folder / p) for p in pipeline.REPORT_FILES}}
        write_json(folder / 'verification.json', verification)
        write_json(folder / 'timing.json', {})
        write_json(folder / 'completed.json', {'status': 'completed', 'signature': signature,
            'files': {p: file_hash(folder / p) for p in (*pipeline.REPORT_FILES,'verification.json','timing.json')}})
        summary = {'variant': name, 'family': spec['family'], 'split': 'test', 'num_queries': 7199,
            'precision10_hits': 7199*5, 'Precision@10': .5, 'mAP@100': .2, 'Coverage-nDCG@10': .8}
        child_request = {k:v for k,v in request.items() if k not in ('ranked_variants','evaluation_paths')}
        child_request.update(identities)
        child_request['variants'] = {name: spec}
        write_json(child / 'evaluation_request_test.json', child_request)
        child_index = {**child_request, 'variant_signatures': {name: signature},
            'completed_sha256': {name: file_hash(folder / 'completed.json')}, 'summary_rows': [summary]}
        child_index_path = child / 'evaluation_index_test.json'
        write_json(child_index_path, child_index)
        index['children'][name] = {'path': str(child.relative_to(output)),
            'index': str(child_index_path.relative_to(output)), 'sha256': file_hash(child_index_path)}
        index['variant_signatures'][name] = signature
        index['completed_sha256'][name] = file_hash(folder / 'completed.json')
        index['summary_rows'].append(summary)
    write_json(output / 'evaluation_request_test.json', request)
    write_json(output / 'evaluation_index_test.json', index)
    write_json(output / 'status.json', {'status': 'completed', 'stage': 'test', 'test_evaluated': True, 'new_evaluations': 10})
    seal_root(output)
    provider = SimpleNamespace(signature=cache_signature, marker=cache_marker, split='test',
        protocol_signature=BASE_PROTOCOL_SIGNATURE, query_ids=np.arange(7199), gallery_ids=np.arange(7199))
    return output, parent_manifest, chosen, provider


class StateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.output, self.parent_manifest, self.chosen, self.provider = sealed_fixture(self.root)
        self.names = read_json(self.output / 'frozen_selection.json')['ranked_variants']
        self.parent_patch = patch('fgw_grid_search.state.verify_run', return_value=(self.parent_manifest,self.chosen))
        self.parent_verify = self.parent_patch.start()
    def tearDown(self): self.parent_patch.stop(); self.temp.cleanup()
    def verify(self, check_cache=False): return state.verify_run(self.output, check_cache=check_cache)
    def update(self, path, transform, *, resign=False):
        value = read_json(path)
        if resign: value.pop('signature')
        transform(value)
        write_json(path, state.signed(value) if resign else value)
    def rebind_child(self, name, *, report_changed=False):
        child, folder = state.variant_output(self.output,name), state.evaluation_folder(self.output,name)
        if report_changed:
            self.update(folder/'verification.json',lambda v: v.update(files={p:file_hash(folder/p) for p in pipeline.REPORT_FILES},
                                                                     ranking_sha256=file_hash(folder/'rankings.csv')))
        self.update(folder/'completed.json',lambda v: v.update(files={p:file_hash(folder/p) for p in v['files']}))
        sha=file_hash(folder/'completed.json')
        self.update(child/'evaluation_index_test.json',lambda v: v.update(completed_sha256={name:sha}))
        def update_root(v):
            v['completed_sha256'][name]=sha
            v['children'][name]['sha256']=file_hash(child/'evaluation_index_test.json')
        self.update(self.output/'evaluation_index_test.json',update_root)
        seal_root(self.output)
    def test_complete_verify_is_read_only_and_returns_frozen_valid_selection(self):
        before=snapshot(self.output)
        manifest,frozen=self.verify()
        self.assertEqual(frozen['valid_selection'],self.chosen)
        self.assertEqual(frozen['primary_variant'],self.chosen['winner'])
        self.assertEqual(list(manifest['context']['specs']),self.names)
        self.assertEqual(snapshot(self.output),before)
        self.parent_verify.assert_called_once_with(Path(manifest['context']['selection_source']['root']),
                                                   required_stage='search',check_cache=False)
    def test_complete_cache_is_bound_to_exact_input_split_protocol_and_ids(self):
        with patch('fgw_grid_search.cache.FullScoreCache',return_value=self.provider):
            self.verify(check_cache=True)
            for field,value in [('signature','other'),('split','valid'),('protocol_signature','other'),
                                ('query_ids',np.arange(7198))]:
                original=getattr(self.provider,field);setattr(self.provider,field,value)
                with self.subTest(field=field), self.assertRaises(ValueError): self.verify(check_cache=True)
                setattr(self.provider,field,original)
    def test_missing_frozen_selection_is_rejected(self):
        (self.output/'frozen_selection.json').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)): self.verify()
    def test_rebound_frozen_prefix_reordering_is_rejected(self):
        self.update(self.output/'frozen_selection.json',lambda v:v['ranked_variants'].reverse(),resign=True)
        seal_root(self.output)
        with self.assertRaisesRegex(ValueError,'Frozen top ten'): self.verify()
    def test_parent_selection_file_change_is_rejected_before_parent_recheck(self):
        parent=Path(read_json(self.output/'run_manifest.json')['context']['selection_source']['root'])
        (parent/'selection.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'valid source changed'): self.verify()
        self.parent_verify.assert_not_called()
    def test_missing_parent_provenance_file_is_rejected(self):
        self.update(self.output/'run_manifest.json',lambda v:v['context']['selection_source']['files'].pop('settings.json'),resign=True)
        seal_root(self.output)
        with self.assertRaisesRegex(ValueError,'Incomplete original'): self.verify()
    def test_source_code_environment_and_test_selection_plan_changes_rejected(self):
        path=self.output/'run_manifest.json'; original=read_json(path)
        for field,value in [('source_run','other'),('grid_code',{}),('environment',{}),('top_k',11),
                            ('specs',dict(reversed(list(original['context']['specs'].items()))))]:
            changed=deepcopy(original);changed.pop('signature');changed['context'][field]=value
            write_json(path,state.signed(changed));seal_root(self.output)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'Test plan/source/code/environment'): self.verify()
        write_json(path,original)
    def test_protocol_change_in_aggregate_request_or_index_is_rejected(self):
        path=self.output/'evaluation_index_test.json'
        self.update(path,lambda v:v.update(retrieval_protocol_signature='old_2000x1000'))
        seal_root(self.output)
        with self.assertRaises(ValueError): self.verify()
    def test_child_query_digest_cannot_differ_from_aggregate(self):
        name=self.names[0];child=state.variant_output(self.output,name)
        self.update(child/'evaluation_index_test.json',lambda v:v.update(query_ids_digest='different'))
        self.rebind_child(name)
        with self.assertRaises(ValueError): self.verify()
    def test_child_request_cannot_differ_from_its_completed_index(self):
        name=self.names[0];child=state.variant_output(self.output,name)
        self.update(child/'evaluation_request_test.json',lambda v:v.update(gallery_ids_digest='different'))
        seal_root(self.output)
        with self.assertRaises(ValueError): self.verify()
    def test_missing_required_report_is_rejected(self):
        (state.evaluation_folder(self.output,self.names[0])/'coverage_per_class.csv').unlink()
        with self.assertRaises((ValueError,FileNotFoundError)): self.verify()
    def test_rebound_metric_change_disagrees_with_summary(self):
        name=self.names[0];folder=state.evaluation_folder(self.output,name)
        self.update(folder/'metrics.json',lambda v:v['metrics'].update(**{'Precision@10':.6}))
        self.rebind_child(name,report_changed=True)
        with self.assertRaisesRegex(ValueError,'summary differs'): self.verify()
    def test_rebound_partial_candidate_verification_is_rejected(self):
        name=self.names[0];folder=state.evaluation_folder(self.output,name)
        self.update(folder/'verification.json',lambda v:v.update(full_candidates_checked=7199*500))
        self.rebind_child(name)
        with self.assertRaisesRegex(ValueError,'independent all-query'): self.verify()
    def test_new_test_selection_or_extra_variant_is_rejected(self):
        (self.output/'selection.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'fresh selection'): self.verify()
        (self.output/'selection.json').unlink()
        (self.output/'evaluations'/'extra_recipe').mkdir()
        with self.assertRaisesRegex(ValueError,'extra test child'): self.verify()
    def test_incomplete_status_or_completion_is_not_repaired(self):
        self.update(self.output/'status.json',lambda v:v.update(status='running'))
        seal_root(self.output);before=snapshot(self.output)
        with self.assertRaisesRegex(ValueError,'incomplete'): self.verify()
        self.assertEqual(snapshot(self.output),before)
    def test_source_protection_summary_must_match_snapshot(self):
        self.update(self.output/'source_protection_after.json',lambda v:v.update(inventory_digest='changed'))
        seal_root(self.output)
        with self.assertRaisesRegex(ValueError,'source protection'): self.verify()


class OutputIsolationTests(unittest.TestCase):
    def test_new_output_must_be_separate_and_explicit_resume_preserves_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve();now=datetime(2026,11,19,20,0,0,tzinfo=state.TZ)
            target=state.choose_output(now=now,root=root)
            self.assertEqual(target,root/'artificial/fgw_topk_test_20261119T200000+0800')
            write_json(target/'run_manifest.json',{'signature':'placeholder'})
            with self.assertRaises(FileExistsError): state.choose_output(output_root=target,root=root)
            self.assertEqual(state.choose_output(resume_root=target,root=root),target)
            for bad in [root/'outside',target/'fgw_topk_test_nested']:
                with self.assertRaises(ValueError): state.choose_output(output_root=bad,root=root)
            with self.assertRaises(ValueError):state.separate(target,[target.parent])


if __name__ == '__main__': unittest.main()
