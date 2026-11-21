"""Global-score provenance, exact copy and held-out-input isolation checks."""
from pathlib import Path
import ast
import inspect
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f_region_wbbs import inputs, compat
from f_region_structure.state import signed, record, seal
from wbm_retrieval.common import digest, file_hash, read_json, write_json


class Pool:
    split_name = 'valid'
    protocol_signature = 'frozen-protocol'
    def __init__(self, queries=(10,20), gallery=(10,20,30,40)):
        self.queries=np.asarray(queries,dtype=np.int64);self.gallery=np.asarray(gallery,dtype=np.int64)
        self.candidates_per_query=len(self.gallery)-1
    def __iter__(self):return iter(self.queries)
    def __len__(self):return len(self.queries)
    def __getitem__(self,q):return self.gallery[self.gallery!=q]


def provider_fixture(tmp, pool=None):
    pool=pool or Pool()
    matrix=np.random.default_rng(2026).random((len(pool),len(pool.gallery)))
    for row,q in enumerate(pool):matrix[row,pool.gallery==q]=np.nan
    folder=Path(tmp)/'historical';folder.mkdir()
    marker=folder/'completed.json';record(marker,{'status':'completed'})
    cache=SimpleNamespace(split='valid',signature='original-cache',protocol_signature=pool.protocol_signature,
        query_ids=pool.queries,gallery_ids=pool.gallery,marker=marker,matrices={'WBBS':matrix})
    receipt={'status':'verified','source_cache':str(folder),'source_scores_sha256':'frozen'}
    return inputs.GlobalWBBSProvider(cache,pool,receipt),pool


class GlobalProviderTests(unittest.TestCase):
    def test_subset_reordered_candidates_preserve_exact_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider,pool=provider_fixture(tmp)
            ids=np.array([40,20],dtype=np.int64)
            np.testing.assert_array_equal(provider.scores(10,ids),provider.matrix[0,[3,1]])
            ids,values=provider.full_row(10)
            np.testing.assert_array_equal(ids,pool[10])
            self.assertEqual(values.tobytes(),provider.matrix[0,1:].tobytes())
            for q,c in [(True,[20]),(99,[20]),(10,[10]),(10,[20,20]),(10,[99]),(10,[20.])]:
                with self.assertRaises(ValueError):provider.scores(q,np.asarray(c))

    def test_pool_split_order_protocol_and_eligibility_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider,pool=provider_fixture(tmp)
            bad=Pool(queries=(20,10))
            with self.assertRaises(ValueError):inputs.GlobalWBBSProvider(provider.cache,bad,{})
            for field,value in [('split_name','test'),('protocol_signature','changed'),('candidates_per_query',2)]:
                bad=Pool();setattr(bad,field,value)
                with self.assertRaises(ValueError):inputs.GlobalWBBSProvider(provider.cache,bad,{})
            class BadPool(Pool):
                def __getitem__(self,q):return super().__getitem__(q)[::-1]
            with self.assertRaises(ValueError):inputs.GlobalWBBSProvider(provider.cache,BadPool(),{})

    def test_export_immutable_nonself_copy_self_only_conversion_and_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider,pool=provider_fixture(tmp)
            before=provider.matrix.tobytes()
            output=Path(tmp)/'new'
            copied=inputs.export_global(output,provider,pool,'context')
            self.assertFalse(copied.matrix.flags.writeable)
            for q in pool:
                self.assertEqual(copied.scores(q,pool[q]).tobytes(),provider.scores(q,pool[q]).tobytes())
                self.assertEqual(copied.matrix[copied.query_positions[int(q)],copied.gallery_positions[int(q)]],0.)
            self.assertEqual(provider.matrix.tobytes(),before)
            receipt=read_json(output/'global_wbbs_copy_valid.json')
            self.assertEqual(receipt['full_nonself_pairs_checked'],6)
            self.assertTrue(receipt['nonself_scores_bitwise_identical'])
            old_sha=file_hash(copied.marker)
            reused=inputs.export_global(output,provider,pool,'context')
            self.assertEqual(file_hash(reused.marker),old_sha)
            with self.assertRaises(ValueError):inputs.export_global(output,provider,pool,'changed')
            provider.source_receipt['source_scores_sha256']='different'
            with self.assertRaises(ValueError):inputs.export_global(output,provider,pool,'context')

    def test_export_rejects_incomplete_target_corrupt_copy_and_wrong_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            provider,pool=provider_fixture(tmp)
            bad=Pool();bad.split_name='test'
            with self.assertRaises(ValueError):inputs.export_global(Path(tmp)/'bad',provider,bad,'context')
            output=Path(tmp)/'new';(output/'score_cache/valid/global_wbbs').mkdir(parents=True)
            with self.assertRaises(ValueError):inputs.export_global(output,provider,pool,'context')
            output=Path(tmp)/'other';copied=inputs.export_global(output,provider,pool,'context')
            file=copied.marker.parent/'scores.npy'
            with file.open('r+b') as f:
                f.seek(-1,2);old=f.read(1);f.seek(-1,2);f.write(bytes([old[0]^1]))
            with self.assertRaises(ValueError):inputs.export_global(output,provider,pool,'context')


class SourceBindingTests(unittest.TestCase):
    def fixture(self,tmp):
        root=Path(tmp);folder=root/'component_cache/valid';complete=folder/'completed';complete.mkdir(parents=True)
        pool=Pool(queries=range(500),gallery=range(3599))
        code='fgw_fusion/wbbs_backend.py'
        context={'split':'valid','stage':'search','provenance':{'source':'frozen'},'code':{code:file_hash(inputs.ROOT/code)}}
        manifest=signed({'version':'fgw_grid_search_v1','stage':'search','context':context})
        inp=signed({'split':'valid','parent_signature':manifest['signature'],'context_digest':digest(context),'provenance':context['provenance']})
        signature=digest({'inputs':inp['signature'],'operation':'full_F_G_W_scores'})
        record(root/'run_manifest.json',manifest);record(root/'inputs.json',inp)
        record(folder/'plan.json',{'signature':signature})
        for name in ['WBBS.npy','query_ids.npy','gallery_ids.npy']:(complete/name).write_bytes(name.encode())
        files={p.name:file_hash(p) for p in complete.glob('*.npy')}
        record(complete/'completed.json',{'status':'completed','signature':signature,'files':files})
        root_files={name:file_hash(root/name) for name in ['run_manifest.json','inputs.json','component_cache/valid/plan.json','component_cache/valid/completed/completed.json']}
        record(root/'completed.json',{'status':'completed','version':'fgw_grid_search_v1','signature':manifest['signature'],'stage':'search','files':root_files})
        cache=SimpleNamespace(split='valid',signature=signature,protocol_signature=pool.protocol_signature,
            query_ids=pool.queries,gallery_ids=pool.gallery,marker=complete/'completed.json',
            matrices={'WBBS':np.zeros((500,3599))},completion={'files':files})
        return root,pool,cache

    def test_signed_global_source_chain_and_all_final_file_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,pool,cache=self.fixture(tmp);binding={}
            with patch.object(inputs,'FullScoreCache',return_value=cache):
                provider,receipt=inputs.global_valid_source(pool,binding,root)
            self.assertEqual(receipt['nonself_pairs'],1799000)
            self.assertEqual(provider.signature,cache.signature)
            self.assertEqual(len(binding),9)
            self.assertIn(str((root/'component_cache/valid/completed/WBBS.npy').resolve()),binding)
            # FullScoreCache independently handles matrix bytes/NaN semantics;
            # this loader additionally requires the old root -> inputs -> cache signature chain.
            cache.signature='detached'
            with patch.object(inputs,'FullScoreCache',return_value=cache):
                with self.assertRaises(ValueError):inputs.global_valid_source(pool,{},root)

    def test_missing_root_binding_changed_inputs_and_code_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,pool,cache=self.fixture(tmp)
            path=root/'completed.json';saved=read_json(path)
            value={**saved,'files':dict(saved['files'])};value['files'].pop('component_cache/valid/plan.json')
            write_json(path,value)
            with patch.object(inputs,'FullScoreCache',return_value=cache):
                with self.assertRaises(ValueError):inputs.global_valid_source(pool,{},root)
            write_json(path,saved)
            p=root/'inputs.json';p.write_text(p.read_text()+' ')
            with patch.object(inputs,'FullScoreCache',return_value=cache):
                with self.assertRaises(ValueError):inputs.global_valid_source(pool,{},root)

    def test_two_legacy_cross_metadata_extensions_and_child_signatures(self):
        with tempfile.TemporaryDirectory() as tmp:
            parents = [Path(tmp)/'regional', Path(tmp)/'global']
            for parent, variant, spec in [
                (parents[0], inputs.SOURCE_VARIANT, {'family': 'add', 'alpha': .4, 'beta': .25}),
                (parents[1], 'WBBS', {'formula': 'WBBS'})]:
                folder=parent/'test'/variant;folder.mkdir(parents=True)
                context={'source':str(parent)};cs=digest(context)
                signature=digest({'context_signature':cs,'variant':variant,'spec':spec})
                child={'context_signature':cs,'split':'test','signature':signature,'variant':variant,'spec':spec,
                    'candidate_rule':inputs.RULE,'protocol':inputs.PROTOCOL,
                    'retrieval_protocol_signature':inputs.FROZEN['test']['derived_signature'],
                    'base_protocol_signature':inputs.BASE_PROTOCOL_SIGNATURE,'ks':inputs.KS,'tie_seed':2026,
                    'num_queries':7199,'gallery_count':7199}
                record(folder/'evaluation_manifest.json',child)
                record(folder/'coverage_metrics.json',{'metadata':{**child,'candidates_scored':'full_filtered_pool','smoke':False}})
                seal(folder,signature)
                record(parent/'run_manifest.json',signed({'context':context}))
                seal(parent,cs)
            with patch.object(inputs,'CROSS_J_SOURCE',parents[0]),patch.object(inputs,'CROSS_GLOBAL_SOURCE',parents[1]):
                binding={};refs=inputs._reference_folders('test',binding)
            self.assertEqual(set(refs),{'region_j_cross','global_wbbs_cross'})
            self.assertEqual(len(binding),10)

    def test_valid_does_not_access_test_references_and_invalid_split_rejected(self):
        with patch.object(inputs,'read_signed',side_effect=AssertionError('must not read test')):
            self.assertEqual(inputs._reference_folders('valid',{}),{})
        with patch.object(inputs.area_inputs,'load_valid',side_effect=AssertionError('invalid split')):
            for split in ('train','smoke','Test'):
                with self.assertRaises(ValueError):inputs.load(split)


class AdministrativeSourceCompatibilityTests(unittest.TestCase):
    def fixture(self, tmp):
        root = Path(tmp)
        current = root/compat.HISTORICAL_PATH
        archived = root/compat.ARCHIVED_PATH
        current.parent.mkdir(parents=True);archived.parent.mkdir(parents=True)
        current.write_bytes((compat.ROOT/compat.HISTORICAL_PATH).read_bytes())
        archived.write_bytes((compat.ROOT/compat.ARCHIVED_PATH).read_bytes())
        return root, current, archived

    def test_archived_original_bytes_and_truthful_two_file_binding(self):
        self.assertEqual(file_hash(compat.ROOT/compat.ARCHIVED_PATH), compat.HISTORICAL_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            root,current,archived = self.fixture(tmp)
            with patch.object(compat, 'ROOT', root):
                legacy_binding = {'unchanged-old-source': 'original-sha'}
                before = dict(legacy_binding)
                receipts = compat.validate_code({compat.HISTORICAL_PATH: compat.HISTORICAL_SHA256})
                self.assertEqual(legacy_binding, before)
                self.assertEqual(len(receipts), 1)
                row = receipts[0]
                self.assertEqual(set(row), {'historical_path','expected_sha256','archived_path',
                    'archived_sha256','current_path','current_sha256','scope'})
                self.assertEqual(row['scope'], 'catalog administration only')
                self.assertNotEqual(row['archived_sha256'], row['current_sha256'])
                final_binding = dict(legacy_binding)
                compat.bind_compatibility(receipts, final_binding)
                self.assertEqual(legacy_binding, before)
                self.assertEqual(final_binding[str(current.resolve())], file_hash(current))
                self.assertEqual(final_binding[str(archived.resolve())], file_hash(archived))
                self.assertEqual(len(final_binding), len(before)+2)
                self.assertEqual(compat.unique_receipts(receipts+receipts), receipts)

    def test_science_hash_unknown_revision_and_noncanonical_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,current,archived = self.fixture(tmp)
            science = root/'science.py';science.write_text('frozen scientific bytes\n')
            with patch.object(compat, 'ROOT', root):
                path, rows = compat.resolve_source('science.py', file_hash(science))
                self.assertEqual(path, science);self.assertEqual(rows, [])
                for name, expected in [('science.py', compat.HISTORICAL_SHA256),
                        (compat.HISTORICAL_PATH, 'not-the-allowlisted-old-hash'),
                        ('../results_catalog/build.py', compat.HISTORICAL_SHA256),
                        ('./results_catalog/build.py', compat.HISTORICAL_SHA256),
                        (str(current.resolve()), compat.HISTORICAL_SHA256)]:
                    with self.assertRaises(ValueError):compat.resolve_source(name, expected)
                current.write_text('unapproved administrative revision\n')
                with self.assertRaises(ValueError):
                    compat.resolve_source(compat.HISTORICAL_PATH, compat.HISTORICAL_SHA256)

    def test_corrupt_archive_and_mutated_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,current,archived = self.fixture(tmp)
            with patch.object(compat, 'ROOT', root):
                rows = compat.validate_code({compat.HISTORICAL_PATH: compat.HISTORICAL_SHA256})
                bad = [{**rows[0], 'scope': 'pretend all canonical code unchanged'}]
                with self.assertRaises(ValueError):compat.bind_compatibility(bad, {})
                with self.assertRaises(ValueError):compat.unique_receipts(rows+bad)
                archived.write_bytes(archived.read_bytes()+b'\n')
                with self.assertRaises(ValueError):
                    compat.resolve_source(compat.HISTORICAL_PATH, compat.HISTORICAL_SHA256)

    def test_canonical_unchanged_file_needs_no_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root,current,archived = self.fixture(tmp)
            current.write_bytes(archived.read_bytes())
            with patch.object(compat, 'ROOT', root):
                binding = {}
                rows = compat.validate_code({compat.HISTORICAL_PATH: compat.HISTORICAL_SHA256}, binding)
                self.assertEqual(rows, [])
                self.assertEqual(binding, {str(current.resolve()): compat.HISTORICAL_SHA256})

    def test_full_legacy_gates_preserved_except_explicit_source_resolution(self):
        # Compare complete function ASTs to ensure this exceptional adapter does
        # not drop a legacy seal, selection, metrics, cache or binding-equality gate.
        old = inspect.getsource(inputs.area_inputs.frozen_choice)
        old = old.replace('from .register import proposal', 'from f_region_area_fusion.register import proposal')
        old = old.replace("    for name, expected_hash in context['code'].items():\n"
            "        if file_hash(ROOT/name) != expected_hash:\n"
            "            raise ValueError('Scientific code changed since valid: '+name)\n",
            "    source_compatibility = validate_code(context['code'])\n")
        old = old.replace('return manifest, wrapper, binding',
                          'return manifest, wrapper, binding, source_compatibility')
        self.assertEqual(ast.dump(ast.parse(old)), ast.dump(ast.parse(inspect.getsource(compat.frozen_choice))))
        old = inspect.getsource(inputs.area_inputs.load_test)
        old = old.replace('manifest, choice, binding = frozen_choice(source)',
            'manifest, choice, binding, source_compatibility = frozen_choice(source)')
        old = old.replace('baseline_folder=baseline)',
                          'baseline_folder=baseline, source_compatibility=source_compatibility)')
        self.assertEqual(ast.dump(ast.parse(old)), ast.dump(ast.parse(inspect.getsource(compat.load_test))))


if __name__=='__main__':unittest.main()
