from pathlib import Path
import tempfile
import unittest
from f_region_wbbs import run, state
from wbm_retrieval.common import file_hash, write_json, read_json


def valid_fixture(output, ctx):
    results = None
    for stage in ('valid', 'cross_valid'):
        results = {}
        for name in state.evaluation_specs(stage):
            folder = output/stage/name
            write_json(folder/'coverage_metrics.json', {'metrics': {'Coverage-nDCG@10': .75}})
            state.seal(folder, state.evaluation_signature(ctx,stage,name))
            results[name] = {'coverage_metrics': {'Coverage-nDCG@10': .75},
                             'completed_sha256': file_hash(folder/'completed.json')}
        write_json(output/f'evaluation_index_{stage}.json', {'context_signature': ctx,
            'stage':stage,'split':'valid','specs':state.evaluation_specs(stage),'results':results})
    return results


class FrozenGateTests(unittest.TestCase):
    def test_complete_cross_valid_is_mandatory_and_freeze_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            rows = valid_fixture(folder,'ctx')
            chosen = run.freeze(folder,'ctx',rows)
            self.assertEqual(run.freeze(folder,'ctx',rows),chosen)
            self.assertEqual(run.frozen_selection(folder,'ctx'),chosen)
            with self.assertRaises(ValueError):run.frozen_selection(folder,'other')
            path=folder/'cross_valid/global_wbbs/coverage_metrics.json'
            write_json(path,{'changed':True})
            with self.assertRaises(ValueError):run.frozen_selection(folder,'ctx')

    def test_self_consistent_reseal_cannot_change_valid_after_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            rows=valid_fixture(folder,'ctx')
            run.freeze(folder,'ctx',rows)
            name='denoised_wbbs_a06'
            index=read_json(folder/'evaluation_index_cross_valid.json')
            index['results'][name]['coverage_metrics']['Coverage-nDCG@10']=.99
            write_json(folder/'evaluation_index_cross_valid.json',index)
            with self.assertRaises(ValueError):run.frozen_selection(folder,'ctx')
            with self.assertRaises(ValueError):run.freeze(folder,'ctx',index['results'])

    def test_incomplete_selection_cannot_release_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            rows=valid_fixture(folder,'ctx')
            run.freeze(folder,'ctx',rows)
            (folder/'valid_completed.json').unlink()
            with self.assertRaises(FileNotFoundError):run.frozen_selection(folder,'ctx')


if __name__=='__main__':unittest.main()
