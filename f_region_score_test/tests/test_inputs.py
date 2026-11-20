"""Selection cannot be retuned and test tokens must retain their original IDs."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from f_region_score_test import inputs
from f_region_structure.state import seal, signed
from wbm_retrieval.common import digest, write_json
from wbm_retrieval.retrieval_protocol import FullGalleryPool


def cache_fixture():
    ids = np.arange(4,14402,2,dtype=np.int64)
    counts = np.zeros(7199,dtype=np.int64)
    counts[[0,3]]= [1,2]
    offsets = np.r_[0,np.cumsum(counts)]
    h = np.zeros((3,512),dtype=np.float32); h[:,0]=1
    joint = np.full((3,128),1/128)
    radial = np.full((3,16),1/16)
    data = {'map_ids':ids,'offsets':offsets,'area':np.full(3,5,dtype=np.int64),
            'original_count':np.full(3,5,dtype=np.int64),'retained_count':np.full(3,5,dtype=np.int64),
            'owners':np.repeat(ids,counts),'features':h,'histograms':radial,'radial_hist16':radial.copy(),
            'relative_area':np.full(3,.05),'quality':np.ones(3)}
    for suffix in ('','_all'):
        data['joint'+suffix]=joint.copy()
        data['shape'+suffix]=np.full((3,16),1/16)
        data['empty'+suffix]=np.zeros(3,dtype=bool)
    protocol=SimpleNamespace(ids={'test':ids},retrieval=SimpleNamespace(
        test=FullGalleryPool(ids,ids,'test','fixture')))
    return data,protocol


class TestCacheGuards(unittest.TestCase):
    def test_noncontiguous_raw_ids_empty_maps_and_valid_tokens(self):
        data,protocol=cache_fixture()
        inputs._check_cache(data,protocol)

    def test_rejects_shifted_ids_subset_query_and_wrong_split(self):
        data,protocol=cache_fixture()
        for pool in (FullGalleryPool(data['map_ids'][:500],data['map_ids'],'test','fixture'),
                     FullGalleryPool(data['map_ids'],data['map_ids'],'valid','fixture')):
            other=deepcopy(protocol);other.retrieval.test=pool
            with self.assertRaises(ValueError):inputs._check_cache(data,other)
        data['map_ids']=data['map_ids']+1
        with self.assertRaises(ValueError):inputs._check_cache(data,protocol)

    def test_rejects_owner_area_padding_and_feature_mismatch(self):
        for name,mutate in [
            ('owners',lambda x:x.__setitem__(0,123)),
            ('area',lambda x:x.__setitem__(0,4)),
            ('offsets',lambda x:x.__setitem__(1,5)),
            ('features',lambda x:x.__setitem__((0,0),2.)),
            ('relative_area',lambda x:x.__setitem__(0,0.)),
            ('joint_all',lambda x:x.__setitem__((0,0),np.nan)),
            ('shape_all',lambda x:x.__setitem__((0,0),.5)),
            ('empty_all',lambda x:x.__setitem__(0,True)),
            ('quality',lambda x:x.__setitem__(0,.5))]:
            with self.subTest(name=name):
                data,protocol=cache_fixture();mutate(data[name])
                with self.assertRaises(ValueError):inputs._check_cache(data,protocol)


class FrozenSelectionGuards(unittest.TestCase):
    def fixture(self,p,variant=None,spec=None):
        manifest={'context':{'code':{}},'source_run':'old-cache-source'}
        write_json(p/'run_manifest.json',signed(manifest))
        selected={'winner':variant or inputs.VARIANT,'selected_spec':spec or inputs.SPEC}
        write_json(p/'selection.json',signed(selected))
        write_json(p/'evaluation_index_valid.json',{'fixture':True})
        seal(p,digest(manifest['context']))

    def test_preserves_exact_signed_selection_and_binds_completion(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(inputs.valid_register,'proposal',return_value={'runs':[{}]*15}):
            p=Path(tmp);self.fixture(p)
            _,choice,binding=inputs.frozen_choice(p)
            self.assertEqual(choice['winner'],inputs.VARIANT)
            self.assertEqual(choice['selection']['selected_spec'],inputs.SPEC)
            self.assertIn(str((p/'completed.json').resolve()),binding)
            self.assertIn(str((p/'selection.json').resolve()),binding)

    def test_rejects_a_different_valid_winner_even_if_sealed(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(inputs.valid_register,'proposal',return_value={'runs':[{}]*15}):
            p=Path(tmp);self.fixture(p,variant='product')
            with self.assertRaises(ValueError):inputs.frozen_choice(p)

    def test_rejects_retuned_weights_or_incomplete_grid(self):
        spec={**inputs.SPEC,'alpha':.75}
        with tempfile.TemporaryDirectory() as tmp,patch.object(inputs.valid_register,'proposal',return_value={'runs':[{}]*15}):
            p=Path(tmp);self.fixture(p,spec=spec)
            with self.assertRaises(ValueError):inputs.frozen_choice(p)
        with tempfile.TemporaryDirectory() as tmp,patch.object(inputs.valid_register,'proposal',return_value={'runs':[{}]}):
            p=Path(tmp);self.fixture(p)
            with self.assertRaises(ValueError):inputs.frozen_choice(p)

    def test_rejects_report_tampering_and_missing_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.fixture(p)
            write_json(p/'evaluation_index_valid.json',{'changed':True})
            with self.assertRaises(ValueError):inputs.frozen_choice(p)
            (p/'completed.json').unlink()
            with self.assertRaises(ValueError):inputs.frozen_choice(p)


if __name__=='__main__':unittest.main()
